import contextlib
import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    PrefillPlacement,
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_pd_draft_kv import (
    _SEED_CHUNK_TOKENS,
    RandomDraftKVSeeder,
    build_hidden_bank,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _routed(pool):
    """A stand-in pool whose layers its own plain router hands out."""
    pool.start_layer, pool.layer_transfer_counter = 0, None
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


class _Attention:
    num_kv_heads = 1
    head_dim = 3

    def __init__(self) -> None:
        self.attn = SimpleNamespace(layer_id=0, k_scale=0.5, v_scale=0.25)
        self.input = None

    def kv_proj_only(self, hidden_states: torch.Tensor):
        self.input = hidden_states
        return hidden_states + 1, hidden_states + 2

    def apply_k_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 3

    def apply_k_rope(
        self, _positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states + 4


class DSparkTargetHiddenProjectionTest(CustomTestCase):
    def test_nonfinal_prefill_stage_only_forwards_target_proxies(self) -> None:
        proxy = object()
        target = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=object(), spec_algorithm=None),
            device="cpu",
            forward_batch_generation=lambda batch, *, pp_proxy_tensors, capture_hidden_mode: (
                SimpleNamespace(pp_hidden_states_proxy_tensors=pp_proxy_tensors)
            ),
        )
        with (
            get_parallel().override(pp_group=SimpleNamespace(is_last_rank=False)),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=1),
            ),
        ):
            worker = DSparkWorkerV2(None, 0, 0, target)
        worker.alloc_memory_pool()
        worker.init_attention_backends()
        worker.init_cuda_graphs()
        batch = SimpleNamespace(seq_lens=torch.tensor([8]))
        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)
        self.assertIs(result.pp_hidden_states_proxy_tensors, proxy)
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertIsNone(worker.get_confidence_budget_prepare())
        self.assertIsNone(worker.primary_draft_kv_pool)
        self.assertEqual(worker.preloaded_weights_bytes, 0)
        self.assertEqual(
            worker.spec_v2_attn_backends, (target.model_runner.attn_backend,)
        )

    def test_decode_seeded_pd_prefill_runs_target_only(self) -> None:
        """The last-stage scheduler dereferences next_draft_input, so a draftless
        last PP stage must still return one."""
        next_token_ids = torch.tensor([7])
        logits_output = SimpleNamespace(
            hidden_states=torch.zeros(8, 4),
            hidden_states_token_indices=torch.arange(8),
        )
        target = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=object(), spec_algorithm=None),
            device="cpu",
            forward_batch_generation=lambda batch, *, pp_proxy_tensors, capture_hidden_mode: (
                SimpleNamespace(
                    next_token_ids=next_token_ids, logits_output=logits_output
                )
            ),
        )
        with published_topology(
            speculative_algorithm="DSPARK",
            disaggregation_mode="prefill",
            speculative_dspark_pd_draft_kv_source="random",
        ):
            with (
                get_parallel().override(pp_group=SimpleNamespace(is_last_rank=True)),
                mock.patch(
                    "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_schedule",
                    return_value=SimpleNamespace(page_size=1),
                ),
                mock.patch(
                    "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_dp_tp_group",
                    return_value=SimpleNamespace(world_size=1, rank_in_group=0),
                ),
            ):
                worker = DSparkWorkerV2(None, 0, 0, target)
        worker.alloc_memory_pool()
        worker.init_attention_backends()
        worker.init_cuda_graphs()
        batch = SimpleNamespace(seq_lens=torch.tensor([8]))
        result = worker.forward_batch_generation(batch)
        self.assertIs(result.next_draft_input.bonus_tokens, next_token_ids)
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertIsNone(logits_output.hidden_states)
        self.assertIsNone(worker.pd_transfer_draft_kv_pool)
        self.assertEqual(worker.weight_update_runners(), [])

    def test_decode_seeds_each_received_row_once_keyed_by_position(self) -> None:
        """Chunking must cover every row exactly once, and a position must map to
        the same hidden row in every request: a radix-shared prefix is seeded by
        each request that reuses it."""
        placement = PrefillPlacement(
            cache_loc=torch.arange(100, 100 + 2 * _SEED_CHUNK_TOKENS + 5),
            positions=torch.cat(
                [torch.arange(_SEED_CHUNK_TOKENS + 5), torch.arange(_SEED_CHUNK_TOKENS)]
            ),
            state_slot=None,
            final_pos=None,
            swa_loc=None,
        )
        writes = []

        def inject_prefill(
            placement, *, target_hidden, token_indices, target_hidden_is_projected
        ):
            self.assertFalse(target_hidden_is_projected)
            rows = placement.select(token_indices)
            writes.append((rows.cache_loc, rows.positions, target_hidden))

        injector = SimpleNamespace(
            swa_ring_size=None,
            place_prefill=lambda batch: placement,
            inject_prefill=inject_prefill,
        )
        seeder = RandomDraftKVSeeder(
            kv_injector=injector,
            draft_context=contextlib.nullcontext,
            feature_size=6,
            seed=0,
            dtype=torch.float32,
            device="cpu",
        )
        seeder.seed(
            SimpleNamespace(extend_lens=[_SEED_CHUNK_TOKENS + 5, _SEED_CHUNK_TOKENS])
        )

        self.assertEqual(len(writes), 3)
        cache_loc = torch.cat([w[0] for w in writes])
        positions = torch.cat([w[1] for w in writes])
        hidden = torch.cat([w[2] for w in writes])
        self.assertTrue(torch.equal(cache_loc, placement.cache_loc))
        first, second = (
            hidden[: _SEED_CHUNK_TOKENS + 5],
            hidden[_SEED_CHUNK_TOKENS + 5 :],
        )
        self.assertTrue(torch.equal(first[:_SEED_CHUNK_TOKENS], second))
        self.assertFalse(torch.equal(hidden[0], hidden[1]))
        # Attention TP ranks project their head shards from the same rows.
        peer_bank = build_hidden_bank(
            feature_size=6, seed=0, dtype=torch.float32, device="cpu"
        )
        self.assertTrue(torch.equal(hidden, peer_bank[positions % peer_bank.shape[0]]))

        seeder.seed(SimpleNamespace(extend_lens=[0]))
        self.assertEqual(len(writes), 3)

    def test_preprojected_hidden_is_not_projected_again(self) -> None:
        attention = _Attention()
        draft_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
            project_target_hidden=mock.Mock(
                side_effect=AssertionError("projection must not run twice")
            ),
            _fused_kv_write_bundle=lambda _pool: None,
            _stacked_ctx_kv_params=lambda: None,
        )
        draft_model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, draft_model
        )
        pool = _routed(SimpleNamespace(set_kv_buffer=mock.Mock()))
        injector = TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=SimpleNamespace(
                token_to_kv_pool=pool,
                runner_ids=SimpleNamespace(),
            ),
            model_runner=SimpleNamespace(device=torch.device("cpu")),
            device=torch.device("cpu"),
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )
        projected_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        injector.inject_target_hidden(
            target_hidden=projected_hidden,
            cache_loc=torch.arange(2),
            positions=torch.arange(2),
            target_hidden_is_projected=True,
        )

        draft_model.project_target_hidden.assert_not_called()
        self.assertIs(attention.input, projected_hidden)


if __name__ == "__main__":
    unittest.main()
