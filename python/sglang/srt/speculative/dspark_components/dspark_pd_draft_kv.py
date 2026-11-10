"""Decode-side draft context KV for a PD-transferred prompt
(--speculative-dspark-pd-draft-kv-source random).

The prefill server hosts no draft, so no draft KV crosses the wire and the two
roles need not share a draft KV layout. The decode server fills the prompt's
draft KV rows from a fixed bank of random target hidden states; verify commits
store real rows after them, so the draft context turns real as tokens commit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ContextManager, Optional

import torch

from sglang.srt.runtime_context import get_disagg, get_spec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
        TargetHiddenKvInjector,
    )

PD_DRAFT_KV_SOURCE_RANDOM = "random"

# Arbitrary; the bank only has to decorrelate neighbouring positions.
_BANK_ROWS = 1024
# Bounds the projection's transient memory; arbitrary, not tuned.
_SEED_CHUNK_TOKENS = 4096


def is_pd_draft_kv_seeded_on_decode(
    *,
    speculative_algorithm: Optional[str],
    disaggregation_mode: str,
    draft_kv_source: str,
) -> bool:
    return (
        speculative_algorithm is not None
        and speculative_algorithm.upper() == "DSPARK"
        and disaggregation_mode in ("prefill", "decode")
        and draft_kv_source == PD_DRAFT_KV_SOURCE_RANDOM
    )


def pd_draft_kv_seeded_on_decode() -> bool:
    spec = get_spec()
    return is_pd_draft_kv_seeded_on_decode(
        speculative_algorithm=spec.speculative_algorithm,
        disaggregation_mode=get_disagg().disaggregation_mode,
        draft_kv_source=spec.speculative_dspark_pd_draft_kv_source,
    )


def build_hidden_bank(
    *, feature_size: int, seed: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """[_BANK_ROWS, feature_size]; a CPU draw, so every rank of an attention TP
    group holds the same rows for its own head shard to project."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bank = torch.randn((_BANK_ROWS, feature_size), generator=generator)
    return bank.to(device=device, dtype=dtype)


class RandomDraftKVSeeder:
    def __init__(
        self,
        *,
        kv_injector: TargetHiddenKvInjector,
        draft_context: Callable[[], ContextManager],
        feature_size: int,
        seed: int,
        dtype: torch.dtype,
        device,
    ) -> None:
        if kv_injector.swa_ring_size is not None:
            raise ValueError(
                "--speculative-dspark-pd-draft-kv-source random does not support a "
                "draft whose sliding-window KV is a per-request ring."
            )
        self._kv_injector = kv_injector
        self._draft_context = draft_context
        self._bank = build_hidden_bank(
            feature_size=feature_size, seed=seed, dtype=dtype, device=device
        )
        self.num_seed_tokens = 0

    def seed(self, batch: ScheduleBatch) -> None:
        """Write draft KV for the rows a prebuilt batch received from prefill."""
        if not batch.extend_lens or sum(batch.extend_lens) == 0:
            return
        placement = self._kv_injector.place_prefill(batch)
        positions = placement.positions.to(torch.int64)
        num_tokens = positions.shape[0]
        # A position always maps to the same bank row, so a prefix shared through
        # the decode radix cache is seeded identically by every request.
        bank_rows = positions % _BANK_ROWS
        with self._draft_context():
            for start in range(0, num_tokens, _SEED_CHUNK_TOKENS):
                token_indices = torch.arange(
                    start,
                    min(start + _SEED_CHUNK_TOKENS, num_tokens),
                    device=positions.device,
                )
                self._kv_injector.inject_prefill(
                    placement,
                    target_hidden=self._bank.index_select(
                        0, bank_rows.index_select(0, token_indices)
                    ),
                    token_indices=token_indices,
                    target_hidden_is_projected=False,
                )
        self.num_seed_tokens += num_tokens
