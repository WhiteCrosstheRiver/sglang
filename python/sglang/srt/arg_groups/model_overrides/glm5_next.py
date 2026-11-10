"""Config-time override declarations for glm5_next."""

from sglang.srt.arg_groups.model_override_base import register_dspark_dp_a2a_backends

# GLM-5-Next's expert parallelism runs on DeepEP and has no MegaMoE path, so an
# EP + DP-attention decode server can only pair DSpark with DeepEP.
register_dspark_dp_a2a_backends("Glm5NextForConditionalGeneration", ("none", "deepep"))
