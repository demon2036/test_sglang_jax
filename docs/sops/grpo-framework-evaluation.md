# GRPO Framework Evaluation SOPs

- **Title**: SOP: Compare EasyDeL vs Tunix for GRPO training readiness
  **Prereqs**: Local repo at `/home/john/test_sglang_jax`
  **Steps**:
  - `rg -n "GRPO|grpo" -S easydel tunix | head -n 50`
  - `sed -n '1,200p' easydel/docs/trainers/grpo.md`
  - `sed -n '1,260p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `sed -n '1,240p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_config.py`
  - `rg -n "def generate_unified" -n easydel/easydel/trainers`
  - `sed -n '1080,1300p' easydel/easydel/trainers/base_trainer.py`
  - Multi-host + kernel backend checks (important for TRC / TPU slices):
    - `sed -n '1,220p' easydel/easydel/trainers/utils.py`
    - `sed -n '60,110p' sglang-jax/python/sgl_jax/srt/server_args.py`
    - `sed -n '440,520p' sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
    - `sed -n '520,720p' sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
    - `sed -n '1,260p' sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py`
  - `sed -n '1,240p' tunix/README.md`
  - `sed -n '1,260p' tunix/tunix/rl/grpo/grpo_learner.py`
  - `sed -n '1,260p' tunix/tunix/rl/rollout/sglang_jax_rollout.py`
  - `sed -n '1,260p' tunix/tunix/generate/sglang_jax_sampler.py`
  - `rg -n "sglang" tunix/scripts/grpo_demo_llama3_qwen2.py`
  - `sed -n '1000,1080p' tunix/scripts/grpo_demo_llama3_qwen2.py`
  **Expected Result**: Identify GRPO trainer maturity, rollout integration path,
  sglang-jax weight sync approach, and whether multi-host TPU rollouts are
  realistically supported by the current wiring.
  **Troubleshooting**: N/A
  **References**:
  - `easydel/docs/trainers/grpo.md`
  - `easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `easydel/easydel/trainers/group_relative_policy_optimization/grpo_config.py`
  - `easydel/easydel/trainers/base_trainer.py`
  - `easydel/easydel/trainers/utils.py`
  - `tunix/README.md`
  - `tunix/tunix/rl/grpo/grpo_learner.py`
  - `tunix/tunix/rl/rollout/sglang_jax_rollout.py`
  - `tunix/tunix/generate/sglang_jax_sampler.py`
  - `tunix/scripts/grpo_demo_llama3_qwen2.py`
  - `sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
  - `sglang-jax/python/sgl_jax/srt/server_args.py`
  - `sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
  - `sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py`
