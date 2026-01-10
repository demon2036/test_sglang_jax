# EasyDeL GRPO Mechanics SOPs

- **Title**: SOP: Inspect GRPO rollout + scheduling flow in EasyDeL
  **Prereqs**: local repo at `/home/john/test_sglang_jax/easydel`
  **Steps**:
  - `rg -n "GRPO" easydel`
  - `sed -n '1,260p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `sed -n '260,620p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `sed -n '620,1020p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `rg -n "def generate_unified" -n easydel/easydel/trainers`
  - `sed -n '1080,1320p' easydel/easydel/trainers/base_trainer.py`
  - `sed -n '200,360p' easydel/easydel/trainers/prompt_transforms.py`
  - `sed -n '1,240p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_config.py`
  - `rg -n "_preprocess_batch_input" -n easydel/easydel/trainers`
  - `sed -n '600,780p' easydel/easydel/trainers/trainer/trainer.py`
  - `rg -n "def minibatch_call|def make_assertions_and_get_sizes" easydel/easydel/trainers/training_utils.py`
  - `sed -n '180,360p' easydel/easydel/trainers/training_utils.py`
  - `sed -n '1,200p' easydel/docs/trainers/grpo.md`
  **Expected Result**: Identify GRPO rollout (generate_unified), reward + advantage computation in `GRPOTrainer._preprocess_batch_input`, and loss/optimization details in `group_relative_policy_optimization/_fn.py`, with step order in `trainer/trainer.py`.
  **Troubleshooting**: N/A
  **References**:
  - `easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `easydel/easydel/trainers/group_relative_policy_optimization/_fn.py`
  - `easydel/easydel/trainers/base_trainer.py`
  - `easydel/easydel/trainers/prompt_transforms.py`
  - `easydel/easydel/trainers/group_relative_policy_optimization/grpo_config.py`
  - `easydel/easydel/trainers/trainer/trainer.py`
  - `easydel/easydel/trainers/training_utils.py`
  - `easydel/docs/trainers/grpo.md`

- **Title**: SOP: Confirm EasyDeL training attention kernels + Ray launcher (GRPO context)
  **Prereqs**: local repo at `/home/john/test_sglang_jax/easydel`
  **Steps**:
  - Training attention kernel selection (FlashAttn2 vs Splash/BlockSparse):
    - `sed -n '260,380p' easydel/easydel/layers/attention.py`
    - `sed -n '1,260p' easydel/easydel/layers/operations/modules/flash_attention.py`
    - `sed -n '1,220p' easydel/easydel/layers/operations/modules/blocksparse_attention.py`
    - `sed -n '340,520p' easydel/easydel/infra/base_config.py`
  - GRPO rollout path (eSurge vs compiled) and default toggle:
    - `sed -n '340,420p' easydel/easydel/trainers/training_configurations.py`
    - `sed -n '1112,1320p' easydel/easydel/trainers/base_trainer.py`
    - `sed -n '360,720p' easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - Ray usage pattern in EasyDeL (job launcher, not per-step rollout workers):
    - `sed -n '1,220p' easydel/tutorials/post-training/group_relative_policy_optimization/launch.py`
    - `sed -n '160,260p' easydel/docs/trainers/ray_distributed_trainer.md`
  **Expected Result**:
  - `attn_mechanism="auto"` maps to TPU v4+ -> `blocksparse` (SplashAttention), TPU v3 -> `flash_attn2` (see `get_optimal_config` in `easydel/easydel/layers/attention.py`).
  - `flash_attn2` is a training-capable kernel via `ejkernel.modules.flash_attention` (TPU Pallas / GPU Triton), while `blocksparse` is SplashAttention via `ejkernel.modules.blocksparse_attention`.
  - GRPO rollouts call `generate_unified`; unless overridden, it uses eSurge because `use_esurge_generation=True` by default (see `easydel/easydel/trainers/training_configurations.py`).
  - Ray is used as an external launcher in tutorials (`@execute(tpu_config) @ray.remote`); `RayDistributedTrainer` explicitly does not directly interact with Ray.
  **Troubleshooting**: N/A
  **References**:
  - `easydel/easydel/layers/attention.py`
  - `easydel/easydel/layers/operations/modules/flash_attention.py`
  - `easydel/easydel/layers/operations/modules/blocksparse_attention.py`
  - `easydel/easydel/infra/base_config.py`
  - `easydel/easydel/trainers/training_configurations.py`
  - `easydel/easydel/trainers/base_trainer.py`
  - `easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
  - `easydel/tutorials/post-training/group_relative_policy_optimization/launch.py`
  - `easydel/docs/trainers/ray_distributed_trainer.md`

- **Title**: SOP: Diagnose eSurge multi-host mismatch during GRPO rollouts
  **Prereqs**: local repo at `/home/john/test_sglang_jax/easydel`; multi-host GRPO run that triggers eSurge device_put mismatch
  **Steps**:
  - `rg -n "req_num_tokens_full_buf|_empty_sharding|device_put" easydel/easydel/inference/esurge -g"*.py"`
  - `sed -n '1100,1195p' easydel/easydel/inference/esurge/runners/model_runner.py`
  - `rg -n "scheduler_thread|_scheduler_loop|async" easydel/easydel/inference/esurge/esurge_engine.py`
  - `sed -n '680,850p' easydel/easydel/inference/esurge/esurge_engine.py`
  - `rg -n "process_index\\(\\)|process_count\\(\\)|multihost" easydel/easydel/inference/esurge -g"*.py"`
  **Expected Result**:
  - `ModelRunner._execute_model_impl` uses `jax.device_put(req_num_tokens_np, self._empty_sharding)` where `_empty_sharding` is a fully replicated `NamedSharding`, so multi-host requires identical host arrays.
  - eSurge scheduling runs per-process in a background thread (`esurge_engine.py`), with no multihost synchronization or `process_index` logic.
  - Multi-host GRPO can hit `AssertionError` from `jax.experimental.multihost_utils.assert_equal` (array mismatch across processes), matching the error seen during rollout.
  **Troubleshooting**:
  - For multi-host GRPO, disable eSurge generation (`use_esurge_generation=False`) until eSurge adds cross-host synchronization.
  **References**:
  - `easydel/easydel/inference/esurge/runners/model_runner.py`
  - `easydel/easydel/inference/esurge/esurge_engine.py`
