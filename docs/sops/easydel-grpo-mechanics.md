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
