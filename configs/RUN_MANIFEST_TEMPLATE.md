# Run Manifest

## Identity
- run_name:
- date_utc:
- git_commit:
- seed:
- notes:

## Models
- base_model:
- chat_model:
- provider:
- api_base:

## Data
- train_path:
- test_path:
- train_size:
- test_size:

## ICM Hyperparameters
- alpha:
- initial_temperature:
- final_temperature:
- cooling_beta:
- num_seed_K:
- max_iterations:
- consistency_fix: disabled (required by assignment)

## Evaluation Setup
- conditions:
  - zero_shot_base
  - zero_shot_chat
  - golden_label_icl_base
  - icm_label_icl_base
- metric: accuracy
- runs_per_condition:

## Cost + Runtime
- estimated_cost_usd:
- actual_cost_usd:
- wall_clock_time:
