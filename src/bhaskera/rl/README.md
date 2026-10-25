# bhaskera.rl: post-training RL (GRPO)

Improves a LoRA-adapted language model from a **verifiable reward** instead of from labelled
completions. The model writes several answers per prompt, a plain function checks each one, and
the model is updated to make the better answers more likely.

```
bhaskera-rl --config configs/rl_gsm8k.yaml
```

## One training step

1. Take `prompts_per_step` prompts from the dataset.
2. Sample `group_size` completions for each (plain HuggingFace `generate`, temperature sampling).
3. Score every completion with the reward function (`rewards.py`).
4. Advantage = the completion's reward relative to its own group: `(r - mean) / (std + eps)`.
   A group where every answer scored the same teaches nothing and is skipped.
5. Optimizer step on the clipped GRPO loss: PPO's clipped ratio objective, plus `kl_coef` times a
   KL penalty that keeps the model close to the base model.

No value network and no reward model are needed. The base model stays frozen; only the LoRA
adapter trains. The KL reference is the same network with the adapter switched off, so it costs
one extra forward pass and no extra memory.

## Files

| File | Role |
|---|---|
| `grpo.py` | The math: group advantages, completion logprobs, the loss. Small pure functions. |
| `rollout.py` | Sampling completions; marks which tokens are real (up to and including EOS). |
| `rewards.py` | Reward registry and the built-in `math_boxed` reward. |
| `data.py` | JSONL prompts, chat-template formatting, a seeded resumable prompt stream. |
| `trainer.py` | `GRPOTrainer`: one step, held-out evaluation. |
| `../launcher/rl_train.py` | The `bhaskera-rl` CLI and `run(cfg)`. |

## Data and rewards

Prompts are JSONL, one object per line: `{"prompt": "...", "answer": "..."}`.
`scripts/prepare_gsm8k_rl.py` builds GSM8K files in this format.

The built-in reward `math_boxed` gives 1.0 when the last `\boxed{...}` in the completion equals the
gold answer (exact fraction comparison, so `1,000`, `$12.50`, `\frac{1}{2}` all work) and 0.0
otherwise. A missing box scores 0, which also teaches the format. For GSM8K the gold answer is the
number after `####`.

Add a task by registering a function. Rewards must be deterministic and must not run model output:

```python
from bhaskera.rl.rewards import register_reward

@register_reward("my_task")
def my_task(completion: str, answer: str) -> float:
    return 1.0 if completion.strip() == answer else 0.0
```

then set `rl.reward: my_task` and make sure the module is imported before training.

## Configuration

Everything under `rl:` in the YAML (defaults in `config.py`, class `RLConfig`). See
`configs/rl_gsm8k.yaml` for a complete file. Learning rate, steps and seed come from the usual
`training:` section; the model and adapter from `model:` and `lora:` (`lora.enabled: true` is
required).

| Key | Default | Meaning |
|---|---|---|
| `train_path`, `eval_path` | none | Prompt files. `eval_path` is optional and enables held-out evaluation. |
| `reward` | `math_boxed` | Reward function name. |
| `group_size` | 8 | Completions per prompt. At least 2. |
| `prompts_per_step` | 8 | Prompts per optimizer step (rollouts per step = this x group_size). |
| `max_new_tokens`, `temperature` | 256, 1.0 | Sampling. Sampling uses the full softmax (no top-k/top-p) so it matches the trained distribution. |
| `kl_coef` | 0.04 | KL penalty weight. `0` disables it and skips the reference pass. |
| `clip_eps` | 0.2 | PPO ratio clipping. |
| `epochs_per_rollout` | 1 | Optimizer steps per batch of rollouts. |
| `micro_batch_size` | 8 | Sequences per forward/backward; lower it if you run out of memory. |
| `eval_every`, `eval_max_prompts` | 0, 200 | Greedy held-out evaluation cadence (0 = start and end only). |

## Output

One line per step on stdout, in the same prefix format as SFT logs:

```
[epoch 0][step 7] loss=0.000803 reward=0.3828 kl=0.02002 clip_frac=0.0000 grad_norm=0.1597 completion_len=245.3 samples=896 tokens=241596
[eval][step 25] eval_reward=0.4150 examples=200
```

`reward` is the mean training reward of the step (sampled at `temperature`); `eval_reward` is greedy
accuracy on the held-out file. Checkpoints go to `checkpoint.save_dir/step_NNNNNNN/` in the same
layout SFT uses, but contain only the LoRA weights (`save_checkpoint(..., lora_only=True)`).

## Limits

* Single GPU. `--num-workers` other than 1 is ignored with a warning.
* Generation is plain HuggingFace `generate`: correct and simple, not fast. A dedicated rollout
  engine can replace `rollout.generate_rollouts` without touching the rest.
* Experiment trackers (W&B, MLflow) are not wired in; progress goes to stdout.
* Rewards that execute generated code are not provided; they would need a sandbox.

## Tests

```
pytest src/bhaskera/rl/tests -v
```

CPU only, about 20 seconds. They use a tiny real Llama and check the loss against hand-computed
values, that left-padded scoring equals unpadded scoring, that the policy learns a verifiable
task, that the base weights and the KL reference never move, and the checkpoint/resume contract
used by Slakshna.
