# rlhf with ppo on small llms using trl and lora

end-to-end reinforcement learning from human feedback pipeline using proximal policy optimization on small language models. built with huggingface trl and peft for efficient training on consumer hardware including apple silicon.

## project goal

implement a complete rlhf pipeline on small llms:
1. supervised fine-tuning (sft)
2. reward model training using bradley-terry model
3. ppo-based policy optimization with lora

target model: qwen/qwen2.5-1.5b-instruct (or meta-llama/llama-3.2-1b-instruct)

## technical pipeline

| stage | script                        | description                                      |
|-------|-------------------------------|--------------------------------------------------|
| 1     | supervised_finetuning.py      | supervised fine-tuning on instruction data       |
| 2     | training_reward_model.py      | train bradley-terry reward model on preferences  |
| 3     | tuning_lm_with_rl.py          | ppo alignment using learned reward model         |

## key technical components

- **ppo trainer** from trl with clipped surrogate objective, generalized advantage estimation (gae), and adaptive kl penalty control
- **bradley-terry reward modeling** for pairwise preference learning
- **lora / peft** for parameter-efficient fine-tuning (low memory footprint)
- **trl library** (huggingface) for stable rlhf implementation
- **apple silicon optimization** using pytorch mps backend

## model and hardware

- base model: qwen2.5-1.5b-instruct
- fine-tuning method: lora (rank 16-32)
- hardware target: apple m4 air (unified memory, mps backend)
- batch size: 1 with gradient accumulation for memory efficiency

## installation

```bash
pip install -r requirements.txt
