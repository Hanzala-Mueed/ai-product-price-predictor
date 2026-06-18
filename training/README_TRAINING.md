# Fine-Tuning Training Notes

This project uses LoRA fine-tuning for the local Llama 3.2 model.

## Base model

Training script base model:

```text
unsloth/Llama-3.2-3B-Instruct-bnb-4bit
```

## Important GPU Requirement

LoRA fine-tuning with Unsloth requires a GPU environment.

If you run this script on a local CPU-only Windows machine, you may see:

```text
Unsloth cannot find any torch accelerator? You need a GPU.
```