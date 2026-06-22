# Task 3: Text Summarization with Phi-2 on XSum

## Purpose

This repository contains the implementation of abstractive text summarization using Microsoft Phi-2 fine-tuned on the XSum dataset. The goal is to demonstrate how a large decoder-only language model can be efficiently fine-tuned using LoRA (Low-Rank Adaptation) and 4-bit quantization to generate one-sentence summaries of BBC news articles.

## Project Overview

Phi-2 is a decoder-only language model with 2.7 billion parameters. Unlike encoder-decoder models such as T5, it generates text auto-regressively by predicting the next token. Summarization is framed as an instruction-following task using the following prompt format:

```
Article: {news article text}

Summarize the article above in one sentence:
{model generates summary here}
```

Due to the large model size, two memory-saving techniques are applied:
- 4-bit NF4 quantization via BitsAndBytes, reducing VRAM from ~10 GB to ~5 GB
- LoRA fine-tuning, which trains only ~0.5% of total parameters (~13M out of 2.7B)

The notebook covers dataset exploration, model loading with quantization, LoRA configuration, instruction-format preprocessing, causal LM fine-tuning, ROUGE evaluation, and inference demo.

## Models Used

| Model | Architecture | Parameters | Trainable (LoRA) |
|-------|-------------|------------|-----------------|
| microsoft/phi-2 | Decoder-only Transformer | 2.7B | ~13M (0.5%) |

## Dataset

| Split | Samples Used | Total Available |
|-------|-------------|-----------------|
| Train | 1,000 | 204,045 |
| Validation | 200 | 11,332 |
| Test | -- | 11,334 |

Dataset: `EdinburghNLP/xsum` (BBC news articles with single-sentence summaries)

XSum is considered a highly abstractive dataset — summaries are not extracted spans but are written independently, making it significantly harder than extractive benchmarks.

## Results

| Metric | Expected Range |
|--------|---------------|
| ROUGE-1 | 25-35% |
| ROUGE-2 | 8-15% |
| ROUGE-L | 20-28% |

Note: XSum is a difficult benchmark. Even state-of-the-art models score approximately 45% ROUGE-1. Results vary depending on training data size and number of epochs.

### LoRA vs Full Fine-tuning Comparison

| Method | Trainable Params | VRAM Required | Performance |
|--------|-----------------|---------------|-------------|
| Full fine-tune | 2.7B | 10+ GB | Best |
| LoRA (r=16) | ~13M (0.5%) | ~5-6 GB | Near-full |
| LoRA (r=8) | ~7M (0.3%) | ~4-5 GB | Good |

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 1-2 |
| Learning Rate | 2e-4 |
| Batch Size (train) | 4-8 |
| Gradient Accumulation Steps | 2-4 |
| Effective Batch Size | 16 |
| Max Length | 256-512 tokens |
| Warmup Steps | 50 |
| Weight Decay | 0.01 |
| Optimizer | paged_adamw_8bit |
| Quantization | 4-bit NF4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Target Modules | q_proj, k_proj, v_proj, dense |
| Precision | bfloat16 |
| Gradient Checkpointing | True |

## How to Navigate

```
Task3_Phi2_XSum_FIXED.ipynb
|
|-- Step 1: Install Dependencies
|-- Step 2: Load & Explore XSum
|   |-- Article and summary length distributions
|   |-- Compression ratio analysis
|
|-- Step 3: Load Phi-2 with 4-bit Quantization
|   |-- BitsAndBytesConfig (NF4, bfloat16)
|   |-- prepare_model_for_kbit_training
|
|-- Step 4: LoRA Configuration
|   |-- LoraConfig (r=16, alpha=32)
|   |-- get_peft_model
|   |-- Trainable parameter summary
|
|-- Step 5: Preprocessing - Instruction Format
|   |-- make_prompt() function
|   |-- preprocess_xsum() tokenization
|   |-- Subset selection
|
|-- Step 6: Training
|   |-- DataCollatorForLanguageModeling (mlm=False)
|   |-- TrainingArguments
|   |-- Trainer (processing_class=tokenizer)
|   |-- Training loop
|
|-- Step 7: Evaluation with ROUGE
|   |-- generate_summary() function
|   |-- ROUGE-1, ROUGE-2, ROUGE-L scores
|   |-- Visualizations (ROUGE bars, length scatter, training curve)
|
|-- Step 8: Inference Demo
|   |-- XSum validation samples
|   |-- Custom article summarization
|
|-- Step 9: Save Model & Metrics
```

## Requirements

```
transformers
datasets
evaluate
accelerate
peft
bitsandbytes
rouge_score
torch
```

## Notes

- The `generate_summary()` function uses `max_new_tokens` (not `max_length`) since Phi-2 is a causal LM and does not have a conflicting default `max_length` like T5.
- `tokenizer` is passed as `processing_class=tokenizer` in the `Trainer` constructor, as the `tokenizer` argument was deprecated in newer versions of the transformers library.
- `warmup_ratio` is deprecated in transformers v5.2+; use `warmup_steps` instead.
- Labels are not manually set in preprocessing; `DataCollatorForLanguageModeling(mlm=False)` handles label creation automatically from `input_ids`.
- To load the saved LoRA model after training:
  ```python
  from peft import PeftModel
  base = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', ...)
  model = PeftModel.from_pretrained(base, './phi2-xsum-lora')
  ```
