# gemma3-270m-finetuned-finance-instructor

A fine-tuning project that adapts Google's Gemma 3 (270M parameters) model for finance-specific instruction following using LoRA (Low-Rank Adaptation) and the TRL library.

## Overview

This project demonstrates how to fine-tune a small language model on financial instruction data to create a specialized finance assistant. The implementation uses Parameter Efficient Fine-Tuning (PEFT) with LoRA to reduce computational requirements while maintaining model performance.

## Features

- **Base Model**: Google Gemma 3 (270M parameters)
- **Dataset**: Finance-Instruct-500k dataset with financial Q&A pairs
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) for efficient training
- **Training Framework**: Hugging Face Transformers + TRL's SFTTrainer
- **Memory Optimization**: Support for 4-bit quantization and mixed precision training

## Project Structure

```
├── main.ipynb          # Main training notebook with all implementation
├── README.md           # This file
├── LICENSE             # Project license
└── .gitignore          # Git ignore rules
```

## Requirements

The project requires the following Python packages (installed in [main.ipynb](main.ipynb)):

```bash
pip install transformers datasets peft accelerate trl bitsandbytes
```

## Quick Start

1. **Setup Environment**: Open [main.ipynb](main.ipynb) and run the first cell to install dependencies
2. **Authentication**: Login to Hugging Face Hub (required for model access)
3. **Run Training**: Execute all cells sequentially to complete the fine-tuning process
4. **Test Model**: The final cells demonstrate inference with the fine-tuned model

## Implementation Details

### Model Configuration

- **Base Model**: `google/gemma-3-270m` (loaded via `model_id` variable)
- **LoRA Config**: r=16, alpha=32, targeting q_proj and v_proj modules
- **Max Sequence Length**: 512 tokens

### Dataset Processing

- **Source**: Josephgflowers/Finance-Instruct-500k dataset
- **Format**: Converts system/user/assistant format to unified text format
- **Training Split**: 50k samples for faster iteration
- **Evaluation Split**: 5k samples

### Training Configuration

- **Batch Size**: 4 per device with 4 gradient accumulation steps (effective batch size: 16)
- **Learning Rate**: 1e-4 with 1000 warmup steps
- **Epochs**: 3 full passes through the dataset
- **Optimization**: AdamW with fp16 mixed precision
- **Checkpointing**: Every 5000 steps, keeping 2 latest checkpoints

## Key Functions and Variables

The [main.ipynb](main.ipynb) notebook contains several important components:

- `format_example()`: Converts dataset entries to training format
- `tokenize_fn()`: Tokenizes text with proper truncation and padding
- `lora_config`: LoRA configuration for efficient fine-tuning
- `training_args`: Training hyperparameters and settings
- `trainer`: SFTTrainer instance for supervised fine-tuning

## Usage Example

After training, the model can be used for financial question answering:

```python
from transformers import pipeline

agent = pipeline(
    "text-generation",
    model="./gemma3-finance-agent",
    tokenizer=tokenizer,
    device=0
)

response = agent(
    "Explain the effects of increasing interest rates on bond prices.",
    max_new_tokens=200
)
print(response[0]["generated_text"])
```

## Model Output

The fine-tuned model is saved locally as `./gemma3-finance-agent` and can generate responses to financial queries with improved domain-specific knowledge compared to the base model.

## Hardware Requirements

- **GPU**: Recommended for training (the notebook includes Google Colab integration)
- **Memory**: ~8GB GPU memory (with mixed precision and efficient batch sizing)
- **Storage**: ~2GB for model checkpoints and datasets

## Customization

To adapt this project for your needs:

1. **Dataset**: Replace the dataset loading in [main.ipynb](main.ipynb) with your own financial dataset
2. **Model Size**: Change `model_id` to use larger Gemma variants (e.g., gemma-3-2b, gemma-3-9b)
3. **LoRA Config**: Adjust rank (r) and alpha values in `lora_config` for different adaptation strengths
4. **Training**: Modify `training_args` for different batch sizes, learning rates, or training duration

## Contributing

This is a demonstration project. Feel free to fork and modify for your specific use cases. Key areas for improvement:

- Experiment with different LoRA configurations
- Add evaluation metrics for financial accuracy
- Implement proper validation splits
- Add support for multi-turn conversations

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- Google for the Gemma 3 model family
- Hugging Face for the transformers and PEFT libraries
- The creators of the Finance-Instruct-500k dataset
