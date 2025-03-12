# LADDER Framework

## Learning through Autonomous Difficulty-Driven Example Recursion

LADDER is a framework for self-improving language models through recursive problem decomposition, focusing on mathematical integration tasks but designed to be extensible to other domains.

## Overview

The LADDER framework enables language models to improve their capabilities through a structured approach to problem-solving:

1. Given a complex problem, decompose it into progressively simpler variants
2. Use these variants to construct a natural difficulty gradient for reinforcement learning
3. Apply test-time adaptation techniques to enhance performance on new problems

## Key Components

### 1. Variant Generation Subsystem (VGS)

Generates a hierarchical tree of problem variants, starting from complex problems and recursively creating simpler versions using the model's own capabilities. Implements temperature cycling and persona-based prompting to enhance diversity.

### 2. Verification Subsystem (VS) 

Provides ground-truth verification of solutions using numerical integration methods. Handles both definite and indefinite integrals with robust error handling for singularities and edge cases.

### 3. Reinforcement Learning Subsystem (RLS)

Implements Group Relative Policy Optimization (GRPO) for fine-tuning the model. Processes variant trees to create natural curriculum learning that improves performance on integration tasks.

### 4. Test-Time Reinforcement Learning Subsystem (TTRL)

Enables on-the-fly adaptation to specific test problems through focused variant training. Maintains a parameter cache for similar problems to enhance performance during inference.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- SciPy and NumPy for numerical computations
- SymPy for symbolic mathematics

### Installation

```bash
# Clone the repository
git clone https://github.com/Gajesh2007/ladder-framework.git
cd ladder-framework

# Install dependencies
pip install -r requirements.txt
```

### NVIDIA-Specific Setup

For optimal performance on NVIDIA GPUs:

```bash
# Install CUDA-specific bitsandbytes version
pip install bitsandbytes>=0.42.0

# For faster inference with Tensor Cores
pip install nvidia-tensorrt

# Optional: Install Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

To verify your setup is using the GPU correctly:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Usage

### Running the Full Pipeline

```bash
python main.py pipeline --input problems.txt --model mistralai/Mistral-7B-v0.1 --output-dir ./outputs --eval-problems test_problems.txt
```

### Individual Components

#### Generate Problem Variants

```bash
python main.py generate-variants --problem "Integrate x^2 * sin(x) dx" --output variants.json --max-depth 3
```

#### Verify Solutions

```bash
python main.py verify --problem "Integrate x^2 * sin(x) dx" --solution "sin(x) - x*cos(x)"
```

#### Train with GRPO

```bash
python main.py train --dataset variants.json --model mistralai/Mistral-7B-v0.1 --output-dir ./trained_model
```

#### Solve with Test-Time Adaptation

```bash
python main.py solve --problem "Integrate x^2 * sin(x) dx" --model ./trained_model
```

## Project Structure

```
LADDER/
├── config/
│   └── config.py         # Configuration parameters
├── variant_generator/
│   └── generator.py      # Variant generation subsystem
├── verification/
│   └── verifier.py       # Solution verification subsystem
├── rl_subsystem/
│   └── trainer.py        # GRPO training implementation
├── ttrl_subsystem/
│   └── adaptor.py        # Test-time adaptation
├── main.py               # Main entry point
└── README.md             # This file
```

## Extending to Other Domains

While the current implementation focuses on mathematical integration, the LADDER framework can be extended to other domains by:

1. Implementing a domain-specific verification system
2. Adjusting the variant generation prompts for the new domain
3. Fine-tuning reward functions for the specific task

## Performance Metrics

The framework tracks several key metrics:

- Success rate on increasingly difficult problems
- Inference-time adaptation improvements
- Model drift measured by KL divergence
- Solution quality beyond binary correctness

## Scaling Up for Large-Scale Training

To effectively train models using the LADDER framework at scale, follow these guidelines:

### Hardware Requirements

| Training Scale | Recommended Hardware | Estimated Time |
|----------------|----------------------|----------------|
| Small (3B models) | Single A100/H100 GPU or equivalent | 12-24 hours |
| Medium (7B models) | 2-4 A100/H100 GPUs | 2-3 days |
| Large (13B+ models) | 8+ A100/H100 GPUs | 5-7 days |

### Distributed Training Setup

For multi-GPU or multi-node training:

1. **Data Parallel Training**:
   ```bash
   # Example for 4-GPU training
   python -m torch.distributed.launch --nproc_per_node=4 main.py train \
     --dataset large_variants.json --model meta-llama/Llama-2-7b \
     --output-dir ./trained_model --distributed
   ```

2. **Cloud Setup** (AWS example):
   ```bash
   # Launch p4d.24xlarge instance with 8 A100 GPUs
   aws ec2 run-instances --image-id ami-12345678 --instance-type p4d.24xlarge \
     --key-name my-key --security-group-ids sg-12345678
   
   # Clone and set up environment
   git clone https://github.com/Gajesh2007/ladder-framework.git
   cd ladder-framework
   pip install -r requirements.txt
   
   # Run distributed training
   python -m torch.distributed.launch --nproc_per_node=8 main.py pipeline \
     --input problems.txt --model meta-llama/Llama-2-7b \
     --output-dir ./outputs --distributed
   ```

### Dataset Preparation

For serious training, prepare larger datasets:

1. **Generate Extensive Variant Trees**:
   ```bash
   # Generate 100+ root problems with deep trees
   python main.py generate-dataset \
     --input comprehensive_problems.txt \
     --output large_dataset.json \
     --max-depth 4 --problems-per-tree 50
   ```

2. **Dataset Optimization**:
   - Pre-verify all variants to avoid computation during training
   - Filter out invalid or duplicate variants
   - Balance the difficulty distribution within batches

### Optimization Strategies

1. **Memory Optimization**:
   - Use 4-bit quantization for training (LoRA adapters)
   - Implement gradient checkpointing
   - Consider DeepSpeed ZeRO-3 for very large models

2. **Training Efficiency**:
   - Start with smaller batch sizes and increase gradually
   - Use early stopping based on validation performance
   - Apply learning rate warmup and cosine decay
   - Consider gradient accumulation for effective larger batches

3. **Monitoring & Debugging**:
   - Use Weights & Biases or TensorBoard for training visualization
   - Save checkpoints every 500-1000 steps
   - Implement robust logging for variant quality and RL statistics

### Production Deployment

For deploying trained models:

1. **Model Serving**:
   - Use vLLM for optimized inference
   - Implement API endpoints for TTRL adaptation
   - Cache similar problem variants to reduce adaptation time

2. **Resource Management**:
   - Scale TTRL horizontally for multiple concurrent requests
   - Implement timeout policies for adaptation steps
   - Consider hybrid CPU/GPU inference for cost optimization

### Practical Tips

- Start with smaller models (3B range) to debug the pipeline
- Train on a subset of variants before scaling to the full dataset
- Focus on hyperparameter tuning for the GRPO step, as this significantly impacts performance
- For optimal results, the Qwen2.5 7B Deepseek-R1 Distilled model is recommended as per the original research
- When scaling to new domains, allocate significant time to developing and testing the verification component

## License

[MIT License](LICENSE)

## Acknowledgments

This is an implementation of the LADDER (Learning through Autonomous Difficulty-Driven Example Recursion) framework which was originally proposed in the research paper authored by:

- **Toby Simonds** (Tufa Labs)
- **Akira Yoshiyama** (Tufa Labs)

We are merely the implementers of this innovative research work and have no claim to the conceptual development or intellectual property of the LADDER framework itself. All credit for the theoretical foundation, algorithmic design, and experimental validation goes to the original authors.

The framework builds upon research in curriculum learning, recursive self-improvement, and test-time adaptation techniques in language models. This implementation aims to provide a practical realization of the concepts described in their paper.
