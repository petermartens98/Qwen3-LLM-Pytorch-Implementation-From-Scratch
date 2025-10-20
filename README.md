# Qwen3-LLM-Pytorch-Implementation-From-Scratch
`For Full Code Navigate to Jupyter Notebook: `
### Project Overview
- Lightweight LLM inspired by Qwen3, built from scratch in PyTorch.
- Implements modern transformer components including RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), and SwiGLU feed-forward layers.
- Trained using a hybrid Muon + AdamW optimizer setup with causal masking, efficient batching, and evaluation utilities.
- Includes full training pipeline, model loading, and interactive text generation demos for hands-on experimentation.

### Step by Step Overview (Table of Contents)
1. Imports
2. Utility Functions (set_seed, ...)
3. Model Configuration
4. Key/Value Head Expansion Function
5. Muron Optimizer (Orthogonalized Momentum via Newton–Schulz)
6. Data Loading and Caching
7. TextTokenDataset Class
8. Rotary Position Embeddings (RoPE)
9. Grouped-Query Attention (GQA)
10. SwiGLU Feed-Forward Network (FFN)
11. Transformer block (attention + FFN + RMSNorm + residuals)
12. Language model class (MinimalLLM)
13. Evaluation function (loss, accuracy, perplexity)
14. Optimizer setup (hybrid Muon + AdamW)
15. Training loop (AMP, grad accumulation, schedulers)
16. Training Script
17. Model Loading 
18. Model Inference - Autoregressive Text Generation and Chat Interactive Inference.

### Useful Materials
- Qwen 3 Technical Report PDF: https://arxiv.org/pdf/2505.09388
- Qwen 3 GitHub Repo: https://github.com/QwenLM/Qwen3
