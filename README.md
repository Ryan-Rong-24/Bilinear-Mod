# Bilinear Layer for Modular Addition

This project implements a bilinear layer trained on modular arithmetic (mod 113) and analyzes the learned representations through interaction matrices and eigenvector decomposition.

## Overview

Based on:
- **Blog Post**: "Interpreting Modular Addition in MLPs" (LessWrong)
- **Paper**: "Bilinear MLPs Enable Weight-Based Mechanistic Interpretability" (ICLR 2025)
- **Task**: Train a bilinear layer on modular arithmetic, plot interaction matrices & top eigenvector components

## Implementation

### Bilinear Layer Architecture

The model uses the bilinear formulation from the paper:
- **Forward pass**: `g(x) = (Wx) ⊙ (Vx)`, then `output = P[g(x)]`
- **Interaction matrix**: `Q = Σ_k p_k * (w_k ⊗ v_k^T)` for each output dimension
- **Symmetric form**: Interaction matrices are symmetrized since anti-symmetric components contribute zero

### Key Features

1. **BilinearMLP Class**:
   - Two encoder matrices W and V (no element-wise nonlinearity)
   - Output projection matrix P
   - Methods to extract interaction matrices and bilinear tensor
   - Eigendecomposition of interaction matrices

2. **Training**:
   - Dataset: All pairs (a, b) where a, b ∈ [0, 112], output = (a + b) mod 113
   - One-hot encoding for inputs and outputs
   - AdamW optimizer with weight decay = 0.5 (as required)
   - 1000 epochs, batch size 128, learning rate 0.003

3. **Analysis & Visualizations**:
   - **Interaction Matrices** (Section 3.1): Visualize how input pairs (a, b) interact for each output class
   - **Eigendecomposition** (Section 3.2): Extract top eigenvectors that capture the most important input patterns for known output classes
   - **HOSVD** (Section 3.3): Discover the most important output directions automatically without prior knowledge
   - **Eigenvalue Spectra**: Analyze the low-rank structure across all outputs
   - **Effective Rank**: Measure how many eigenvectors are needed for each output

## Key Results

The notebook generates:

1. **Training curves**: Loss and accuracy over epochs
2. **Interaction matrix heatmaps**: Show structured patterns for different output classes
3. **Eigenvector visualizations** (Section 3.2): 
   - Line plots showing a and b components for known output classes
   - Heatmap views for better pattern visibility
4. **HOSVD Analysis** (Section 3.3):
   - Discovered output directions and their singular values
   - Which output classes each direction emphasizes
   - Interaction matrices for discovered directions
   - Eigenvectors revealing patterns without prior knowledge
5. **Statistical analysis**:
   - Eigenvalue spectra across all 113 output classes
   - Distribution of effective ranks
   - Mean and standard deviation of top eigenvalues

## Key Findings

1. **Interaction matrices show structured patterns** for modular addition computation
2. **Eigenvalue spectra are generally low-rank**, indicating the model uses a small number of important directions for each output
3. **Eigenvectors reveal periodic patterns** in how inputs a and b combine (Section 3.2)
4. **HOSVD discovers important computational directions** automatically without needing to know output class meanings (Section 3.3)
5. **The bilinear formulation enables direct weight-based interpretability** without requiring forward passes

## Understanding Section 3.2 vs 3.3

### Section 3.2: Output Features → Eigendecomposition
- **Start with**: Known output classes (sum = 0, 1, 2, ..., 112)
- **Question**: "What input patterns activate output class X?"
- **Method**: Extract interaction matrix Q for class X, then eigendecompose
- **Use case**: When you know what outputs represent

### Section 3.3: No Features → HOSVD
- **Start with**: Nothing! No prior knowledge of outputs
- **Question**: "What are the most important computational directions?"
- **Method**: SVD on the full bilinear tensor to discover important directions
- **Use case**: Exploring unknown models or discovering hidden structure
- **Result**: Finds directions that combine multiple output classes with shared structure

## Differences from Standard MLPs

Unlike the ReLU-based MLP in the blog post:
- **No element-wise nonlinearity** (no ReLU), enabling exact analysis via interaction matrices
- **Quadratic activation in eigenvector basis**: `z^T Q z = Σ λ_i (v_i^T z)^2`
- **Direct weight decomposition**: Can analyze learned algorithm directly from weights
- **Low-rank structure**: Top eigenvectors capture most of the computation

## Usage

Simply run all cells in `Modular_addition_MLP.ipynb` sequentially. The notebook will:
1. Train the bilinear model (takes ~5-10 minutes for 1000 epochs)
2. Generate all visualizations automatically
3. Print analysis results

## References

1. Bart Bussmann - "Interpreting Modular Addition in MLPs" (LessWrong, 2023)
2. Pearce et al. - "Bilinear MLPs Enable Weight-Based Mechanistic Interpretability" (ICLR 2025)
3. Nanda et al. - "Progress Measures for Grokking via Mechanistic Interpretability"

## Future Work

Potential extensions:
- Explore different moduli (P ≠ 113)
- Compare with standard ReLU MLPs
- Analyze grokking behavior (train longer to see generalization)
- Test on other arithmetic operations (multiplication, subtraction)
- Visualize the full 3D bilinear tensor

