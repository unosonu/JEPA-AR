<div align="center">

```
 ██╗ █████╗ ██████╗ ████████╗
 ██║██╔══██╗██╔══██╗╚══██╔══╝
 ██║███████║██████╔╝   ██║   
 ██║██╔══██║██╔══██╗   ██║   
 ██║██║  ██║██║  ██║   ██║   
 ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
```

# JART — JEPA-Augmented Representation Transformer
### *Writing With the Future in Mind*

[preprint link](https://openreview.net/forum?id=n7wBdVEzDS)

**Atul Anand**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/Paper-JART-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](#)

</div>

---

> *When a person writes a sentence, they do not pick words blindly from the one before. There is something that comes first — a vague semantic intent, a felt sense of where the thought is going — that shapes each word choice before the words themselves exist.*
>
> *JART approximates this. In a statistical sense.*

---

## What is JART?

Standard autoregressive language models generate text **left to right**, conditioning each token only on what came before. They are blind to where they are heading.

**JART** combines autoregressive language modelling with a **Joint Embedding Predictive Architecture (JEPA)** that learns compressed, abstract representations of *future* token windows. A lightweight cross-attention module then injects this future signal directly into the backbone's representations — before the language head — letting every token position ask:

> *"Given where this is probably going, how should I represent what I've seen so far?"*

The result is a model that writes with the future in mind.

---

## Key Results

| Dataset | Model Size | AR-only PPL | JART PPL | Δ |
|---|---|---|---|---|
| WikiText-2 | 1.9M | 169.71 | 160.19 | **+9.52 ✓** |
| WikiText-103 | 9.1M | 85.5 | 75.4 | **+10.1 ✓** |

The three-way ablation is the core finding:

```
AR-only              349.14   ←  baseline
JEPA, no guidance    349.67   ←  JEPA alone does nothing
JEPA, guided         337.16   ←  guidance connection is everything
```

The benefit is not from multi-task learning or regularisation.  
It comes specifically from **where** the future signal is injected.

---

## Architecture

```
Context tokens x₁:T
        │
        ▼
  ┌─────────────┐
  │  Backbone   │  ──── h ∈ ℝ^(T×d)
  └─────────────┘
        │                          ┌──────────────────────┐
        │  h.detach()              │   Target Encoder     │
        ▼                          │   (EMA, no grad)     │
  ┌─────────────┐                  └──────────────────────┘
  │  JEPA Proj  │  z_pred               │ z_target
  └─────────────┘        ◄── L_JEPA ───┘
        │
        │  mean(z_pred).detach()  →  g ∈ ℝ^dj
        │
        ▼
  ┌─────────────────────┐
  │  Guidance XAttn     │   Q=h,  K=V=g   (zero-init output proj)
  └─────────────────────┘
        │  correction
        ▼
      h̃ = h + correction
        │
        ▼
  ┌─────────────┐
  │   AR Head   │  ──── L_AR (end-to-end through guidance)
  └─────────────┘
```

**Two fully isolated gradient streams:**
- `L_AR` → AR Head → Guidance XAttn → Backbone
- `L_JEPA` → Projector only *(backbone detached)*

No gradient crosses between the two paths. Ever.

---

## Why Representation-Level, Not Logit-Level?

Earlier designs injected the JEPA signal as a correction to the **output logits**.  
They all failed. Here's why:

| Injection Point | Result | Why |
|---|---|---|
| Same-window JEPA + logit add | +4.5 PPL (noise) | Trivial prediction target, collapses |
| Future-window + logit add | −10.6 PPL | Gradient conflict destroys backbone |
| Future-window + detach + logit add | −2.8 PPL | Signal too coarse for logit correction |
| **Future-window + detach + repr XAttn** | **+9.5 PPL ✓** | Right interface, right level |

Logit-level guidance asks: *"which word probability should change?"* — a fine-grained, context-specific mapping impossible to learn from a low-dimensional future summary.

Representation-level guidance asks: *"how should my understanding of the current context shift?"* — a coarser, learnable problem that the language head then resolves into words.

---

## Quickstart

```bash
# Install dependencies
pip install torch datasets tokenizers matplotlib tqdm

# Clone and run
git clone https://github.com/atulanandd/jart
cd jart
jupyter notebook JART.ipynb
```

Or run directly on **Google Colab** — the notebook auto-downloads WikiText-103 and trains end-to-end in ~33 minutes on a free T4 GPU.

---

## Notebook Structure

```
JART.ipynb
│
├── 1. Config & Setup          # All hyperparameters in one dataclass
├── 2. Data Pipeline           # WikiText-103, BPE tokeniser, pair dataset
├── 3. Backbone                # Pre-norm transformer + RoPE
├── 4. JEPA Components         # Projector, EMA target encoder, VICReg
├── 5. Guidance XAttn          # The core contribution (zero-init, cross-attn)
├── 6. Training Loop           # Mixed precision, torch.compile, grad isolation
├── 7. Evaluation              # PPL with/without guidance, live delta tracking
├── 8. Ablation Study          # AR-only vs JEPA-no-guidance vs JART
└── 9. Analysis & Plots        # Training curves, ablation plots, results
```

---

## Implementation Details

**Speed** — the notebook runs in ~33 min on a free Colab T4:
- Mixed precision (`torch.amp.autocast`)
- `torch.compile(mode="reduce-overhead")`
- Fused attention (`F.scaled_dot_product_attention`)
- `num_workers=4`, `prefetch_factor=2`, `persistent_workers=True`

**Stability** — zero-initialised guidance output projection means:
- The guidance module starts as an exact identity
- AR loss provides gradients only when the JEPA summary genuinely helps
- No manual warmup scheduling needed — it comes online naturally

**Anti-collapse** — VICReg regularisation on JEPA embeddings:
- Variance term: keeps embeddings spread across the space
- Covariance term: decorrelates embedding dimensions
- Without this, the projector collapses to a constant within one epoch

---

## Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `d_model` | 256 | Backbone hidden dim |
| `n_layers` | 6 | Transformer depth |
| `n_heads` | 8 | Attention heads |
| `jepa_dim` | 128 | JEPA embedding dim |
| `jepa_ema_decay` | 0.996 | Target encoder EMA |
| `guidance_n_heads` | 4 | XAttn heads (lighter than backbone) |
| `guidance_init_scale` | 0.0 | Zero-init output projection |
| `seq_len` | 128 | Context window |
| `batch_size` | 64 | Training batch |
| `lr` | 2e-4 → 2e-5 | Cosine schedule |
| `epochs` | 10 | ~33 min on T4 |

---

## The Core Intuition

JART is not that different from how humans write.

When you write a sentence, you have a *vague semantic intent* — a felt sense of where the thought is going — that shapes each word choice before the words themselves exist. You don't know the exact sentence before you write it. But the intent constrains what feels right to say next.

JART approximates this statistically. The JEPA component learns what kinds of representations tend to follow the current context. The guidance module lets the backbone condition itself on that prediction before committing to a word. The model doesn't *know* what it wants to say — but it has learned that certain states tend to precede certain continuations, and it uses that signal to write better.

That turns out to be enough.

---

## Citation

```bibtex
@article{anand2026jart,
  title   = {JART: JEPA-Augmented Representation Transformers — Writing With the Future in Mind},
  author  = {Anand, Atul},
  year    = {2026}
}
```

---

## References

1. LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
2. Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
3. Assran, M. et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR*.
4. Bardes, A., Ponce, J., LeCun, Y. (2022). VICReg. *ICLR*.
5. Su, J. et al. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv:2104.09864*.

---

<div align="center">

*Built by Atul Anand · 2026*

</div>
