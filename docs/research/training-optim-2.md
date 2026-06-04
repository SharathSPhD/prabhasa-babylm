<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# but isn't unsloth for fine tuning?

Yes, Unsloth is primarily designed and marketed for **finetuning and RLHF/SFT post-training**, not from-scratch pretraining.[^1][^2]

However, the underlying optimizations it ships — **custom Triton kernels for RoPE, attention, cross-entropy, MLP layers, and sample packing** — are architecture-level speedups that work regardless of whether you are finetuning or pretraining.[^1] The kernels themselves don't "know" if the weights are pretrained or randomly initialized. So you *can* use Unsloth's kernel library for pretraining, but you would need to wire it into a training loop manually rather than using their high-level `FastLanguageModel` API, which assumes a pretrained checkpoint as a starting point.

For **true from-scratch pretraining on a single GPU**, the more appropriate tools are:

- **`torch.compile` + Flash Attention 2** — these are the core speedups and apply universally to any training loop.[^3][^4]
- **NanoGPT / LitGPT** — Andrej Karpathy's NanoGPT and Lightning AI's LitGPT are purpose-built for pretraining from scratch on a single GPU with all the right optimizations (fused kernels, gradient accumulation, BF16, compile support) baked in.[^4]
- **TinyLlama's training repo** — the TinyLlama project ran full pretraining on 3T tokens using a lean single-node setup; their scripts are the closest public reference for your exact use case (small model, real pretraining, single or few GPUs).[^5]
- **HuggingFace Accelerate + custom training loop** — gives you gradient checkpointing, mixed precision, and compile without the finetuning-specific abstractions.[^4]

So the corrected priority stack for **from-scratch pretraining on GB10** is:

1. **Flash Attention 2** — biggest single win, fully pretraining-compatible.[^4][^6]
2. **`torch.compile`** — ~40% throughput gain, one line, works in any loop.[^3][^7]
3. **BF16 mixed precision** — tensor core utilization, stable for pretraining.[^4]
4. **Sample packing** — eliminate padding; implement manually or via LitGPT's dataloader.[^2]
5. **Gradient checkpointing** — if batch size is constrained.[^6]
6. **8-bit Adam / fused AdamW** — smaller optimizer state.[^4]
7. **Unsloth kernels** — usable if you extract and integrate them manually into your pretraining loop, but not plug-and-play for pretraining out of the box.[^1]
<span style="display:none">[^8]</span>

<div align="center">⁂</div>

[^1]: https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/

[^2]: https://x.com/Dr_Singularity/status/1998883206934999529

[^3]: https://ai.gopubby.com/torch-compile-how-it-makes-pytorch-models-so-fast-d8362488911f

[^4]: https://huggingface.co/docs/transformers/v4.42.0/perf_train_gpu_one

[^5]: https://www.emergentmind.com/topics/tinyllama-model

[^6]: https://flashattn.dev/blog/gradient-checkpointing-explained

[^7]: https://hippocampus-garden.com/torch_compile/

[^8]: image.jpg

