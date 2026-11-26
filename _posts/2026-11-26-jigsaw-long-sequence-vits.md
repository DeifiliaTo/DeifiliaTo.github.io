---
title: "Splitting tokens and channels: domain and tensor parallelism for long-sequence ViTs"
date: 2026-11-26
published: false # keep hidden until the paper is accepted (double-blind review)
excerpt: "Work in progress: extending Jigsaw to global attention, so that Vision Transformers on high-resolution scientific data can be trained with each sample split along both tokens and channels."
tags:
  - parallelism
  - transformers
  - HPC
  - weather
---

{% include base_path %}

*This post describes work that is currently under review. All numbers come from the submitted manuscript and may still change. The code will be released together with the paper.*

**TL;DR:** Vision Transformers (ViTs) on scientific images work with very long sequences and many channels. Tools built for language models split the sequence across GPUs, but every GPU still holds all channels of its tokens. We extended **Jigsaw** to global attention, so that each sample is split along both tokens and channels and the weights are split too. On long sequences, this matches the throughput of NVIDIA's Transformer Engine with as little as 65% of its memory in training and 33% in inference. An attention layer scales to 64M tokens on 1,024 GPUs. We used this to train a ViT with global attention over 259,200 tokens of global 0.25° weather data.

<div class="result-cards">
  <div class="result-card"><span class="result-card__number">0.33×</span><span class="result-card__label">the inference memory of Transformer Engine at 4M tokens</span></div>
  <div class="result-card"><span class="result-card__number">64M tokens</span><span class="result-card__label">in one attention layer on 1,024 GH200 GPUs</span></div>
  <div class="result-card"><span class="result-card__number">90%</span><span class="result-card__label">of single-node per-GPU throughput kept at 1,024 GPUs</span></div>
  <div class="result-card"><span class="result-card__number">259,200</span><span class="result-card__label">tokens of global attention in a 0.25° weather ViT</span></div>
</div>

## Why long sequences?

A ViT cuts an image into patches and treats each patch as one token. Smaller patches give the model a finer view of the input, but they also give it more tokens. Attention compares every token with every other token, so its cost grows with the square of the sequence length. Halve the patch side length and you get 4× the tokens, about 16× the attention compute, and 4× the activation memory.

To check whether the finer view is worth this cost, we trained three ViTs on global ERA5 weather data at 0.25° resolution (about 30 km):

| Model | Patch size | Tokens | Width | Parameters | Steps to validation loss 0.0295 |
|---|---:|---:|---:|---:|---:|
| P4x4_512 | 4 × 4 | 64,800 | 512 | 27.3M | 655,000 |
| P2x2_512 | 2 × 2 | 259,200 | 512 | 25.7M | 1.4× fewer than P4x4_512 |
| P2x2_2048 | 2 × 2 | 259,200 | 2048 | 404.9M | 146,000 |

<figure class="post-figure">
<img src="{{ base_path }}/images/blog/jigsaw-vit/fig1-validation.png" alt="Three validation curves over training steps: overall validation loss, 2 m temperature RMSE in kelvin and 10 m wind RMSE in m/s, for patch 2x2 with width 2048, patch 2x2 with width 512, and patch 4x4 with width 512">
<figcaption>Validation curves for the three models: overall loss, 2 m temperature RMSE and 10 m wind RMSE. Each curve ends at the first step where the validation loss reaches 0.0295. <span class="credit">Figure from the submitted manuscript.</span></figcaption>
</figure>

Both levers help. More tokens at the same width (P2x2_512 vs. P4x4_512) reach the target loss in 1.4× fewer steps, with slightly fewer parameters. A wider model at the same sequence length (P2x2_2048) gets there faster still. The catch is memory. These models need long sequences, wide layers or both, and a single GPU cannot hold that.

## What sequence splitting misses

Two families of methods split a Transformer across GPUs:

* **Context parallelism** splits the sequence. Each GPU holds a block of tokens. *Ring attention* passes key/value blocks around a ring of GPUs until every query has seen every key. *DeepSpeed-Ulysses* uses an all-to-all to switch from splitting tokens to splitting attention heads, and back again afterwards.
* **Tensor parallelism**, as in Megatron-LM, splits the weights. Each GPU computes a subset of the attention heads, but every GPU holds the full input.

Combined, as in Transformer Engine, each GPU holds 1/*P* of the tokens, but all channels of those tokens. In language models, the channels are a fixed embedding. In scientific ViTs, they are often dozens or hundreds of physical variables, and their activations take up a large share of the memory.

Jigsaw also splits the channels. Each GPU holds 1/*P* of the tokens and 1/*T* of the channels, where *P* is the number of sequence shards and *T* the number of channel shards:

| Scheme | Input on each GPU | Attention weights on each GPU |
|---|---|---|
| Context parallelism | *S*/*P* tokens × all *d* channels | all |
| Megatron-LM tensor parallelism | all *S* tokens × all *d* channels | 1/*T* |
| Context + Megatron-LM tensor parallelism | *S*/*P* tokens × all *d* channels | 1/*T* |
| **Jigsaw** | ***S*/*P* tokens × *d*/*T* channels** | **1/*T*** |

<figure class="post-figure">
<img src="{{ base_path }}/images/blog/jigsaw-vit/fig2-sharding.png" alt="Diagram comparing how the input, the QKV weights and the Q, K, V outputs are split across two context-parallel and two tensor-parallel ranks under context parallelism, Megatron-LM tensor parallelism, their combination, and Jigsaw">
<figcaption>How the input (left), the QKV weights (middle) and the resulting Q, K, V (right) are split. Fill color marks the sequence shard (CP), outline color the tensor shard (TP), gray means replicated. Only Jigsaw splits the input itself along both dimensions. <span class="credit">Figure from the submitted manuscript.</span></figcaption>
</figure>

## How it works

Jigsaw is our PyTorch framework for domain- and tensor-parallel training, first described in our [PASC26 paper]({{ base_path }}/publication/2026-pasc-domain-tensor-parallelism). So far it supported MLP-Mixers and Swin Transformers. Both mix tokens only locally, so they never need attention across the whole sequence. This work adds full global attention.

1. **QKV projection.** A distributed matrix multiplication turns each GPU's tokens × channels block into queries, keys and values for 1/*T* of the heads, for that GPU's tokens.
2. **Attention core.** The *P* GPUs that share these heads run context-parallel attention: DeepSpeed-Ulysses, ring attention, or a hierarchical mix of both. The local computation calls PyTorch's scaled dot-product attention or Transformer Engine's cuDNN kernels, so we benefit from existing optimized kernels.
3. **Output projection.** A second distributed linear layer combines the heads and returns the data to the tokens × channels split.

The patch embedding and the final layer are convolutions, and they use the same split. The patch embedding's stride equals its kernel size, so the patches don't overlap and splitting space requires no communication. For other convolutions, neighboring GPUs exchange boundary regions (halos).

### Ring attention settings

Ring attention has several independent trade-offs between memory and communication. In Jigsaw each one is a separate setting:

* **Communication pattern.** Key/value blocks move by point-to-point messages or by broadcasts. A hierarchical mode runs the ring within each node and overlaps it with the transfers between nodes.
* **What is kept for the backward pass.** Each GPU either keeps every key/value block it received, which costs memory proportional to *P*, or keeps only its own block and sends the blocks around the ring again in the backward pass.
* **Gradient accumulation.** Partial key/value gradients either travel around the ring in a buffer the size of one block, or go into a buffer the size of the full sequence that is all-reduced at the end.
* **Overlap.** Transfers run on a separate CUDA stream with two receive buffers, so block *j*+1 arrives while block *j* is being computed.
* **CPU offloading.** Saved key/value blocks are copied to pinned host memory during the forward pass and prefetched back during the backward pass, hidden behind the attention computation.

With the memory-lean settings, all buffers have a fixed size. Memory per GPU then stays constant no matter how many GPUs share the sequence.

## Results

We compare against NVIDIA's **Transformer Engine** (TE). TE is developed by the hardware vendor and supports context and tensor parallelism inside attention. Both frameworks use the same cuDNN attention kernels, so differences come from how communication, buffers and saved activations are handled, and from how the data is split.

Configuration names combine the attention algorithm: **U** = DeepSpeed-Ulysses, **R** = ring, **HR** = hierarchical ring, **HRU** = hierarchical ring with Ulysses inside each node. A **J-** prefix means Jigsaw, **TE-** means Transformer Engine.

### Scaling one attention layer to 64M tokens

We trained a single attention layer (width 4096, 32 heads) with 64K tokens per GPU on JUPITER's GH200 GPUs, from 4 to 1,024 GPUs. The global sequence grows from 256K to 64M tokens.

* **On one node**, all variants reach 419 TFLOP/s per GPU (42% of peak).
* **TE runs out of memory** at 4M tokens with the flat ring and at 16M tokens with the hierarchical variant.
* **Jigsaw's memory-lean variants run up to 64M tokens at a constant 16–19 GB per GPU.** The sequence length is limited by the number of GPUs, not by GPU memory.
* **J-HRU drops only from 42% to 38% of peak** between 2 and 256 nodes, which is 90% scaling efficiency. Without Ulysses, the hierarchical ring (J-HR) keeps 312 TFLOP/s per GPU at 1,024 GPUs, against 250 for the flat ring (J-R).

<figure class="post-figure">
<img src="{{ base_path }}/images/blog/jigsaw-vit/fig3-weak-scaling.png" alt="Combined line and bar chart: per-GPU throughput as lines and peak memory as bars, from 4 to 1,024 GH200 GPUs with sequence length from 256K to 64M tokens, for TE and Jigsaw variants, with TE variants marked out of memory beyond 2M or 8M tokens">
<figcaption>Weak scaling on GH200. Lines: throughput per GPU (left axis). Bars: peak memory per GPU (right axis). OOM: out of memory. <span class="credit">Figure from the submitted manuscript.</span></figcaption>
</figure>

The fixed buffers have a cost at small scale: on one node, Jigsaw needs 16–19 GB per GPU, while TE's ring needs 11 GB. For small problems, J-R+cache, which keeps all key/value blocks, is faster than the memory-lean ring, but it runs out of memory at the same point as TE.

### A full Transformer block on three GPU generations

Next, we trained one full Transformer block (attention and MLP) on 32 GPUs, split 16 ways along the sequence and 2 ways along channels. We ran it on A100, GH200 and GB200, from 128K to 4M tokens. Both frameworks use the same context and tensor degrees. Only Jigsaw also splits the input channels.

GH200 at 4M tokens, Jigsaw relative to its TE counterpart:

| Comparison | Memory, training | Memory, inference | Throughput, training | Throughput, inference |
|---|---:|---:|---:|---:|
| J-U vs. TE-U | 1.00× | 0.88× | 1.02× | 1.05× |
| J-HRU vs. TE-HRU | 0.83× | 0.44× | 1.03× | 1.09× |
| J-HR vs. TE-R | 0.70× | 0.33× | 1.04× | 1.19× |
| J-HR + offload vs. TE-R | 0.66× | 0.33× | 1.04× | 1.18× |

Across the three GPU generations:

* **Memory savings grow with sequence length.** At 4M tokens, every Jigsaw variant reaches 1.01–1.19× TE's throughput. At 1M tokens on A100, J-HR with offloading uses 0.65× TE's training memory.
* **CPU offloading costs less than 1% throughput** at 1M tokens on all three generations, including A100, because the copies overlap with the attention computation.
* **Up to 40% of peak in training and 49% in inference on GH200, and 42% and 55% on GB200.**
* **Short sequences show no clear advantage.** At 128K tokens, memory ranges from 0.77× to 1.12× TE's and throughput from 0.86× to 1.16×. The ring variants of both frameworks reach only 17–20% of peak, while the Ulysses variants reach 30–35%.

### Which setting to choose

* **Plenty of memory, smaller problem:** J-HRU, or a ring that keeps all key/value blocks. These are the fastest.
* **Very long sequences or deep models:** the hierarchical ring with local buffers, optionally with CPU offloading. It uses the least memory at no loss in throughput.

## Training a weather ViT at 0.25°

The models in the table at the top were trained with Jigsaw. Inputs are 84 ERA5 variables at two time steps, 24 hours apart: six surface variables, plus six variables on 13 pressure levels. The model predicts the state 24 hours ahead. Each model has a patch embedding, eight Transformer layers with global attention, and a transposed convolution that maps the tokens back to the grid.

All layers, including the convolutions, are split along tokens and channels. The largest model, P2x2_2048, ran on 64 GH200 GPUs: 4-way data parallel × 8 spatial tiles × 2 channel groups. The two smaller models ran on 32 GPUs.

To our knowledge, this is the first ViT trained with global attention over the full 0.25° globe at 259,200 tokens.

## Limitations

* **Fewer steps, more compute.** P2x2_512 reaches the target loss in 1.4× fewer steps than P4x4_512, but each step costs about 15× more compute (13,369 vs. 868 TFLOP). Higher resolution improves the model at a higher total cost.
* **Fixed buffers cost memory at small scale.** On a single node, the memory-lean variants need more memory than TE.
* **On GB200 in training,** the Ulysses-based variants use more memory than their TE counterparts: 1.02–1.03× for J-HRU and 1.12–1.23× for J-U.
* **Most benchmarks are single layers or blocks.** We did not measure throughput for the full weather model.

## Outlook

* **Reduced-precision attention** on newer GPU generations.
* **Other high-resolution domains.** Medical image segmentation and satellite remote sensing have the same combination of large images and many channels.
* **Drop-in layers.** Because Transformer, convolutional and linear layers are parallelized the same way, an existing PyTorch model can be parallelized by swapping its layers for Jigsaw's distributed versions.

The code will be released when the paper is published.
