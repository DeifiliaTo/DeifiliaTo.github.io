---
title: "Jigsaw: splitting the data, not just the model, to train AI weather models"
date: 2026-06-30
excerpt: "An overview of Jigsaw, our PASC26 Best Paper: a parallelization scheme that splits each weather sample, the model weights and the optimizer state across GPUs, so that no two GPUs ever hold the same data."
tags:
  - weather
  - parallelism
  - HPC
---

{% include base_path %}

**TL;DR:** A single sample of global weather data can be larger than a gigabyte. Model-parallel methods built for language models split the model across GPUs, but every GPU still loads and stores the full sample. **Jigsaw** also splits the sample. Each GPU holds a different piece of the input, the weights and the optimizer state, and nothing is ever gathered onto one GPU. Splitting the input reduces both memory and the data each GPU has to read. When reading data is the bottleneck, adding GPUs raises the throughput of *each* GPU. The paper received the Best Paper award at PASC26.

[Paper (ACM)](https://doi.org/10.1145/3815572.3815765) · [Code](https://github.com/RAI-SCC/jigsaw) · [Publication page]({{ base_path }}/publication/2026-pasc-domain-tensor-parallelism)

<div class="result-cards">
  <div class="result-card"><span class="result-card__number">6.4×</span><span class="result-card__label">speedup on 8 GPUs for a 1.1-billion-parameter model</span></div>
  <div class="result-card"><span class="result-card__number">2.0×</span><span class="result-card__label">throughput per GPU when data loading is the bottleneck</span></div>
  <div class="result-card"><span class="result-card__number">1.1 GB</span><span class="result-card__label">per sample at 0.125°, trained for the first time</span></div>
  <div class="result-card"><span class="result-card__number">72%</span><span class="result-card__label">scaling efficiency on 256 GPUs, against 51% without Jigsaw</span></div>
</div>

## Weather samples are large

A training sample for a language model is a few thousand tokens. A training sample for a global weather model is a map of the whole planet. At 0.25° resolution (about 30 km), that is 721 × 1440 grid points for each of about 70 variables: temperature, wind and humidity on 13 pressure levels, plus surface fields. At 0.125° (about 15 km), a single sample is more than 1.1 GB. Higher-resolution reanalysis data is coming, which will make samples larger still.

The common ways to split a model across GPUs, such as Megatron-LM, ZeRO and FSDP, were developed for language models. There, the samples are small and the model is large. These methods split the weights, but every GPU still receives the full input. With four-way model parallelism, the same 1 GB sample is read from disk four times and stored on four GPUs. Some methods also gather the full weights of a layer onto each GPU before computing it.

## Three ways to split the work

<figure class="post-figure">
<img src="{{ base_path }}/images/blog/jigsaw-pasc/fig1-three-axes.png" alt="Schematic of four GPUs: GPUs 0 and 1 each hold half of a data sample and half of the model weights, GPUs 2 and 3 do the same for a different sample" style="max-width: 640px; width: 100%;">
<figcaption>Three axes of parallelism on four GPUs. Domain parallelism splits each data sample, tensor parallelism splits the model weights, and data parallelism gives different samples to different groups. <span class="credit">Figure 1 from the paper. © 2026 the authors, publication rights licensed to ACM.</span></figcaption>
</figure>

* **Data parallelism** gives each GPU a full copy of the model and different samples. It is simple and widely used, but every GPU must fit the whole model and a whole sample.
* **Tensor parallelism** splits the weights of each layer across GPUs, so that larger models fit.
* **Domain parallelism** splits each sample, for example by region of the map or by group of variables. The term comes from scientific simulation, where a physical domain is divided between processors. Before this work, AI weather models had not used it.

## The idea: split everything the same way

Jigsaw combines domain and tensor parallelism so that the data and the weights are split along matching dimensions. In an *n*-way split, each GPU stores 1/*n* of:

* the input sample,
* the weights,
* the optimizer state,
* the activations.

**No two GPUs ever hold the same data, and nothing is ever gathered onto a single GPU.** There are no all-gathers or broadcasts in the forward or backward pass.

### How a linear layer works when everything is split

Take two GPUs. The input *X* is split by variables: GPU 0 holds the first half of the channels, GPU 1 the second half. The weights *W* are split the same way. Each GPU must end up with its half of the output *XW*<sup>T</sup>, still split by channels.

Each GPU splits its local pieces in half once more. The output then becomes a 2 × 2 grid of blocks, and each block is a sum of two products. One of the two products uses only data the GPU already holds. The other needs data from the neighbor. So each GPU:

1. computes the partial product the *other* GPU needs and starts sending it,
2. computes its own local product while the message is in transit,
3. adds the received partial product to its local result.

Communication runs during computation, so most of its cost is hidden. The backward pass works the same way with transposed matrices.

Four-way and eight-way splits follow the same pattern. Four GPUs split the map in half along longitude and the variables in half. Eight GPUs split longitude into four parts and the variables into two.

<figure class="post-figure">
<img src="{{ base_path }}/images/blog/jigsaw-pasc/fig2-jigsaw-weathermixer.png" alt="Left: a stack of global maps split across variables for 2-way, and additionally across longitude for 4-way and 8-way. Middle: WeatherMixer's encoder, MLP-Mixer blocks and decoder. Right: the weight matrix split into 2, 4 and 8 blocks">
<figcaption>Jigsaw applied to WeatherMixer. Left: the input is split across variables (2-way) and additionally across space (4-way, 8-way). Middle: the model. Right: the weights of each layer are split into matching blocks. <span class="credit">Figure 2 from the paper. © 2026 the authors, publication rights licensed to ACM.</span></figcaption>
</figure>

### A model that is all matrix multiplications

To make the most of this, we used **WeatherMixer**, a weather model based on the MLP-Mixer architecture. Each mixing block has two small networks:

* **Token mixing** mixes information across all locations on the globe, separately for each channel. Every location can influence every other, so the model sees the whole planet at once.
* **Channel mixing** mixes information across variables, separately for each location. This captures, for example, how local temperature relates to local wind.

Both are sequences of matrix multiplications, so every layer can use the distributed linear layer above. The other components need little or no communication:

* **Activation functions** (GELU) act on each value separately and need no communication.
* **Encoder and decoder** are convolutions over non-overlapping patches, which become linear layers.
* **The optimizer** updates each GPU's own weights. It needs no communication, and runs while the next sample is loading.
* **The data loader** on each GPU reads only its own part of each sample from disk.

## Why domain parallelism matters

### 1. Each GPU reads less data

With *n*-way domain parallelism, each GPU reads 1/*n* of every sample. This matters whenever the GPUs are waiting for data rather than computing.

We measured throughput across model sizes and found two regimes. For small models, the GPUs wait on data loading, and throughput rises with model size. For large models, throughput levels off near the GPU's peak.

<figure class="post-figure">
<img src="{{ base_path }}/images/blog/jigsaw-pasc/fig7-throughput.png" alt="Four log-log plots of throughput per GPU against workload per GPU, for 1-, 2-, 4- and 8-way models in single and TF32 precision, on HoreKa and LEONARDO, at 0.25 and 0.125 degree resolution" style="max-width: 640px; width: 100%;">
<figcaption>Throughput per GPU against workload per GPU, at 0.25° (top) and 0.125° (bottom), on HoreKa (left) and LEONARDO (right). Solid lines: single precision. Dashed: TF32. Rising lines mean data loading is the bottleneck. Flat lines mean the GPU itself is. Horizontal lines mark peak performance. <span class="credit">Figure 7 from the paper. © 2026 the authors, publication rights licensed to ACM.</span></figcaption>
</figure>

When data loading is the bottleneck, splitting the sample helps each GPU, not just the group. For the smallest model (240 million parameters) in TF32 precision, weak-scaling efficiency on HoreKa was **1.7, 1.9 and 2.0 on 2, 4 and 8 GPUs**. Each GPU did up to twice as much work per second as a single GPU alone.

The effect depends on the machine. HoreKa has 76 CPU cores per node to prepare data. LEONARDO has 32. With fewer cores, data loading is the bottleneck for a wider range of model sizes, and domain parallelism helps for more of them.

### 2. No duplicated memory

Because nothing is duplicated, memory per GPU falls with every GPU added. This allows both larger models and larger samples:

* We trained a **3.2-billion-parameter** WeatherMixer with 4-way splitting. A single GPU fits at most about 1.1 billion parameters.
* We trained on **0.125° data**, with more than 1.1 GB per sample. Without model parallelism, only two of the test models fit into GPU memory at this resolution.

### 3. Lower precision makes it more important

Lower-precision arithmetic speeds up computation but does not speed up reading data. In TF32 precision, more model sizes are limited by data loading than in single precision. Hardware trends point the same way: GPU compute has been growing faster than memory and I/O bandwidth. Future training is likely to be limited by data loading more often, and that is where domain parallelism helps most.

### 4. Better use of a fixed budget

With a fixed number of GPUs, model parallelism changes how training uses them. We trained the same 1-billion-parameter model on 8 GPUs in three ways:

| Setup | Model copies | Global batch size | Minutes per epoch |
|---|---:|---:|---:|
| 1-way (plain PyTorch) | 8 | 8 | 108 |
| 2-way Jigsaw | 4 | 4 | 111 |
| 4-way Jigsaw | 2 | 2 | 145 |

The 2-way model reached **2–9% lower forecast error** on key variables for **3% more training time**. With fewer copies, each training step uses a smaller batch, so there are more steps per epoch. AI weather models are known to train worse with large batches, and model parallelism avoids this.

### 5. Data parallelism scales better on top

Across nodes, Jigsaw uses ordinary data parallelism. Because each GPU holds only a slice of the model, gradients are averaged only among the GPUs that hold the same slice. These messages are smaller. On 256 A100 GPUs, scaling efficiency was:

* **51%** without model parallelism,
* **68%** with 2-way Jigsaw, reaching 11 PFLOP/s,
* **72%** with 4-way Jigsaw, reaching 9 PFLOP/s.

## Scaling results

We ran all experiments on two supercomputers: **HoreKa** at KIT, with NVIDIA A100 and H100 GPUs, and **LEONARDO** at CINECA, with NVIDIA A100 GPUs.

**Strong scaling** keeps the model fixed and adds GPUs. Speedups on HoreKa for a 1.1-billion-parameter model:

| GPUs | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Jigsaw, single precision | 1.0 | 1.9 | 3.4 | 6.4 |
| Jigsaw, TF32 | 1.0 | 1.7 | 2.5 | 4.1 |
| Megatron-LM, 1.2B model (published) | 1.0 | 1.6 | 2.3 | 3.0 |

**Weak scaling** grows the model with the number of GPUs. For the 1.1-billion-parameter baseline in TF32, efficiency on HoreKa was 97%, 96% and 95% on 2, 4 and 8 GPUs. Megatron-LM's published results are 95%, 82% and 77%. In single precision, efficiency stayed at 95% or above for every model size, up to a 3.8-billion-parameter model on 8 GPUs.

**Peak performance:** in single precision, large models reached 98–99% of the GPU's theoretical peak on LEONARDO and 83–88% on HoreKa, for every degree of splitting. The communication is almost entirely hidden behind computation.

For comparison, FSDP gave speedups of 1.3 on 2 GPUs and 2.0 on 4 GPUs for the 1.1-billion-parameter model. FSDP gathers each layer's weights before computing it, which costs much more communication.

## Forecast skill

The 3.2-billion-parameter WeatherMixer was trained with 4-way Jigsaw on 32 H100 GPUs. Its 6-hour forecasts were more accurate than a single member of ECMWF's physics-based ensemble (IFS ENS) on all variables we evaluated, and comparable to AtmoRep, a model of similar size. Training took 4 hours per epoch on 32 H100 GPUs, while AtmoRep needed 4.5 hours per epoch on 128 A100 GPUs, on nearly the same dataset. In our measurements an H100 was about twice as fast as an A100, so this is roughly half the compute.

## Limitations

* **Domain and tensor parallelism stay within one node.** The scheme communicates a lot, so it uses only the fast links between GPUs in a node (up to 8). Across nodes, only data parallelism is used.
* **Mixed precision reaches 45–54% of peak** for the largest models that fit, against 83–99% in single precision. On 256 GPUs, the 2- and 4-way runs reached 28% and 23% of peak.
* **The Megatron-LM numbers come from its published results** on a 1.2-billion-parameter language model, not from running it on our weather model, which Megatron-LM does not support.
* **The forecast evaluation is short-range**: 6-hour forecasts, plus rollouts up to 5 days.

## What came next

Since this paper, we have extended Jigsaw to convolutions and to Swin Transformers, and added a fourth axis that splits the weight samples of a Bayesian model. That work is described in the [BEAST post]({{ base_path }}{% post_url 2026-09-11-beast-4d-parallelism %}).

## Acknowledgements

This work was done with Markus Götz, Lars Helge Heyen, Achim Streit and Charlotte Debus at the Karlsruhe Institute of Technology.

## Citation

```bibtex
@inproceedings{kieckhefen2026jigsaw,
  title     = {Combining Domain and Tensor Parallelism to Train Multi-billion-parameter AI Weather Models},
  author    = {Kieckhefen, Deifilia and Götz, Markus and Heyen, Lars Helge and Streit, Achim and Debus, Charlotte},
  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference (PASC '26)},
  publisher = {ACM},
  year      = {2026},
  pages     = {1--13},
  doi       = {10.1145/3815572.3815765},
  url       = {https://doi.org/10.1145/3815572.3815765}
}
```
