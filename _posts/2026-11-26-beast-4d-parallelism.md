---
title: "BEAST: Bayesian weather forecasting with 4D parallelism at exascale"
date: 2026-11-26
excerpt: "An overview of BEAST, a Bayesian AI weather model that estimates both forecast and model uncertainty, and the four-way parallelization that made it trainable on 20,480 GPUs."
tags:
  - weather
  - parallelism
  - Bayesian neural networks
  - HPC
---

{% include base_path %}

**TL;DR:** Most AI weather models do not estimate how uncertain the model itself is. We built **BEAST**, a Bayesian weather model that estimates two kinds of uncertainty: how unpredictable the weather is, and how uncertain the model is. Bayesian models are expensive to train, so we split the work across GPUs along four axes at once. The largest configuration reached a peak of 3.96 exaflops per second on the JUPITER supercomputer. The work is a finalist for the 2026 ACM Gordon Bell Prize for Climate Modelling.

[Paper (arXiv)](https://arxiv.org/abs/2609.12815) · [Jigsaw code](https://github.com/RAI-SCC/jigsaw)

<div class="result-cards">
  <div class="result-card"><span class="result-card__number">3.96 EFLOP/s</span><span class="result-card__label">peak training performance on 20,480 GPUs</span></div>
  <div class="result-card"><span class="result-card__number">96</span><span class="result-card__label">weight samples drawn at every training step</span></div>
  <div class="result-card"><span class="result-card__number">&lt; 20 s</span><span class="result-card__label">for a 15-day global forecast</span></div>
</div>

A forecast is most useful when it also says how confident it is. When a forecast turns out wrong, there are two possible reasons: the weather was inherently hard to predict, or the model had not learned the situation well. This post describes a model that estimates both, and the parallelization needed to train it.

## Two sources of uncertainty

**Aleatoric uncertainty** comes from the atmosphere itself. Weather is chaotic: small differences in the initial state grow into large differences a few days later. Even a perfect model cannot remove this.

**Epistemic uncertainty** comes from the model. An AI weather model is trained on about 40 years of historical data. For situations that are rare in that record, such as record heatwaves, many different models fit the training data equally well but disagree on the new case.

Current probabilistic AI weather models mostly address aleatoric uncertainty. They generate an ensemble of forecasts whose spread reflects how chaotic the situation is. All members come from one fixed model, so the ensemble says nothing about uncertainty in the model itself.

A **Bayesian neural network** addresses epistemic uncertainty. Each weight is a probability distribution, described by a mean and a standard deviation, instead of a single value. A forecast draws several sets of weights from these distributions and runs the model once per set. Where the resulting forecasts disagree, the model is uncertain.

## The cost of Bayesian neural networks

Bayesian networks are expensive for two reasons, and the two costs multiply:

* **Twice the parameters.** Each weight has a mean and a standard deviation, so the model is about twice as large.
* **Many forward passes.** Each training example is processed once per weight sample. We use 96 samples, which means roughly 96 times the compute of a deterministic model.

Global weather models are large to begin with. Our input covers the whole planet at 0.25° resolution (about 30 km) with 84 variables: temperature, wind, humidity and others, at the surface and on 13 pressure levels. With models of hundreds of millions to billions of parameters and 96 samples per step, training requires thousands of GPUs. The main challenge is distributing the work so that communication between GPUs doesn't dominate the run time.

## Four axes of parallelism

We split the work along four independent axes. Because each axis divides the problem in a different direction, they can be combined.

<figure class="post-figure">
<svg class="fig-4d" viewBox="0 0 760 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="fig4d-title"><title id="fig4d-title">The four ways BEAST splits the work: map tiles and variables inside one model copy, weight samples across copies, and dates across groups of copies.</title><rect class="panel" x="5" y="5" width="230" height="260" rx="10"/><rect class="panel" x="265" y="5" width="230" height="260" rx="10"/><rect class="panel" x="525" y="5" width="230" height="260" rx="10"/><text class="title" x="20" y="32">1 + 2. Domain × tensor</text><text class="sub" x="20" y="50">one copy of the model</text><rect class="gpu-a" x="40" y="70" width="38" height="38" rx="2"/><rect class="gpu-a" x="80" y="70" width="38" height="38" rx="2"/><rect class="gpu-a" x="120" y="70" width="38" height="38" rx="2"/><rect class="gpu-a" x="160" y="70" width="38" height="38" rx="2"/><rect class="gpu-b" x="40" y="110" width="38" height="38" rx="2"/><rect class="gpu-b" x="80" y="110" width="38" height="38" rx="2"/><rect class="gpu-b" x="120" y="110" width="38" height="38" rx="2"/><rect class="gpu-b" x="160" y="110" width="38" height="38" rx="2"/><text x="20" y="190">↔ columns: map tiles (domain)</text><text x="20" y="210">↕ rows: variables and</text><text x="38" y="227">weights (tensor)</text><text class="sub" x="20" y="252">each square is one GPU</text><text class="title" x="280" y="32">3. Uncertainty</text><text class="sub" x="280" y="50">copies draw different weights</text><rect class="gpu-a" x="290" y="66" width="18" height="18" rx="2"/><rect class="gpu-a" x="310" y="66" width="18" height="18" rx="2"/><rect class="gpu-a" x="330" y="66" width="18" height="18" rx="2"/><rect class="gpu-a" x="350" y="66" width="18" height="18" rx="2"/><rect class="gpu-b" x="290" y="86" width="18" height="18" rx="2"/><rect class="gpu-b" x="310" y="86" width="18" height="18" rx="2"/><rect class="gpu-b" x="330" y="86" width="18" height="18" rx="2"/><rect class="gpu-b" x="350" y="86" width="18" height="18" rx="2"/><text x="385" y="91">sample 1</text><rect class="gpu-a" x="290" y="112" width="18" height="18" rx="2"/><rect class="gpu-a" x="310" y="112" width="18" height="18" rx="2"/><rect class="gpu-a" x="330" y="112" width="18" height="18" rx="2"/><rect class="gpu-a" x="350" y="112" width="18" height="18" rx="2"/><rect class="gpu-b" x="290" y="132" width="18" height="18" rx="2"/><rect class="gpu-b" x="310" y="132" width="18" height="18" rx="2"/><rect class="gpu-b" x="330" y="132" width="18" height="18" rx="2"/><rect class="gpu-b" x="350" y="132" width="18" height="18" rx="2"/><text x="385" y="137">sample 2</text><rect class="gpu-a" x="290" y="158" width="18" height="18" rx="2"/><rect class="gpu-a" x="310" y="158" width="18" height="18" rx="2"/><rect class="gpu-a" x="330" y="158" width="18" height="18" rx="2"/><rect class="gpu-a" x="350" y="158" width="18" height="18" rx="2"/><rect class="gpu-b" x="290" y="178" width="18" height="18" rx="2"/><rect class="gpu-b" x="310" y="178" width="18" height="18" rx="2"/><rect class="gpu-b" x="330" y="178" width="18" height="18" rx="2"/><rect class="gpu-b" x="350" y="178" width="18" height="18" rx="2"/><text x="385" y="183">sample 3</text><text x="385" y="216">…</text><text x="280" y="240">→ combine: mean and spread</text><text class="title" x="540" y="32">4. Data</text><text class="sub" x="540" y="50">groups train on different dates</text><rect class="gpu-a" x="555" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="567" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="579" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="591" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="555" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="567" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="579" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="591" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="555" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="567" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="579" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="591" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="555" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="567" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="579" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="591" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="555" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="567" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="579" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="591" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="555" y="136" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="567" y="136" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="579" y="136" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="591" y="136" width="10.5" height="10.5" rx="2"/><text x="555" y="172">Mar 1987</text><rect class="gpu-a" x="650" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="662" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="674" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="686" y="68" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="650" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="662" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="674" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="686" y="80" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="650" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="662" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="674" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="686" y="96" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="650" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="662" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="674" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="686" y="108" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="650" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="662" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="674" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-a" x="686" y="124" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="650" y="136" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="662" y="136" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="674" y="136" width="10.5" height="10.5" rx="2"/><rect class="gpu-b" x="686" y="136" width="10.5" height="10.5" rx="2"/><text x="650" y="172">Aug 2011</text><text x="540" y="215">→ average the</text><text x="555" y="232">training updates</text><path class="arrow" d="M237 135 h26 m-7 -6 l7 6 l-7 6"/><path class="arrow" d="M497 135 h26 m-7 -6 l7 6 l-7 6"/></svg>
<figcaption>The four axes. One copy of the model is spread over a grid of GPUs (domain × tensor). Several copies draw different weight samples (uncertainty). Groups of those copies train on different dates (data).</figcaption>
</figure>

### 1. Domain: split the map

The input is a global map with about a million grid points per variable. The map is divided into tiles, and each GPU processes one tile. The model is a Swin Transformer, which computes attention within local windows. When the windows shift, neighboring GPUs exchange the boundary regions of their tiles.

### 2. Tensor: split the variables and the weights

Each GPU also holds only a subset of the variables and a slice of the model weights. Existing parallelism schemes were designed for language models. They split the sequence and the weights, but not the input channels. Our scheme also splits the input along the channel dimension. Each layer is written as a distributed matrix multiplication, so no data has to be redistributed between GPUs within a layer.

Domain and tensor parallelism together make up one copy of the model. This combination is **Jigsaw**, described in our [PASC26 paper]({{ base_path }}/publication/2026-pasc-domain-tensor-parallelism), which received the Best Paper award.

### 3. Uncertainty: split the weight samples

This axis is new in this work. Several copies of the model run side by side. Each copy draws different weight samples and receives its own random noise input, which represents the aleatoric part of the uncertainty.

Combining the copies is cheap. The training loss only needs the ensemble **mean and standard deviation**. Each copy contributes two tensors, the sum of its predictions and the sum of their squares, so the communication cost is small relative to the added compute.

### 4. Data: split the dates

Groups of model copies process different time steps from the 40-year record and average their gradients. This is standard data parallelism.

Data parallelism has a known limit. Each additional group increases the global batch size, and beyond about 16 to 32 samples per step, training quality degrades. **The uncertainty axis avoids this limit.** It adds GPUs by drawing more weight samples, not by increasing the batch size.

As an example, the smallest test configuration uses 16 GPUs. One model copy spans 4 spatial tiles × 2 channel groups, and 2 such copies draw different weight samples.

## Model

**BEAST** stands for Bayesian Exascale Atmospheric Swin Transformer.

* **Input:** the atmospheric state at the current time and 12 hours earlier, from ERA5 reanalysis data at 0.25° resolution.
* **Output:** the state 12 and 24 hours ahead. Longer forecasts are produced by repeating this step.
* **Sizes:** a 2.4-billion-parameter configuration for scaling tests on JUPITER, a 1.1-billion configuration on LUMI, and a 700-million-parameter model that we trained fully to evaluate forecast skill.

## Performance

We ran experiments on two European supercomputers: **JUPITER** at Forschungszentrum Jülich, with NVIDIA GH200 GPUs, and **LUMI** at CSC in Finland, with AMD MI250X GPUs.

**On 20,480 GPUs on JUPITER, the 2.4-billion-parameter model reached a peak of 3.96 exaflops per second**, at 73.99% strong-scaling efficiency. One exaflop is 10<sup>18</sup> floating-point operations. On LUMI, the 1.1-billion-parameter model reached 463.23 petaflops per second on 1,024 nodes, at 92.09% efficiency. The scheme runs on both NVIDIA and AMD hardware.

The full training run used 384 JUPITER nodes with **96 weight samples per step**. One epoch over the 40-year dataset took 25.45 hours, and training took nearly one million gradient updates.

## Forecast skill

We evaluated BEAST on 2022 data against ECMWF's operational physics-based ensemble (IFS-ENS) and against FGN, a leading AI ensemble model. The variables were near-surface temperature and wind, and temperature and wind at 850 hPa (about 1.5 km altitude).

**BEAST's accuracy was competitive with or better than both baselines, with better calibrated uncertainty and stable rollouts up to ten days.** Calibration describes whether the ensemble spread matches the actual forecast error. A well-calibrated model reports high uncertainty when its errors tend to be large, and low uncertainty when they tend to be small.

Larger ensembles reduce the error, with diminishing returns. The table shows 2-meter temperature RMSE by ensemble size:

| Ensemble members | GPUs | 12 h RMSE | 24 h RMSE |
|---:|---:|---:|---:|
| 8 | 128 | 0.7507 | 0.8661 |
| 32 | 512 | 0.7289 | 0.8390 |
| 96 | 1,536 | 0.7237 | 0.8328 |
| 768 | 3,072 | 0.7226 | 0.8316 |

We also examined extreme events, where uncertainty estimates matter most for decision-making:

* **Tropical Cyclone Damien** (February 2020, Western Australia): the ensemble mean followed the observed track through landfall.
* **The August 2022 heatwave** in the southern UK, France and Spain, and **the December 2022 cold spell** in Western and Central Europe: BEAST's errors were comparable to or lower than those of ECMWF's high-resolution forecast (IFS HRES).

## Inference

A trained BEAST model produces a **15-day global forecast in under 20 seconds**. Each ensemble member runs on its own set of GPUs, so larger ensembles need more GPUs but not more time. The paper reports that BEAST generates large ensembles 3 to 4 times faster than the current best AI model.

## Limitations

* Uncertainty parallelism scales well to a few thousand GPUs, but efficiency drops at the largest scales. The main causes are load on the file system and the communication required for the loss computation.
* Forecast skill was evaluated on a single year (2022). A longer evaluation period would give a more complete picture.
* Training remains expensive: one epoch takes about a day on 384 nodes.

## Outlook

* **Kilometer-scale models.** Higher resolution means larger inputs, which domain and tensor parallelism are designed to handle.
* **Climate and ocean emulators.** In these fields, high-quality training data is limited, so estimates of epistemic uncertainty are particularly relevant.
* **Lower precision.** Stable training at lower numerical precision would reduce the cost of each step.

## Acknowledgements

This work was done with Juan Pedro Gutiérrez Hermosillo Muriedas (equal contribution), Lars Helge Heyen, Mathis Bode, Iida Hakulinen, Andreas Herten, Chelsea Maria John, Thorsten Kurth, Anni Moisala, Asena Karolin Özdemir, Kaleb Phipps, Oskar Taubert, Arvid Weyrauch, Markus Götz and Charlotte Debus, at KIT, Forschungszentrum Jülich, CSC and NVIDIA. We thank the JUPITER AI Factory and the JUPITER Research and Early Access Program for computing time on JUPITER.

## Citation

```bibtex
@misc{kieckhefen2026beast,
  title     = {4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling},
  author    = {Kieckhefen, Deifilia and Muriedas, Juan Pedro Gutiérrez Hermosillo and Heyen, Lars Helge and Bode, Mathis and Hakulinen, Iida and Herten, Andreas and John, Chelsea Maria and Kurth, Thorsten and Moisala, Anni and Özdemir, Asena Karolin and Phipps, Kaleb and Taubert, Oskar and Weyrauch, Arvid and Götz, Markus and Debus, Charlotte},
  year      = {2026},
  publisher = {arXiv},
  doi       = {10.48550/arXiv.2609.12815},
  url       = {https://arxiv.org/abs/2609.12815}
}
```
