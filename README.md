# OS-DDPM: One-Step Denoising Diffusion Probabilistic Model for 3D MRI Super-Resolution

Official PyTorch implementation of the MICCAI paper "OS-DDPM: One-Step Denoising Diffusion Probabilistic Model for 3D MRI Super-Resolution". 

Examples of the generated MR volumes via one-step diffusion:

![](C:\Users\yanyanghui\Desktop\one-step-diffusion\OS-DDPM\figs\BraTS2024.gif) 

![](C:\Users\yanyanghui\Desktop\one-step-diffusion\OS-DDPM\figs\IXI.gif)

## Abstract

Magnetic Resonance Imaging (MRI) typically acquires a volume by stacking thick slices due to scanning characteristics, which obtains High-Resolution (HR) contents in the imaging plane direction and Low-Resolution (LR) contents in the through-plane direction. Denoising Diffusion Probabilistic Models (DDPM) have recently been introduced to perform 3D MRI Super-Resolution (SR) along the $z$-axis. However, existing DDPM-based methods generate the HR MR volume iteratively, limiting their application in clinical practice. Additionally, volume-wise training of DDPM suffers from significant memory overhead. In this study, we propose a One-Step DDPM (OS-DDPM) for 3D MRI SR, which can generate a high-quality volume sample in a short inference time. Firstly, we design a training strategy based on variational score distillation to capture a mapping relationship between random noise and HR data. Secondly, noise-aware contrastive learning is proposed to further narrow the distribution difference between generated samples and ground truth. Finally, all models are trained on sub-volume data obtained through the random overlap cropping operation, which alleviates the need for computational resources. Experiments demonstrate that the proposed method generates volume data at a speed of 3.27 seconds and achieves a PSNR of 30.30dB at 8mm slice thickness. Meanwhile, OS-DDPM outperforms other diffusion model-based 3D SR methods on two MRI brain datasets, providing a feasible application of DDPM in clinical practice.

## The proposed method

The training phase ((a) to (d)) and inference phase (e) of OS-DDPM are illustrated in the following figure:

![](C:\Users\yanyanghui\Desktop\one-step-diffusion\OS-DDPM\figs\Framework.png)

The training phase of OS-DDPM can be summarized as the following pseudo-code:

![](C:\Users\yanyanghui\Desktop\one-step-diffusion\OS-DDPM\figs\Algorithm.png)

The derivation of $\nabla_{\theta}L_{VSD}$ (Equation 5 in the paper) is as follows:

![](C:\Users\yanyanghui\Desktop\one-step-diffusion\OS-DDPM\figs\derivation.png)

## Datasets

Two public brain datasets are used to verify OS-DDPM, including the [BraTS 2024 dataset](https://www.synapse.org/Synapse:syn53708249/wiki/626323) and the [IXI dataset](https://brain-development.org/ixi-dataset/). 

## Results

Qualitative results of different methods on the BraTS 2024 dataset and the IXI dataset are shown in the following figure: 

![](C:\Users\yanyanghui\Desktop\one-step-diffusion\OS-DDPM\figs\results.png)

## Dependencies

We recommend using a [conda](https://github.com/conda-forge/miniforge#mambaforge) environment to install the required dependencies. You can create and activate such an environment called `OS-DDPM` by running the following commands:

```python
mamba env create -f environment.yml
mamba activate OS-DDPM
```

## Dataloaders

We provide two dataloaders for the BraTS 2024 dataset and the IXI dataset. Only the `data_dir` and the `dataloader` type need to be changed, and the data structure in the initially downloaded files does not need to be modified.

## Pre-training DDPM 

Run the following command. Change the parameters directly in the `image_train.py` as needed.

```
python image_train.py
```

## Iterative sampling for the trained DDPM

Run the following command. Change the parameters directly in the `image_sample.py` as needed.

```
python image_sample.py
```

## Distill to get OS-DDPM

Run the following command. Change the parameters directly in the `onestep_train.py` as needed.

```
python onestep_train.py
```

## One-step generation

Run the following command. Change the parameters directly in the `onestep_sample.py` as needed.

```
python onestep_sample.py
```

## Acknowledgements

We thank the following repositories:

- [openai/improved-diffusion: Release for Improved Denoising Diffusion Probabilistic Models](https://github.com/openai/improved-diffusion)
- [One-step Diffusion with Distribution Matching Distillation](https://tianweiy.github.io/dmd/)
- [GlassyWu/AECR-Net: Contrastive Learning for Compact Single Image Dehazing, CVPR2021](https://github.com/GlassyWu/AECR-Net)
- [Official Implementation of "Denoising Diffusion Autoencoders are Unified Self-supervised Learners"](https://github.com/FutureXiang/ddae)