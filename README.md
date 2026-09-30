# Photo Super-Resolution Batch Pipeline

A batch image-enhancement pipeline built on a Real-ESRGAN-style deep learning model (RRDBNet architecture), used to upscale and clean up low-quality photos in bulk, originally built to prepare product photos for resale listings.

## Problem

Photos taken quickly (in low light, at odd angles, or on older phone cameras) often aren't sharp or bright enough for a clean product listing. Manually editing dozens or hundreds of images one at a time doesn't scale.

## What this does

Runs a batch of images through a pretrained Real-ESRGAN super-resolution model to upscale resolution (4x) and improve sharpness, with an optional face-restoration pass (GFPGAN) for images containing people. Two implementations are included:

- **batch_enhance_cli.py** - thin wrapper that calls the official `realesrgan` inference CLI tool. Simplest path if the package is installed.
- **enhance_from_scratch.py** - a from-scratch PyTorch implementation of the RRDBNet (Residual-in-Residual Dense Block) architecture used by Real-ESRGAN, with custom tiled inference so large images can be processed on CPU without running out of memory.

## Why the from-scratch version exists

The CLI tool is convenient, but implementing the network architecture directly (ResidualDenseBlock_5C -> RRDB -> RRDBNet) and writing custom tiled inference logic demonstrates a working understanding of how the model actually processes an image: dense residual blocks for feature extraction, learned upsampling via interpolation + convolution, and memory-safe tiling that stitches sub-image predictions back into a full-resolution output.

## Tech stack

- **PyTorch** - model definition and inference
- **OpenCV / NumPy** - image I/O and tensor conversion
- Tiled inference for memory-constrained (CPU-only) environments
- Batch processing with resume support (skips already-processed files)

## Status

Functional for batch folder processing; in progress - next steps are adding a quality-comparison report (before/after sharpness metrics) and packaging the model weights download so the repo is runnable without a manual file drop.

## Usage

```bash
pip install -r requirements.txt

# Download RealESRGAN_x4plus.pth model weights and place in this folder
# (see https://github.com/xinntao/Real-ESRGAN for official weights)

python enhance_from_scratch.py --input ./raw_photos --output ./enhanced_photos
```
