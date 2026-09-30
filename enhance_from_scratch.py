"""
enhance_from_scratch.py

A from-scratch PyTorch implementation of the RRDBNet architecture used by
Real-ESRGAN, with custom tiled inference for memory-safe processing of large
images on CPU-only hardware.

Architecture:
    ResidualDenseBlock_5C -> RRDB (3x dense blocks + residual scaling) -> RRDBNet
    (conv_first -> body of RRDB blocks -> upsampling -> conv_last)

Tiled inference splits large images into overlapping tiles, runs each tile
through the model independently, then stitches the outputs back together
using the non-overlapping center region of each tile - this keeps memory
usage bounded regardless of input image size.

Usage:
    python enhance_from_scratch.py --input ./raw_photos --output ./enhanced_photos \\
        --weights RealESRGAN_x4plus.pth --tile-size 256
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock5C(nn.Module):
    """Five densely-connected convolutions with residual scaling (0.2x)."""

    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: three stacked dense blocks + residual scaling."""

    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(nf, gc)
        self.rdb2 = ResidualDenseBlock5C(nf, gc)
        self.rdb3 = ResidualDenseBlock5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Full super-resolution network: feature extraction -> RRDB body -> upsampling."""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        body_fea = self.conv_body(self.body(fea))
        fea = fea + body_fea
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(fea)))


def load_model(weights_path: str, device: torch.device) -> RRDBNet:
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    if "params_strict" in checkpoint:
        model.load_state_dict(checkpoint["params_strict"], strict=True)
    elif "params" in checkpoint:
        model.load_state_dict(checkpoint["params"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()
    return model


def process_tiled(img_tensor: torch.Tensor, model: RRDBNet, device: torch.device,
                   tile_size: int = 256, tile_pad: int = 10) -> torch.Tensor:
    """Runs inference tile-by-tile to bound memory usage, then stitches results."""
    _, c, h, w = img_tensor.shape
    output = torch.zeros((1, c, h * 4, w * 4), device=device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_start, y_end = max(0, y - tile_pad), min(h, y + tile_size + tile_pad)
            x_start, x_end = max(0, x - tile_pad), min(w, x + tile_size + tile_pad)

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                tile_out = model(tile)

            out_y_start = (y - y_start) * 4
            out_y_end = out_y_start + (min(y + tile_size, h) - y) * 4
            out_x_start = (x - x_start) * 4
            out_x_end = out_x_start + (min(x + tile_size, w) - x) * 4

            tile_y_start = (y - y_start) * 4
            tile_y_end = tile_y_start + (out_y_end - out_y_start)
            tile_x_start = (x - x_start) * 4
            tile_x_end = tile_x_start + (out_x_end - out_x_start)

            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = tile_out[
                :, :, tile_y_start:tile_y_end, tile_x_start:tile_x_end
            ]

    return output


def enhance_image(input_path: str, output_path: str, model: RRDBNet, device: torch.device, tile_size: int):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  Skipped (could not read): {input_path}")
        return

    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

    output_tensor = process_tiled(img_tensor, model, device, tile_size=tile_size)

    output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    cv2.imwrite(output_path, output)


def run_batch(input_folder: str, output_folder: str, weights_path: str, tile_size: int = 256):
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(weights_path, device)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue

            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            target_dir = os.path.join(output_folder, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            output_path = os.path.join(target_dir, file)

            if os.path.exists(output_path):
                print(f"Skipping (already processed): {file}")
                continue

            print(f"Processing: {file}")
            try:
                enhance_image(input_path, output_path, model, device, tile_size)
            except Exception as exc:
                print(f"  Failed on {file}: {exc}")

    print(f"\nBatch complete. Enhanced photos saved in: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch super-resolution using a from-scratch RRDBNet implementation.")
    parser.add_argument("--input", required=True, help="Folder of source images (searched recursively)")
    parser.add_argument("--output", required=True, help="Folder to save enhanced images")
    parser.add_argument("--weights", default="RealESRGAN_x4plus.pth", help="Path to pretrained model weights")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for memory-safe inference")
    args = parser.parse_args()

    run_batch(args.input, args.output, args.weights, args.tile_size)
