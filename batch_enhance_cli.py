"""
batch_enhance_cli.py

Thin wrapper around the official `realesrgan` package's inference CLI,
for batch super-resolution when the package is installed in the environment.

Usage:
    python batch_enhance_cli.py --input ./raw_photos --output ./enhanced_photos
"""

import argparse
import os
import subprocess


def run_batch_enhance(input_dir: str, output_dir: str, model_name: str = "RealESRGAN_x4plus",
                       outscale: int = 2, face_enhance: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "realesrgan.inference_realesrgan",
        "-n", model_name,
        "-i", input_dir,
        "-o", output_dir,
        "--outscale", str(outscale),
        "--format", "jpg",
    ]
    if face_enhance:
        cmd.append("--face_enhance")

    print(f"Running batch enhancement: {model_name}, outscale={outscale}, face_enhance={face_enhance}")
    subprocess.run(cmd, check=True)
    print(f"Finished. Enhanced images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch photo enhancement via Real-ESRGAN CLI.")
    parser.add_argument("--input", required=True, help="Folder of source images (searches recursively)")
    parser.add_argument("--output", required=True, help="Folder to save enhanced images")
    parser.add_argument("--model", default="RealESRGAN_x4plus",
                         help="Model name: RealESRGAN_x4plus (photos) or RealESRGAN_x4plus_anime_6B (graphics/art)")
    parser.add_argument("--outscale", type=int, default=2, help="Output resolution multiplier")
    parser.add_argument("--no-face-enhance", action="store_true", help="Disable GFPGAN face restoration")
    args = parser.parse_args()

    run_batch_enhance(args.input, args.output, args.model, args.outscale, not args.no_face_enhance)
