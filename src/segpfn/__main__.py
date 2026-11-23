import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import MODES


def main():
    parser = argparse.ArgumentParser(description="Segment one query using annotated reference images.")
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("IMAGE", "MASK"),
        help="repeat for multi-shot inference, preserving reference order",
    )
    parser.add_argument("--output", type=Path, required=True, help="native-size binary PNG")
    parser.add_argument(
        "--weights", type=Path, default=Path("weights"), help="directory containing the two checkpoints"
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="full",
        help="full (default), skip reference-guided refinement, or skip both refinement and smoothing",
    )
    parser.add_argument(
        "--episode-id", help="use the original episode identifier to replay its exact support-row draws"
    )
    parser.add_argument("--foreground-value", type=int, help="target label in an indexed mask")
    parser.add_argument("--ignore-value", type=int, help="void label excluded from support patch counts")
    args = parser.parse_args()
    inputs = [args.query, *(path for pair in args.reference for path in pair)]
    if args.output.resolve() in {p.resolve() for p in inputs} or args.output.suffix.lower() != ".png":
        parser.error("--output must name a PNG distinct from all inputs")
    from .model import SegPFN

    def load(path):
        with Image.open(path) as image:
            return image.copy()

    query = load(args.query)
    references = [(load(image), load(mask)) for image, mask in args.reference]
    model = SegPFN(args.weights)
    prediction = model.predict(
        query,
        references,
        mode=args.mode,
        episode_id=args.episode_id,
        foreground_value=args.foreground_value,
        ignore_value=args.ignore_value,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".png", delete=False) as tmp:
        name = Path(tmp.name)
    try:
        Image.fromarray(prediction.astype(np.uint8) * 255).save(name)
        os.replace(name, args.output)
    finally:
        name.unlink(missing_ok=True)
    print(f"{args.output} ({query.width} x {query.height}, {args.mode})")


if __name__ == "__main__":
    main()
