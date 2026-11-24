"""Prepare, predict, and score the two SegPFN one-shot personalized benchmarks."""

import argparse
import base64
import csv
import hashlib
import io
import json
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import PIL
from PIL import Image, ImageDraw

BENCHMARKS = Path(__file__).resolve().parent
DATASETS = ("davis-2017-ics", "cuhk-sysu-ics")
MASKPS_PILLOW = "10.4.0"
INPUT_FIELDS = ("episode_id", "reference_image", "reference_mask", "target_image")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def asset_checksums(dataset):
    return {row["path"]: row for row in read_csv(BENCHMARKS / dataset / "checksums.csv")}


def check_asset(checksums, name, size, content):
    expected = checksums[name]
    if size != (int(expected["width"]), int(expected["height"])):
        raise ValueError(f"Unexpected dimensions for {name}: {size}")
    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise ValueError(f"Content differs from the benchmark for {name}")


@contextmanager
def new_output(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists. Choose a new output directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        yield staged
        staged.rename(path)


def image_filename(task, role):
    if "sequence" in task:
        return f"{task['sequence']}/{task[role + '_frame']}.jpg"
    return task[role + "_image"]


def input_row(task):
    return {
        "episode_id": task["episode_id"],
        "reference_image": "images/" + image_filename(task, "reference"),
        "reference_mask": f"reference_masks/{task['episode_id']}.png",
        "target_image": "images/" + image_filename(task, "target"),
    }


def check_jpeg(checksums, name, jpeg):
    with Image.open(io.BytesIO(jpeg)) as image:
        check_asset(checksums, name, image.size, jpeg)


def davis_assets(root, task, role):
    filename = image_filename(task, role)
    jpeg = (root / "JPEGImages" / "480p" / filename).read_bytes()
    annotation = root / "Annotations" / "480p" / (filename[:-4] + ".png")
    with annotation.open("rb") as stream, Image.open(stream) as indexed:
        if indexed.mode not in ("P", "L"):
            raise ValueError(f"Expected DAVIS instance labels in {annotation}")
        # Preserve palette indices. Converting the palette to luminance changes the labels.
        labels = Image.frombytes("L", indexed.size, indexed.tobytes())
    instance = int(task["instance_id"])
    mask = labels.point([int(i == instance) for i in range(256)])
    return jpeg, mask


def cuhk_assets(root, task, role):
    filename = image_filename(task, role)
    annotation = json.loads((root / Path(filename).with_suffix(".json").name).read_bytes())
    shape = annotation["shapes"][int(task[role + "_shape_index"])]
    if shape["label"] != "person_" + task["person_id"] or shape["shape_type"] != "polygon":
        raise ValueError(f"Unexpected identity or polygon in {filename} for {task['episode_id']}")
    jpeg = base64.b64decode(annotation["imageData"], validate=True)
    with Image.open(io.BytesIO(jpeg)) as image:
        mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in shape["points"]], fill=1)
    return jpeg, mask


def prepare(dataset, source, output):
    tasks = read_csv(BENCHMARKS / dataset / "tasks.csv")
    checksums = asset_checksums(dataset)
    if dataset == "cuhk-sysu-ics" and PIL.__version__ != MASKPS_PILLOW:
        raise ValueError(
            f"CUHK polygon reconstruction requires Pillow {MASKPS_PILLOW} for exact mask boundaries. "
            "Run uv run benchmarks/get_datasets.py to use the pinned preparation environment."
        )
    read_assets = davis_assets if dataset == "davis-2017-ics" else cuhk_assets
    with ExitStack() as stack:
        root = source
        if not root.is_dir():
            root = zipfile.Path(stack.enter_context(zipfile.ZipFile(root)))
        if dataset == "davis-2017-ics" and (root / "DAVIS").is_dir():
            root = root / "DAVIS"
        dest = stack.enter_context(new_output(output))
        (dest / "reference_masks").mkdir()
        (dest / "ground_truth").mkdir()
        written = set()
        with (dest / "inputs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for task in tasks:
                eid = task["episode_id"]
                inputs = input_row(task)
                for role, directory in (("reference", "reference_masks"), ("target", "ground_truth")):
                    name = inputs[role + "_image"]
                    jpeg, mask = read_assets(root, task, role)
                    if name not in written:
                        check_jpeg(checksums, name, jpeg)
                        (dest / name).parent.mkdir(parents=True, exist_ok=True)
                        (dest / name).write_bytes(jpeg)
                        written.add(name)
                    mask_name = f"{directory}/{eid}.png"
                    check_asset(checksums, mask_name, mask.size, mask.tobytes())
                    mask.save(dest / mask_name)
                    written.add(mask_name)
                writer.writerow(inputs)
        if written != set(checksums):
            raise ValueError("Prepared files do not match the benchmark inventory")
    print(f"Prepared and verified {len(tasks)} tasks in {output}", flush=True)


def verify_prepared(dataset, root):
    tasks = read_csv(BENCHMARKS / dataset / "tasks.csv")
    if read_csv(root / "inputs.csv") != [input_row(task) for task in tasks]:
        raise ValueError(f"{root / 'inputs.csv'} differs from the benchmark task list")
    checksums = asset_checksums(dataset)
    for name in checksums:
        if name.startswith("images/"):
            check_jpeg(checksums, name, (root / name).read_bytes())
        else:
            with Image.open(root / name) as mask:
                if mask.format != "PNG" or mask.mode != "L":
                    raise ValueError(f"Expected an 8-bit grayscale PNG at {root / name}")
                check_asset(checksums, name, mask.size, mask.tobytes())
    print(f"Verified existing {len(tasks)} tasks in {root}", flush=True)


def predict(args):
    import numpy as np

    from segpfn.model import SegPFN

    tasks = read_csv(args.data / "inputs.csv")
    with new_output(args.output) as dest:
        model = SegPFN(args.weights)
        for task in tasks:
            eid = task["episode_id"]
            with (
                Image.open(args.data / task["target_image"]) as target,
                Image.open(args.data / task["reference_image"]) as reference,
                Image.open(args.data / task["reference_mask"]) as mask,
            ):
                prediction = model.predict(target, [(reference, mask)], episode_id=eid)
            Image.fromarray(prediction.astype(np.uint8) * 255).save(dest / f"{eid}.png")
    print(f"Wrote {len(tasks)} predictions to {args.output}")


def binary_mask(path):
    import numpy as np

    with path.open("rb") as stream, Image.open(stream) as image:
        pixels = np.asarray(image)
        if image.format != "PNG" or pixels.ndim != 2 or pixels.dtype.kind not in "uib":
            raise ValueError(f"Expected a single-channel binary PNG at {path}")
        values = set(np.unique(pixels))
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"Expected 0/1 or 0/255 mask values at {path}")
        return pixels != 0


def score(args):
    import numpy as np

    tasks = read_csv(BENCHMARKS / args.dataset / "tasks.csv")
    checksums = asset_checksums(args.dataset)
    root = args.data or Path("data") / args.dataset
    intersection, union = 0, 0
    for task in tasks:
        filename = task["episode_id"] + ".png"
        name = "ground_truth/" + filename
        truth = binary_mask(root / name)
        check_asset(checksums, name, (truth.shape[1], truth.shape[0]), truth.tobytes())
        prediction = binary_mask(args.predictions / filename)
        if prediction.shape != truth.shape:
            raise ValueError(f"{filename} has shape {prediction.shape}, expected {truth.shape}")
        intersection += np.count_nonzero(prediction & truth)
        union += np.count_nonzero(prediction | truth)
    print(
        f"{args.dataset} | {len(tasks)} tasks | intersection={intersection} | union={union} | "
        f"pooled foreground IoU={100 * intersection / union:.4f}%"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="build images, binary masks, and inputs.csv")
    prep.add_argument("dataset", choices=DATASETS)
    prep.add_argument("--source", type=Path, required=True, help="upstream ZIP or extracted dataset root")
    prep.add_argument("--output", type=Path, required=True, help="new prepared dataset directory")
    inference = commands.add_parser("predict", help="run the released SegPFN predictor on inputs.csv")
    inference.add_argument("--data", type=Path, required=True, help="prepared dataset directory")
    inference.add_argument("--weights", type=Path, default=Path("weights"))
    inference.add_argument("--output", type=Path, required=True, help="new prediction directory")
    evaluation = commands.add_parser("score", help="score a complete set of native-resolution binary PNGs")
    evaluation.add_argument("dataset", choices=DATASETS)
    evaluation.add_argument("--data", type=Path, help="prepared directory (default: data/<dataset>)")
    evaluation.add_argument("--predictions", type=Path, required=True, help="one <episode_id>.png per task")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.dataset, args.source, args.output)
        elif args.command == "predict":
            predict(args)
        else:
            score(args)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
