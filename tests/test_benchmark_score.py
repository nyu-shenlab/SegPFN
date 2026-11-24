import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_score_pools_foreground_and_requires_every_task(tmp_path):
    script = tmp_path / "benchmark.py"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "benchmarks" / script.name, script)
    dataset = tmp_path / "davis-2017-ics"
    predictions = tmp_path / "predictions"
    dataset.mkdir()
    predictions.mkdir()
    data = tmp_path / "prepared"
    (data / "ground_truth").mkdir(parents=True)

    (dataset / "tasks.csv").write_text("episode_id\nsmall\nlarge\n")
    with (dataset / "checksums.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "width", "height", "sha256"])
        for name, labels in (("small", [[1, 0], [0, 0]]), ("large", [[1, 1], [1, 1]])):
            mask = np.array(labels, dtype=np.uint8)
            member = "ground_truth/" + name + ".png"
            Image.fromarray(mask).save(data / member)
            writer.writerow([member, 2, 2, hashlib.sha256(mask.tobytes()).hexdigest()])

    command = [
        sys.executable,
        str(script),
        "score",
        dataset.name,
        "--data",
        str(data),
        "--predictions",
        str(predictions),
    ]
    Image.fromarray(np.array([[255, 0], [0, 0]], np.uint8)).save(predictions / "small.png")
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0 and "large.png" in missing.stderr

    Image.fromarray(np.zeros((2, 2), np.uint8)).save(predictions / "large.png")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "intersection=1 | union=5 | pooled foreground IoU=20.0000%" in result.stdout
