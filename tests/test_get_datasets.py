import csv
import hashlib
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def test_davis_uses_palette_indices_and_rechecks_existing_output(tmp_path):
    for filename in ("get_datasets.py", "benchmark.py"):
        shutil.copy(BENCHMARKS / filename, tmp_path)
    dataset = "davis-2017-ics"
    metadata = tmp_path / dataset
    metadata.mkdir()
    (metadata / "tasks.csv").write_text(
        "episode_id,sequence,instance_id,reference_frame,target_frame\nexample,clip,2,00000,00001\n"
    )
    indexed = Image.frombytes("P", (2, 2), bytes([2, 1, 255, 2]))
    indexed.putpalette([0, 0, 0] * 256)
    png = io.BytesIO()
    indexed.save(png, format="PNG")
    jpeg = io.BytesIO()
    Image.new("RGB", (2, 2)).save(jpeg, format="JPEG")
    expected_mask = bytes([1, 0, 0, 1])
    downloads = tmp_path / "local downloads"
    downloads.mkdir()
    source = downloads / "DAVIS-2017-trainval-480p.zip"
    with zipfile.ZipFile(source, "w") as archive, (metadata / "checksums.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "width", "height", "sha256"])
        for frame, directory in (("00000", "reference_masks"), ("00001", "ground_truth")):
            archive.writestr(f"DAVIS/JPEGImages/480p/clip/{frame}.jpg", jpeg.getvalue())
            archive.writestr(f"DAVIS/Annotations/480p/clip/{frame}.png", png.getvalue())
            writer.writerow([f"images/clip/{frame}.jpg", 2, 2, hashlib.sha256(jpeg.getvalue()).hexdigest()])
            writer.writerow([f"{directory}/example.png", 2, 2, hashlib.sha256(expected_mask).hexdigest()])

    command = [sys.executable, "get_datasets.py", "--source-dir", str(downloads)]
    incomplete = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert incomplete.returncode != 0 and "mask_images_json_sysu.zip" in incomplete.stderr
    assert "Missing " + str(source) not in incomplete.stderr
    assert not (tmp_path / "data").exists()

    command.extend(["--dataset", dataset])
    subprocess.run(command, cwd=tmp_path, check=True)
    source.unlink()
    verified = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Verified existing" in verified.stdout

    query_mask = tmp_path / "data" / dataset / "ground_truth/example.png"
    Image.new("L", (2, 2)).save(query_mask)
    altered = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert altered.returncode != 0 and "ground_truth/example.png" in altered.stderr


def test_missing_archives_point_to_both_official_sources(tmp_path):
    command = [sys.executable, str(BENCHMARKS / "get_datasets.py")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "DAVIS-2017-trainval-480p.zip" in result.stderr
    assert "davischallenge.org/davis2017/code.html" in result.stderr
    assert "mask_images_json_sysu.zip" in result.stderr
    assert "ShuangLI59/person_search#experiments" in result.stderr
    assert "Dingyuan-Zheng/maskPS#cuhk-sysu-dataset" in result.stderr
    assert not (tmp_path / "data").exists()
