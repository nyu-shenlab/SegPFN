"""Build the pinned dense CRF for the active PyTorch/CUDA environment."""

import argparse
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REVISION = "13a123f7cd3ea1f975e6c483b1e7c52112d7c151"
SHA256 = "bee12d74e684cca98bfcd82e90b81f497b3f724f9bd0170f53569947df19daf6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="local copy of the pinned GitHub zip")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp) / "crf.zip"
            urllib.request.urlretrieve(f"https://codeload.github.com/netw0rkf10w/CRF/zip/{REVISION}", archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Unexpected CRF source archive checksum")
        with zipfile.ZipFile(archive) as source:
            source.extractall(tmp)
        root = Path(tmp) / f"CRF-{REVISION}"
        setup = root / "setup.py"
        # Let PyTorch target the installed GPU or TORCH_CUDA_ARCH_LIST.
        setup.write_text(setup.read_text().replace("'nvcc': nvcc_ARCH", "'nvcc': []"))
        filters = root / "src" / "PermutohedralFiltering" / "source"
        for path in (filters / "cpu/LatticeFilterKernel.cpp", filters / "gpu/LatticeFilter.cu"):
            code = path.read_text().replace("input_tensor.type()", "input_tensor.options()")
            code = code.replace("x.type().is_cuda()", "x.is_cuda()").replace(
                ".data<float>()", ".data_ptr<float>()"
            )
            path.write_text(code)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(root)],
            check=True,
        )


if __name__ == "__main__":
    main()
