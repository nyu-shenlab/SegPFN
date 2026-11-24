# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["Pillow==10.4.0"]
# ///
"""Build the SegPFN benchmarks from local copies of the authors' releases."""

import argparse
import zipfile
from pathlib import Path

from benchmark import DATASETS, prepare, verify_prepared

DAVIS_ARCHIVE = "DAVIS-2017-trainval-480p.zip"
DAVIS_DOWNLOAD = "https://davischallenge.org/davis2017/code.html"
CUHK_ARCHIVE = "mask_images_json_sysu.zip"
CUHK_ACCESS = "https://github.com/ShuangLI59/person_search#experiments"
MASKPS_DOWNLOAD = "https://github.com/Dingyuan-Zheng/maskPS#cuhk-sysu-dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("."),
        help="folder containing the downloaded ZIPs (default: current directory)",
    )
    parser.add_argument("--dataset", choices=DATASETS, help="prepare just one benchmark (default: both)")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="output parent (default: data)")
    parser.add_argument("--davis-source", type=Path, help="local DAVIS TrainVal 480p ZIP or extracted root")
    parser.add_argument("--cuhk-source", type=Path, help="local maskPS CUHK ZIP or extracted JSON directory")
    args = parser.parse_args()

    sources = {
        "davis-2017-ics": args.davis_source or args.source_dir / DAVIS_ARCHIVE,
        "cuhk-sysu-ics": args.cuhk_source or args.source_dir / CUHK_ARCHIVE,
    }
    instructions = {
        "davis-2017-ics": f"Download the semi-supervised TrainVal 480p ZIP from {DAVIS_DOWNLOAD}",
        "cuhk-sysu-ics": (
            f"Request CUHK-SYSU academic access at {CUHK_ACCESS}\n"
            f"Get the CUHK-SYSU annotation archive from the maskPS authors at {MASKPS_DOWNLOAD}"
        ),
    }
    datasets = (args.dataset,) if args.dataset else DATASETS
    missing = [
        f"Missing {sources[dataset]}\n{instructions[dataset]}"
        for dataset in datasets
        if not (args.output_root / dataset).exists() and not sources[dataset].exists()
    ]
    if missing:
        parser.exit(1, "\n\n".join(missing) + "\n")

    try:
        for dataset in datasets:
            output = args.output_root / dataset
            if output.exists() or output.is_symlink():
                verify_prepared(dataset, output)
            else:
                prepare(dataset, sources[dataset], output)
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
