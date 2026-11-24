# Get DAVIS-2017-ICS and CUHK-SYSU-ICS

## Download the sources

1. **DAVIS.** Download `DAVIS-2017-trainval-480p.zip` under **Semi-supervised**, **TrainVal – Images and Annotations**, **480p**. The archive includes both JPEG frames and indexed instance masks. [Official DAVIS downloads](https://davischallenge.org/davis2017/code.html) · [TrainVal 480p ZIP](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)
2. **CUHK-SYSU and maskPS.** Request academic access from the CUHK-SYSU authors at `tong.xiao.work@gmail.com`. After obtaining access, download `mask_images_json_sysu.zip` from the **CUHK-SYSU Dataset** section of the maskPS authors’ page. Its JSON annotations include the JPEG bytes and polygon coordinates used for this benchmark. [Official CUHK-SYSU access instructions](https://github.com/ShuangLI59/person_search#experiments) · [Official maskPS downloads](https://github.com/Dingyuan-Zheng/maskPS#cuhk-sysu-dataset)

## Build both benchmarks

Leave both ZIPs in the same folder under their original names. From the SegPFN repository root, run the following command, replacing `~/Downloads` with that folder. Preparation runs on CPU. On first use, `uv` installs Pillow 10.4.0 to reproduce the polygon boundaries. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv run benchmarks/get_datasets.py --source-dir ~/Downloads
```

The script reads your local archives and writes **53 DAVIS tasks** to `data/davis-2017-ics/` and **151 CUHK tasks** to `data/cuhk-sysu-ics/`. It checks each JPEG file and each mask’s pixel values against the paper’s SHA-256 checksums. Running it again verifies the existing data.

To build a single benchmark, append `--dataset davis-2017-ics` or `--dataset cuhk-sysu-ics`. For archives in separate locations or extracted directories, use `--davis-source PATH` and `--cuhk-source PATH`. With Python 3.12 and Pillow 10.4.0 installed, the same command runs offline with `python` in place of `uv run`.

## Use the prepared data

Open `inputs.csv` in either dataset directory. Each row gives an `episode_id`, `reference_image`, `reference_mask`, and `target_image`. These paths are relative to that directory. The evaluation mask is `ground_truth/<episode_id>.png`. Each mask uses 1 for the selected instance and 0 for background at the original image size.

In DAVIS-2017-ICS, changing the reference mask selects a different instance in the same video frame pair. Each task records the sequence, frame numbers, and instance label. In CUHK-SYSU-ICS, the reference mask identifies the person to segment in a second image containing other people. Each task records the identity, image filenames, and maskPS polygon indices. [DAVIS task list](davis-2017-ics/tasks.csv) · [CUHK task list](cuhk-sysu-ics/tasks.csv)

<details>
<summary>Run SegPFN or score predictions from another model</summary>

With SegPFN and its checkpoints installed, run the predictor in the model environment. For CUHK, substitute `cuhk-sysu-ics` in both paths. [SegPFN installation and checkpoints](../README.md#quick-start)

```bash
python benchmarks/benchmark.py predict --data data/davis-2017-ics \
  --weights weights --output outputs/davis-2017-ics
```

To score a model, save one binary PNG named `<episode_id>.png` per task at the target image’s original size. The scorer reads the prepared ground truth and accepts 0/1 or 0/255 predictions. If you used a custom output location, pass the prepared dataset directory with `--data`.

```bash
python benchmarks/benchmark.py score davis-2017-ics --predictions outputs/davis-2017-ics
python benchmarks/benchmark.py score cuhk-sysu-ics --predictions outputs/cuhk-sysu-ics
```

</details>

## Sources and terms

- **DAVIS.** Pont-Tuset and colleagues provide the semi-supervised TrainVal 480p ZIP as a direct download. Its README specifies CC BY-NC 4.0 and points to video-specific notices in `SOURCES.md`. Each SegPFN mask selects one instance from the original indexed annotation. [DAVIS download and citation](https://davischallenge.org/davis2017/code.html) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **CUHK-SYSU and maskPS.** Xiao and colleagues provide academic access to the original person-search dataset. Zheng and colleagues provide the maskPS polygons. Son and colleagues report that the parent CUHK-SYSU terms restrict use to noncommercial research and education and prohibit redistribution. The SegPFN benchmark release provides task lists, checksums, and code to reconstruct the data locally. [CUHK-SYSU authors](https://github.com/ShuangLI59/person_search#experiments) · [maskPS authors](https://github.com/Dingyuan-Zheng/maskPS#cuhk-sysu-dataset) · [Son et al., supplemental license discussion](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Son_Towards_Robustness_of_ICCV_2025_supplemental.pdf)
