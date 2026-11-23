# SegPFN inference guide

Run the commands below from the repository root after installing SegPFN and downloading both checkpoints. [Quick start](../README.md#quick-start)

## Reference masks and refinement modes

For multi-shot inference, repeat `--reference IMAGE MASK` in the original reference order. Supply a single-channel or palette-indexed mask aligned to each reference image. Binary masks can use 0/1 or 0/255. For an indexed mask, `--foreground-value 1 --ignore-value 255` selects label 1 and excludes void pixels. SegPFN resizes a mask on a different-sized grid to its reference image using nearest-neighbor interpolation.

| `--mode` | Cluster smoothing | Reference-guided correction | Boundary CRF |
| --- | --- | --- | --- |
| `full` (default) | On | On | On |
| `no-refinement` | On | Off | On |
| `crf-only` | Off | Off | On |

Select one mode per prediction. All three apply min–max normalization to the target scores, use a 0.5 threshold, and run the same boundary CRF.

```bash
segpfn --query examples/query.png --reference examples/reference.png examples/reference-mask.png \
  --mode no-refinement --output outputs/no-refinement.png
segpfn --query examples/query.png --reference examples/reference.png examples/reference-mask.png \
  --mode crf-only --output outputs/crf-only.png
```

## Python API

Load the model once and call `predict` for each target image. Pass references as a list of Pillow image and mask pairs. The result is a boolean array with the target image's height and width.

```python
from PIL import Image
from segpfn.model import SegPFN

model = SegPFN("weights")
with (
    Image.open("examples/query.png") as query,
    Image.open("examples/reference.png") as reference,
    Image.open("examples/reference-mask.png") as mask,
):
    prediction = model.predict(query, [(reference, mask)])
```

## Reproducing a prediction

- **Patch geometry.** RGB images are resized along the long edge to 1024 and centered on an ImageNet-mean canvas. At least half of a 16 × 16 patch must be valid. For reference patches, at least half the valid pixels must belong to the target for the patch to count as foreground.
- **Reference views.** One-shot inference retains the original reference. A foreground fraction strictly between 0 and 5% adds two weak color views. Otherwise, four color and geometry views keep the foreground bounding box in frame, retain every foreground component, and require at least the original counts of foreground pixels and patches. Multiple references use their original views and append RGB and XY patch columns.
- **Tabular contexts.** The raw and position-debiased feature branches fit separate full-SVD PCAs on original reference rows. Each retains 128 of up to 192 components and uses two deterministically sampled TabPFN contexts of up to 1,024 rows. The two branch probabilities are averaged equally.
- **Score refinement.** Cluster smoothing mixes each patch probability with its cluster mean in equal proportions. Average linkage uses a cosine distance of 0.4 (similarity 0.6). Reference-guided correction computes a FoRIS seed and bounds its effect using class-matching evidence calibrated on held-out reference blocks. The maximum change to the normalized patch score is 0.25. If the reference supplies no usable calibration fold, the baseline score passes through.
- **Output geometry.** Each padding patch takes the score of the nearest valid patch. The scores are resized bilinearly to the canvas. A ten-iteration Frank–Wolfe dense CRF refines the thresholded boundary. The final mask is cropped to the image region, resized bilinearly, and thresholded at 0.5.

Sampling uses seed 17. By default, the episode identifier derives from the input pixels, the reference masks, and the reference order. Pass `--episode-id ID` to replay a recorded episode with its original sampling identity.

The reference runtime uses an A100, Python 3.12.4, FP32, and the pinned package versions. The predictor disables TF32. Export these variables before starting Python to match the reference process settings. The reference x86 hosts also used `OPENBLAS_CORETYPE=SkylakeX`. Use that override only on CPUs supporting its instruction set. Hardware and numerical library changes can affect borderline pixels. [Package versions](../pyproject.toml)

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Check the installation

```bash
python -m pip install '.[test]'
python -m pytest -q
```

Geometry, mask decoding, context sampling, and score tests run on CPU. The three image-to-mask tests also run when CUDA and the `weights/` directory are available. Set `SEGPFN_TEST_WEIGHTS` to test a different checkpoint directory.
