# SegPFN

Segment an object in a query image using one or more reference images annotated for that object. SegPFN combines DINOv3-L/16 patch features and a frozen TabPFN 3.5 classifier. The output is a single-channel PNG at the original query size, with background 0 and foreground 255.

## Install

Use Linux, Python 3.12, an NVIDIA GPU, a C++ compiler, and a CUDA toolkit matching your PyTorch build. The following commands use the CUDA 13.0 build. Run the CRF build on the target GPU or set `TORCH_CUDA_ARCH_LIST` to its compute capability.

```bash
git clone https://github.com/nyu-shenlab/SegPFN.git
cd SegPFN
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install .
python scripts/install_crf.py
```

The CRF installer verifies the pinned upstream archive and builds its filtering extension against the active PyTorch installation. If `nvcc` is outside the default search path, set `CUDA_HOME` to the toolkit directory. For offline installation, pass the pinned zip to `scripts/install_crf.py --archive FILE`. [Dense CRF source](https://github.com/netw0rkf10w/CRF/tree/13a123f7cd3ea1f975e6c483b1e7c52112d7c151)

## Get the checkpoints

Obtain access to both model repositories through your Hugging Face account, then download the following revisions. The predictor checks the file hashes once at startup. Use `--weights DIR` if the downloads are outside `weights/`, keeping the same directory layout. [DINOv3-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) · [TabPFN 3.5](https://huggingface.co/Prior-Labs/tabpfn_3_5)

```bash
hf auth login
hf download facebook/dinov3-vitl16-pretrain-lvd1689m config.json preprocessor_config.json model.safetensors \
  --revision ea8dc2863c51be0a264bab82070e3e8836b02d51 --local-dir weights/dinov3-large
hf download Prior-Labs/tabpfn_3_5 tabpfn-v3.5-20260909.safetensors \
  --revision 06bf2ba35c80a92a3b9abb436b99cf49e7a0365e --local-dir weights
```

## Predict

The three small input images below are included in the checkout.

```bash
segpfn --query examples/query.png --reference examples/reference.png examples/reference-mask.png \
  --output outputs/mask.png
```

| `--mode` | Cluster smoothing | Reference-guided refinement | Boundary CRF |
| --- | --- | --- | --- |
| `full` (default) | On | On | On |
| `no-refinement` | On | Off | On |
| `crf-only` | Off | Off | On |

Select a single mode per prediction. All three use query min–max normalization, a 0.5 threshold and the same boundary CRF. Disabled stages are skipped.

```bash
segpfn --query examples/query.png --reference examples/reference.png examples/reference-mask.png \
  --mode no-refinement --output outputs/no-refinement.png
segpfn --query examples/query.png --reference examples/reference.png examples/reference-mask.png \
  --mode crf-only --output outputs/crf-only.png
```

Repeat `--reference IMAGE MASK` for multi-shot inference, in the original reference order. Masks may use 0/1 or 0/255. For indexed masks, supply the target class with `--foreground-value`. For example, `--foreground-value 1 --ignore-value 255` selects label 1 and excludes void pixels. Use single-channel or palette-indexed masks registered to the reference image. A mask on a different-sized grid is resized to that image using nearest-neighbor interpolation.

The same interface is available with `python -m segpfn`. In Python, reuse a model across queries.

```python
from PIL import Image
from segpfn.model import SegPFN

model = SegPFN("weights")
with Image.open("examples/query.png") as query, Image.open("examples/reference.png") as ref, \
     Image.open("examples/reference-mask.png") as mask:
    prediction = model.predict(query, [(ref, mask)])  # bool array, (height, width)
```

## Reproducing a prediction

* RGB images are resized along the long edge to 1024 and centered on an ImageNet-mean canvas. At least half of a 16 × 16 patch must be valid. For support patches, at least half the valid pixels must belong to the target for the patch to count as foreground.
* One-shot inference retains the original reference. A foreground fraction strictly between 0 and 5% adds two weak color views. Otherwise four color/geometry views protect the foreground bounding box, connected components, pixel count and foreground patch count. Multiple references use their original views and append RGB and XY patch columns.
* Raw and position-debiased branches fit separate full-SVD PCAs on original support rows. Each retains 128 of up to 192 components. Each branch uses two deterministically sampled TabPFN contexts of up to 1,024 rows. The two branch probabilities are averaged equally.
* Cluster smoothing mixes each patch probability with its cluster mean in equal proportions. Average linkage uses cosine distance 0.4 (similarity 0.6). Reference-guided refinement computes a FoRIS seed and bounds its effect using class-matching evidence calibrated on held-out support blocks. The maximum change to the normalized patch score is 0.25. When the reference supplies no usable calibration fold, the baseline score passes through.
* Valid patch scores extend into padding by nearest valid patch, then resize bilinearly to the canvas. A ten-iteration Frank–Wolfe dense CRF refines the thresholded boundary. The final mask is cropped to the image region, resized bilinearly and thresholded at 0.5.

Sampling uses seed 17. By default the episode identifier derives from the input pixels, the reference mask contents and reference order. Pass `--episode-id ID` to replay a recorded episode with its original sampling identity.

The reference runtime uses an A100, Python 3.12.4, FP32, and the package versions in `pyproject.toml`. The predictor disables TF32. To match the reference process settings, export these variables before starting Python. The reference x86 hosts also used `OPENBLAS_CORETYPE=SkylakeX`. Use that override only on a CPU supporting its instruction set. Hardware and numerical library changes can affect borderline pixels.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

To check the installation, run `python -m pip install '.[test]'` followed by `python -m pytest -q`. Geometry, mask decoding, context sampling and score tests run on CPU. The three image-to-mask tests also run when CUDA and the `weights/` directory are available. Set `SEGPFN_TEST_WEIGHTS` to test another checkpoint directory.

## Upstream components

The seed construction and narrow-band CRF integration adapt FoRIS. The included 1024 × 500 positional basis was computed once from the DINOv3-L response to a constant image. Its SHA-256 is `6838d91c843bb59d72ccb458e58c82a9a7d705a60f8325fd97a876eae3fc8037` (compressed file). The corresponding license texts accompany the distribution. [FoRIS revision](https://github.com/Xi-Mu-Yu/FoRIS/tree/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e) · [FoRIS license](licenses/FoRIS-Apache-2.0.txt) · [DINOv3 license](licenses/DINOv3.txt)
