# Spine-DINO

This workspace contains a portable downstream benchmark for spine X-rays.

The package is designed to be uploaded to a Linux training server. Dataset
paths are relative to the repository and no experiment depends on the original
Windows `F:` or `H:` drives.

## First targets

- `Downstream/07_ningbo_whole_spine_keypoint`
- `Downstream/08_ningbo_flexion_extension_keypoint`
- `Downstream/09_vertebra_numbering`

RAD-DINO and RAD-DINO-MAIRA-2 are exposed through one backbone interface. A
task changes the backbone through its YAML config rather than changing task
code.

## Build the portable data package

The local source paths are recorded only in the builder:

```powershell
python tools/build_portable_downstream.py
python tools/verify_portable_downstream.py
```

The generated `Downstream/data` directory is self-contained and can be
uploaded with the repository.

## Backbone smoke test

On a machine with the models downloaded:

```bash
python tools/smoke_test_backbone.py --model microsoft/rad-dino
python tools/smoke_test_backbone.py --model microsoft/rad-dino-maira-2
```

The model weights are intentionally not stored in this repository.
