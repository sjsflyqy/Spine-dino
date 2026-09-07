# Uploading Tasks 02--09

Upload the repository while preserving relative paths. In particular, do not
move `task08` or `task09` independently after upload: Task 09 intentionally
references 802 PNG images stored once under Task 08.

Recommended upload unit:

```text
spine-dino/
├── backbone/
├── configs/
├── Downstream/
├── tools/
└── requirements-downstream.txt
```

The local data package is approximately 11.23 GiB. Do not create a second ZIP
copy on a disk with less than that amount of free working space. Uploading with
`rsync`, SFTP, SCP, or a sync client is safer because interrupted transfers can
resume.

After upload:

```bash
cd /path/to/spine-dino
python tools/verify_portable_downstream_02_09.py
```

Expected result:

```text
02: 800
03: 2077
04: 579
05: 714
06: 338
07: 1016
08: 1207
09: 1310
total: 8041
```

Task 09 must retain its official split:

```text
train: 919
val:   253
test:  138
```

Model checkpoints and Hugging Face weights are not included. Install the
runtime and download weights on the GPU server:

```bash
pip install -r requirements-downstream.txt
python tools/smoke_test_backbone.py --model microsoft/rad-dino
python tools/smoke_test_backbone.py --model microsoft/rad-dino-maira-2
```
