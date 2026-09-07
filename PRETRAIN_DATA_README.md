# SpinePretrain-v1

## Completed local build

The validated build is located at:

```text
H:\SpinePretrain-v1
```

Final audited counts:

| Source | Candidates | Included unique |
|---|---:|---:|
| NHANES II | 16,531 | 16,531 |
| BUU2000 AP | 2,000 | 1,998 |
| BUU2000 LAT | 2,000 | 1,999 |
| VinDr train | 8,389 | 8,372 |
| CSXA | 4,963 | 4,963 |
| Nanning | 20,558 | 20,558 |
| Ningbo | 14,493 | 6,149 |
| Total | 68,934 | 60,570 |

There are 8,364 byte-identical exclusions, recorded without deleting any
source file. Ningbo accounts for 8,344 of these exclusions.

The final WebDataset representation contains 61 uncompressed tar shards,
79,296,440,320 bytes in total. Both full image SHA-256 and full shard SHA-256
validation passed on the local build.

The build command creates two byte-identical image representations:

```powershell
python tools/build_spine_pretrain_v1.py --output H:\SpinePretrain-v1
```

## Direct-image DINOv2 mode

Use:

```text
H:\SpinePretrain-v1\spine_dino_dataset\train\spine
H:\SpinePretrain-v1\spine_dino_dataset\extra
```

Dataset configuration:

```text
SpineDINO:split=TRAIN:root=/data/SpinePretrain-v1/spine_dino_dataset:extra=/data/SpinePretrain-v1/spine_dino_dataset/extra
```

## WebDataset mode

Upload:

```text
H:\SpinePretrain-v1\shards\train-*.tar
```

Use `vitb14_spine_webdataset.yaml` and update its shard and weight paths.
Shards are uncompressed tar files, contain at most 1,000 samples or 2 GiB of
image payload, and store `<key>.<image extension>` plus `<key>.json`.

## Validation

Quick image-directory validation (shards are optional):

```powershell
python tools/verify_spine_pretrain_v1.py --root H:\SpinePretrain-v1
```

Full byte-level validation:

```powershell
python tools/verify_spine_pretrain_v1.py `
  --root H:\SpinePretrain-v1 `
  --full-image-hash `
  --full-shard-hash
```

## Linux server upload and validation

Upload either or both of:

```text
SpinePretrain-v1/spine_dino_dataset
SpinePretrain-v1/shards
```

Always upload:

```text
SpinePretrain-v1/metadata
```

Validate individual transferred shards:

```bash
cd /data/SpinePretrain-v1/shards
for checksum in *.tar.sha256; do sha256sum -c "$checksum"; done
```

Run the repository validator from anywhere inside the repository. When
`SpinePretrain-v1` is located in the repository root, `--root` is optional:

```bash
python tools/verify_spine_pretrain_v1.py
```

For a second full byte-level audit on the server:

```bash
python tools/verify_spine_pretrain_v1.py \
  --root /data/SpinePretrain-v1 \
  --full-image-hash \
  --check-shards \
  --full-shard-hash

If only the direct-image dataset was uploaded, omit `--check-shards` and
`--full-shard-hash`. To validate all uncompressed tar files later, run:

```bash
python tools/verify_spine_pretrain_v1.py --check-shards
```
```
