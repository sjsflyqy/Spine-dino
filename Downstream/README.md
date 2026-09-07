# Downstream benchmark

`data/` is a shared, deduplicated image pool for Tasks 02--09. Each numbered
task owns its annotations, split definitions, model head and evaluation code.

Do not place model checkpoints or experiment outputs inside `data/`.

Validated logical image counts:

| Task | Images |
|---|---:|
| 02 BUU400 | 800 |
| 03 VinDr test | 2,077 |
| 04 AASCE 2019 | 579 |
| 05 Spondylolisthesis landmarks | 714 |
| 06 Three-class spine set | 338 |
| 07 Whole-spine paired views | 1,016 |
| 08 Flexion/extension landmarks | 1,207 |
| 09 Vertebra numbering | 1,310 |

Task 09 shares 802 source PNGs with Task 08. They are stored once and
referenced by relative path.
