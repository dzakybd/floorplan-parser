# raster-weakest: 1010-0022 (raster-300)

Ground-truth lines: 229
Vector elements per ground-truth line: 2.9

| setting | precision | recall | f1 | endpoint_err_median | length_error_pct | elements | seconds |
|---|---|---|---|---|---|---|---|
| vector | 1.0 | 1.0 | 1.0 | 0.0 | 0.0802 | 667 | 0.0232 |
| raster-150 | 0.8418 | 0.9168 | 0.8777 | 0.5329 | 2.3511 | 1217 | 0.1715 |
| raster-300 | 0.8382 | 0.9494 | 0.8903 | 0.4498 | 11.9148 | 1371 | 0.4325 |
| raster-600 | 0.8174 | 0.9539 | 0.8804 | 0.4727 | 25.747 | 1675 | 1.6881 |

Recall by class (raster-300), lowest first:

| class | gt lines | recall |
|---|---|---|
| double door | 117 | 0.8897 |
| folding door | 10 | 0.9628 |
| escalator | 17 | 0.9692 |
| unlabelled | 70 | 0.986 |
| single door | 13 | 0.9887 |
| wall | 2 | 1.0 |

Overlay (local only): runs/overlays/1010-0022/raster-300.png. Blue: extracted. Red: missed ground truth. Orange: extracted but not in ground truth.
