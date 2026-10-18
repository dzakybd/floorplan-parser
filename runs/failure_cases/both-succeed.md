# both-succeed: 1261-0083 (raster-300)

Ground-truth lines: 91
Vector elements per ground-truth line: 1.0

| setting | precision | recall | f1 | endpoint_err_median | length_error_pct | elements | seconds |
|---|---|---|---|---|---|---|---|
| vector | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 91 | 0.0046 |
| raster-150 | 0.9998 | 0.9982 | 0.999 | 0.4649 | 0.5175 | 76 | 0.0804 |
| raster-300 | 1.0 | 1.0 | 1.0 | 0.3694 | 0.2463 | 66 | 0.304 |
| raster-600 | 1.0 | 1.0 | 1.0 | 0.3924 | 2.5974 | 68 | 1.9739 |

Recall by class (raster-300), lowest first:

| class | gt lines | recall |
|---|---|---|
| unlabelled | 31 | 1.0 |
| single door | 60 | 1.0 |

Overlay (local only): runs/overlays/1261-0083/raster-300.png. Blue: extracted. Red: missed ground truth. Orange: extracted but not in ground truth.
