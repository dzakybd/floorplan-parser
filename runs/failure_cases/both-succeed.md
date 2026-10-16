# both-succeed: 0674-0822 (raster-300)

Ground-truth lines: 114
Vector elements per ground-truth line: 1.0

| setting | precision | recall | f1 | endpoint_err_median | length_error_pct | elements | seconds |
|---|---|---|---|---|---|---|---|
| vector | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 114 | 0.0051 |
| raster-150 | 0.9997 | 0.9972 | 0.9985 | 0.2426 | 1.3194 | 110 | 0.0568 |
| raster-300 | 0.9999 | 0.9996 | 0.9998 | 0.235 | 0.5922 | 102 | 0.2468 |
| raster-600 | 0.9995 | 0.9996 | 0.9996 | 0.331 | 0.0006 | 104 | 1.2957 |

Recall by class (raster-300), lowest first:

| class | gt lines | recall |
|---|---|---|
| unlabelled | 84 | 0.9992 |
| railing | 30 | 1.0 |

Overlay (local only): runs/overlays/0674-0822/raster-300.png. Blue: extracted. Red: missed ground truth. Orange: extracted but not in ground truth.
