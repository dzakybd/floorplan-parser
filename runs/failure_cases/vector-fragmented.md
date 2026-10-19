# vector-fragmented: 0721-0004 (vector)

Ground-truth lines: 123
Vector elements per ground-truth line: 14.3

| setting | precision | recall | f1 | endpoint_err_median | length_error_pct | elements | seconds |
|---|---|---|---|---|---|---|---|
| vector | 1.0 | 1.0 | 1.0 | 0.0 | 0.0971 | 1759 | 0.0844 |
| raster-150 | 0.996 | 0.9849 | 0.9904 | 0.3949 | 2.111 | 135 | 0.0488 |
| raster-300 | 0.9886 | 0.986 | 0.9873 | 0.4261 | 0.8407 | 133 | 0.1668 |
| raster-600 | 0.9862 | 0.9899 | 0.988 | 0.3709 | 2.0108 | 145 | 0.7809 |

Recall by class (vector), lowest first:

| class | gt lines | recall |
|---|---|---|
| unlabelled | 81 | 1.0 |
| toilet | 12 | 1.0 |
| row chairs | 10 | 1.0 |
| single door | 17 | 1.0 |
| blind window | 3 | 1.0 |

Overlay (local only): runs/overlays/0721-0004/vector.png. Blue: extracted. Red: missed ground truth. Orange: extracted but not in ground truth.
