# vector-fragmented: 0393-0004 (vector)

Ground-truth lines: 807
Vector elements per ground-truth line: 9.9

| setting | precision | recall | f1 | endpoint_err_median | length_error_pct | elements | seconds |
|---|---|---|---|---|---|---|---|
| vector | 1.0 | 1.0 | 1.0 | 0.0 | 0.1407 | 7967 | 0.3848 |
| raster-150 | 0.9983 | 0.9544 | 0.9758 | 0.7283 | 9.0432 | 439 | 0.1111 |
| raster-300 | 0.9949 | 0.9734 | 0.984 | 0.6476 | 4.1195 | 515 | 0.4006 |
| raster-600 | 0.989 | 0.9805 | 0.9847 | 0.6151 | 0.8456 | 641 | 1.9175 |

Recall by class (vector), lowest first:

| class | gt lines | recall |
|---|---|---|
| unlabelled | 363 | 1.0 |
| refrigerator | 27 | 1.0 |
| squat toilet | 30 | 1.0 |
| bath | 30 | 1.0 |
| bath tub | 107 | 1.0 |
| elevator | 53 | 1.0 |
| escalator | 35 | 1.0 |
| single door | 81 | 1.0 |
| blind window | 19 | 1.0 |
| sliding door | 34 | 1.0 |
| revolving door | 4 | 1.0 |
| urinal | 24 | 1.0 |

Overlay (local only): runs/overlays/0393-0004/vector.png. Blue: extracted. Red: missed ground truth. Orange: extracted but not in ground truth.
