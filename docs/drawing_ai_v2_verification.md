# Drawing AI v2 refactor — verification report

## Result

The refactor preserves the existing public scoring criteria, criterion order,
maximum weights, total-score formulas, and submission result contract.

## Locked score formulas

- Etalon: `3 + 6 + 8 + 12 + 18 + 10 + 24 + 15 + 4 = 100`.
- Optional: `15 + 10 + 15 + 15 + 10 + 10 = 75`, then the existing
  `round(raw_total / 75 * 100)` conversion.

## Regression snapshot

| Mode | Fixture | Before | After |
|---|---|---:|---:|
| Etalon | identical | 69.00 | 69.00 |
| Etalon | shifted | 55.33 | 55.33 |
| Etalon | missing_views | 32.33 | 32.33 |
| Etalon | noisy | 70.33 | 70.33 |
| Etalon | blank_frame | 18.00 | 18.00 |
| Optional | identical | 88 | 88 |
| Optional | shifted | 88 | 88 |
| Optional | missing_views | 69 | 69 |
| Optional | noisy | 100 | 100 |
| Optional | blank_frame | 16 | 16 |

For all 10 cases, both `total_score` and the full criterion table matched the
pre-refactor baseline exactly.

## Automated checks

- Drawing AI contract/integration tests: **8 passed**.
- Repository test discovery: **52 passed** in this runtime with a temporary,
  test-only pgvector compatibility module because the runner did not have the
  `pgvector` package installed. This compatibility module is not included in
  production code or the final archive.
- Python bytecode compilation: passed for `app`, `tests`, and `scripts`.
- Backward-compatible imports from the two old AI module paths: passed.

## Performance sample

Fresh-process service benchmark, median of three runs on the verification
machine:

- Etalon: 1.88 s → 1.61 s; peak RSS ~301 MB → ~266 MB.
- Optional: 1.12 s → 1.09 s; peak RSS remained ~228 MB.

These values are environment-dependent and should be re-measured under the
production worker configuration.
