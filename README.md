# BS_EventDetection

Proof-of-concept sound event *spotter* for bowel sounds (b / mb / h).
This repo has two tracks:
- `main`: clean pipeline (config-driven, minimal examples).
- `investigating`: sandbox for notebooks, throwaway code, quick tests.

## Layout
- `configs/spotter.yaml` — dataset, features, model, train, infer.
- `scripts/*.py` — build coverage, train, infer CLIs.
- `src/bsed/*` — library code (I/O, preprocessing, labeling, viz, train loop, postprocess).

