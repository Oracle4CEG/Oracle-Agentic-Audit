# Models and archived inference implementation

`tabular.py` refits the original logistic ensemble, a stronger LightGBM baseline and a training-prior baseline. Training uses 489 cases; calibrators use only the 161 validation cases.

`run_required.py` implements supplemental Qwen inference and calls the archived implementation in `reference/`. Recorded prompts, sampling seeds, serving settings and checkpoint hashes are in reference code, experiments/config.json and manifests/. The public notebook reuses saved responses and does not contact the serving endpoint. A public immutable checkpoint revision and training-data cutoff remain unverified; the recorded local service name must not be treated as an independently verified public checkpoint identity.
