# Experiment records and fixed inputs

`download.py` obtains and verifies immutable public Atlas-input and research-record archives. `scope.py` enforces the necessary-only experiment manifest. `qwen_records.py` verifies and normalizes original Qwen responses. `prepare.py` preserves earlier local source import and separate historical GPT provenance.

The fixed supplemental scope is 483 validation, 320 matched B2, and 480 A1 guard runs. Optional low-temperature experiments are cancelled. `infrastructure_recovery.py` and the archived manifests retain failures, interruptions and superseded attempts. Scheduling modules are for explicitly requested fresh inference; they are not called by the public replay notebook.

`public_artifacts.py` packages existing input/run records into separate Atlas and research archives. It does not recollect the Atlas or generate missing model outputs. See ../docs/PUBLIC_REPRODUCTION.md.
