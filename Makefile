PYTHON ?= python3
.PHONY: smoke test template-check release-check download replay verify
download:
	$(PYTHON) -m experiments.download
replay:
	$(PYTHON) -m analysis.replay --retrain --out outputs/replay
verify:
	$(PYTHON) -m analysis.verify_replay --reference outputs/final --candidate outputs/replay
smoke:
	$(PYTHON) -m oracle_audit smoke
test:
	$(PYTHON) -m unittest discover -s tests -v
template-check:
	$(PYTHON) -m oracle_audit template-check
release-check:
	$(PYTHON) -m oracle_audit release-check
