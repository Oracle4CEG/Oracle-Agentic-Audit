PYTHON ?= python3
.PHONY: smoke test template-check release-check
smoke:
	$(PYTHON) -m oracle_audit smoke
test:
	$(PYTHON) -m unittest discover -s tests -v
template-check:
	$(PYTHON) -m oracle_audit template-check
release-check:
	$(PYTHON) -m oracle_audit release-check
