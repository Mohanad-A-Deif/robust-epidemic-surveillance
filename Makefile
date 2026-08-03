.PHONY: test verify data quick manifest

test:
	python scripts/run_tests.py

verify:
	python scripts/validate_repository.py

data:
	python scripts/reproduce_data.py

quick:
	python scripts/run_quick_smoke.py

manifest:
	python scripts/build_manifest.py
