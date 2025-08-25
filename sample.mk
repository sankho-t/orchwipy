include Makefile

sample-template.yml: tests/sample.py
	$(PY) tests/sample.py lambda --create-template sample-template.yml --code-path ./tests/sample.py

sample.dot: tests/sample.py
	$(PY) tests/sample.py check --save-call-graph sample.dot
