allsrc := $(wildcard orchwipy/*.py)

PY ?= ${CURDIR}/.venv/bin/python3.12
PIP_INSTALL ?= $(PY) -m pip install -q

.PHONY: client_code

client_code = $(allsrc) pyproject.toml

install: $(client_code)
	@rm -rf layer
	$(PIP_INSTALL) . --force-reinstall

layer: $(client_code)
	@rm -rf layer/python/orchwipy-*
	$(PIP_INSTALL) . --platform manylinux2014_x86_64 --only-binary=:all: -t $@/python

example.py: README.md
	$(PY) py_from_md.py README.md > $@

example.dot: example.py
	$(PY) $< check --save-call-graph $@

example.svg: example.dot
	dot -Tsvg example.dot > $@

example.template.yml: example.py
	$(PY) example.py

tests = $(wildcard tests/test_*.py)

test-all:
	$(PY) -m unittest tests/test_*.py

test-make-template:
	$(MAKE) sample-template.yml -f sample.mk

.SUFFIXES:
