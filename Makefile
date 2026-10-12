.PHONY: install test lint validate review eval eval-provisional corpus

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check .

validate:            ## check the answer key against the corpus
	python -m guichet.golden

review:              ## walk through unreviewed golden items
	python -m scripts.review_golden

eval:                ## reviewed items only — the numbers you can report
	python -m guichet.run_eval

eval-provisional:    ## includes unreviewed items — development only
	python -m guichet.run_eval --include-unreviewed --out reports/retrieval-provisional.json

corpus:              ## re-crawl service-public.gouv.fr and rebuild chunks
	python -m guichet.corpus && python -m guichet.chunk
