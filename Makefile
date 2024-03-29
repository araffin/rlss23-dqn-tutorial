LINT_PATHS=dqn_tutorial/ tests/

pytest:
	python3 -m pytest --cov-report html --cov-report term --cov=. -v --color=yes -m "not expensive"

mypy:
	mypy ${LINT_PATHS}

missing-annotations:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports dqn_tutorial

# missing-docstrings:
# 	pylint -d R,C,W,E -e C0116 dqn_tutorial -j 4

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

.PHONY: clean spelling doc lint format check-codestyle commit-checks
