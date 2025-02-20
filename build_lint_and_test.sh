pip install -U -r dev-requirements.txt
python -m black .
python -m isort .
python -m flake8 .
python -m mypy .
pip install -e .
python -m pytest SampleEfficientRL/tests/
