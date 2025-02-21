pip install -U -r dev-requirements.txt
python -m black SampleEfficientRL/
python -m isort SampleEfficientRL/
pip install -e .
python -m flake8 SampleEfficientRL/
python -m mypy SampleEfficientRL/
python -m pytest SampleEfficientRL/tests/
