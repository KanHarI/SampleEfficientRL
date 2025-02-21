echo "Installing dev dependencies"
pip install -U -r dev-requirements.txt

echo "Black formatting:"
python -m black SampleEfficientRL/

echo "Isort formatting:"
python -m isort --profile black SampleEfficientRL/

echo "Installing package:"
pip install -e .

echo "Linting:"
python -m flake8 SampleEfficientRL/

echo "Mypy type checking:"
python -m mypy SampleEfficientRL/

echo "Testing:"
python -m pytest SampleEfficientRL/tests/
