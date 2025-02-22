
# Check we are in a venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Not in a venv"
    exit 1
fi

echo "Updating pip"
python -m pip install --upgrade pip

echo "Installing dev dependencies"
python -m pip install -U -r dev-requirements.txt

echo "Black formatting:"
python -m black SampleEfficientRL/

echo "Isort formatting:"
python -m isort --profile black SampleEfficientRL/

echo "Installing package:"
pip install -e .

echo "Linting:"
python -m flake8 SampleEfficientRL/

echo "Mypy type checking:"
python -m mypy --strict SampleEfficientRL/

echo "Testing:"
python -m pytest SampleEfficientRL/tests/
