#!/bin/bash

cd "$(dirname "$0")/.."
set -e

echo '=== Linting ==='
flake8 social_reward_function *.py

echo '=== Static Type Checking ==='
mypy --strict social_reward_function setup.py

echo '=== Testing ==='
pytest -v .

echo '=== Checking Manifest ==='
check-manifest -v

echo '=== Checking setup.py ==='
python setup.py check
