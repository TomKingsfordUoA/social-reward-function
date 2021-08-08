#!/bin/bash

cd "$(dirname "$0")/.."
set -e

echo '=== Linting ==='
flake8 social_robotics_reward *.py

echo '=== Static Type Checking ==='
mypy --strict social_robotics_reward *.py

# echo '=== Testing ==='
pytest -v .
