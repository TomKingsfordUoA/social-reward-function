#!/bin/bash

cd "$(dirname "$0")/.."
set -e

echo '=== Linting ==='
flake8 social_robotics_reward

echo '=== Static Type Checking ==='
mypy --strict social_robotics_reward

# echo '=== Testing ==='