#!/bin/bash

# 1. Create the standard Machine Learning folder structure
echo "Creating project directories..."
mkdir -p src
mkdir -p output
mkdir -p data/raw
mkdir -p data/processed

# 2. Generate the blank placeholder Python files in the src/ directory
echo "Generating pipeline source files..."
touch src/__init__.py
touch src/data_cleaning.py
touch src/train_model.py
touch src/evaluate.py
touch src/predict.py

# 3. Generate root configuration and environment files
echo "Generating root configuration files..."
touch main.py
touch README.md

# 4. Create a .gitignore file so git ignores your large dataset and local venv

echo "Writing .gitignore configurations..."
cat <<EOT > .gitignore
venv/
__pycache__/
*.pyc
.DS_Store
data/raw/*
!data/raw/.gitkeep
EOT

# 5. Keep data folders tracked by Git even if empty initially
touch data/raw/.gitkeep
touch data/processed/.gitkeep

echo "======================================================="
echo "Project structure successfully generated!"
echo "======================================================="
ls -R src/


