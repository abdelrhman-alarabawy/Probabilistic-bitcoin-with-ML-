# Probabilistic-bitcoin-with-ML

A Python-based framework designed to quantify risk in Bitcoin trading using probabilistic machine learning. The goal is to model uncertainty explicitly instead of only predicting point estimates.

## Repo layout

- `data/`: local datasets (raw, interim, processed, external)
- `notebooks/`: exploration and prototyping
- `src/bitcoin_probabilistic_learning/`: reusable package code
- `scripts/`: command-line utilities and one-off tasks
- `reports/`: generated reports and figures
- `models/`: saved model artifacts
- `tests/`: unit and integration tests
- `docs/`: design notes and documentation

## Quickstart

1) Create a virtual environment
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
2) Install the package for development
   - `pip install -e .`

## Data policy

- Place raw inputs under `data/raw/` and derived datasets under `data/processed/`.
- CSV files are ignored by git by default.
