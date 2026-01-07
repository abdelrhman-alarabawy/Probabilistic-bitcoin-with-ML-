# Probabilistic-bitcoin-with-ML
A Python-based framework to quantify risk in Bitcoin trading using probabilistic machine learning. The focus is on modeling uncertainty, not just point forecasts.
## Repo layout
- data/: local datasets (raw, interim, processed, external)
- 
otebooks/: exploration and prototyping
- src/bitcoin_probabilistic_learning/: reusable package code
- scripts/: command-line utilities and one-off tasks
- eports/: generated reports and figures
- models/: saved model artifacts
- 	ests/: unit and integration tests
- docs/: design notes and documentation
## Quickstart
1) Create a virtual environment
   - python -m venv .venv
   - \.\.venv\Scripts\activate
2) Install the package for development
   - pip install -e .
## Data policy
- Place raw inputs under data/raw/ and derived datasets under data/processed/.
- CSV files are ignored by git by default, so keep local datasets in data/.
## Labels
The labeling script lives at scripts/signals_code_hour_v1_0.py. It is configured for 1h labeling of the merged features file in data/processed/.
