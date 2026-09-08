# GraphTCN

Implementation of GraphTCN, a geometric extension of ModernTCN for multivariate time-series forecasting.

This repository contains the final financial and weather forecasting pipelines used in the dissertation. Note that all exploratory notebooks and data plotting utilities have been excluded from this repo.

## Repository structure

```text
GraphTCN/
├── external/
│   ├── ModernTCN/              # ModernTCN Git submodule
│   └── Kronos/                 # Kronos Git submodule
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── finance.py          # Financial data loading, windowing and graph prior
│   │   ├── tokens.py           # Kronos tokenisation and token datasets
│   │   └── weather.py          # Weather data loading, splits and normalisation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── graph_tcn.py        # GraphTCN architecture
│   │   ├── token_graph_tcn.py  # Token GraphTCN architecture
│   │   └── baselines.py        # Statistical, ModernTCN and Kronos baselines
│   │
│   ├── __init__.py
│   ├── training.py             # Finance, token and weather training functions
│   └── evaluation.py           # Forecasting metrics and evaluation functions
│
├── .gitignore
├── .gitmodules
└── README.md
