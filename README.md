# AI-Based Power Grid Balance and Electricity Spot Price Forecasting in Denmark

## Overview

This repository contains an end-to-end machine learning pipeline designed to model and forecast short-term electricity spot prices and stochastic imbalance prices across Danish bidding zones (DK1 and DK2).

The massive integration of variable renewable sources (wind and solar) has introduced severe volatility into the Danish smart grid. Accurate short-term load and price forecasting is critical for maintaining grid frequency, optimizing energy trading portfolios, and avoiding financial penalties in the regulating market. This project bypasses traditional econometric approaches, utilizing a hybrid AI architecture to handle the non-linear dynamics, mean-reversion, and heteroscedasticity inherent to modern wholesale energy markets.

## Research Question

How can multivariate machine learning (XGBoost) and deep learning (LSTM) models be optimally combined to forecast both deterministic day-ahead spot prices and stochastic imbalance prices (regulating market) across Danish bidding zones, while accounting for high intermittent renewable energy penetration?

## Methodology

The forecasting framework models the conditional expectation of future prices given a high-dimensional state vector of physical grid and meteorological data.

- **LSTM Networks (Long Short-Term Memory):** The primary deep learning architecture used to extract sequential, long-term temporal dependencies from market and grid data.
- **XGBoost (eXtreme Gradient Boosting):** Deployed for its resilience to noisy financial data and superior training efficiency. Integrated with **SHAP (SHapley Additive exPlanations)** values to provide rigorous feature importance and model interpretability.
- **Quantile Regression Forests (QRF):** Applied specifically to the regulating market to capture stochastic price spikes (jump-diffusions). This allows the generation of probabilistic confidence intervals for imbalance prices rather than relying on deterministic point forecasts.

## Data Engineering Pipeline

The project utilizes an automated data ingestion pipeline pulling from three modern RESTful APIs:

1. **Energi Data Service (Energinet):** \* `DayAheadPrices` (Settlement prices)
   - `ConsumptionDK3619CodeHour` (Industrial segmented consumption)
   - Real-time physical flows and mFRR reserve activations.
2. **DMI Open Data API (Danish Meteorological Institute):** \* `metObs` dataset covering `wind_speed`, `radia_glob` (global radiation), and `temp_dry` (dry temperature).
3. **ENTSO-E Transparency Platform:** \* Cross-border energy flows and grid constraints between Denmark, Germany, Norway, and Sweden.

**Engineered Features:**

- Auto-regressive time lags (T-1, T-24, T-168 hours).
- Cyclical encoding (sine/cosine transformations) of calendar variables.
- Geospatial price differentials (e.g., DE vs. DK1 price spreads).

## Development Setup

This project uses `uv` for dependency management and environment isolation.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TeebooGH/dk-power-spot-forecaster.git
   cd dk-power-spot-forecaster
   ```
2. **Sync the environment and install dependencies:**
   ```bash
   uv sync
   ```
3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

## Project Roadmap

This project is led by a team of five data scientists from Télécom SudParis (IP Paris), with a structured timeline for development:

- Session 1: Topic selection, scientific literature review, and architectural proposal.

- Session 2: Preliminary data ingestion, API integration, data cleaning, and feature engineering.

- Session 3: Final comparative analysis. Demonstration of model architectures minimizing Mean Absolute Error (MAE) and handling price heteroscedasticity.

## Key Literature

- Kılıç et al., 2024: RNN/LSTM applications for intraday price forecasting in DK1.

- Agbulut et al., 2023: Tree-based ensemble modeling (XGBoost) for volatile market data.

- Sideratos et al., 2023: Probabilistic imbalance forecasting using quantile methods.
