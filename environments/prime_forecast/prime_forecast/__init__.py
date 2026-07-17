"""prime-forecast — agentic forecasting RLVR environment (verifiers).

Multi-turn tool-use agent that researches a resolved Polymarket question
(without being shown the outcome or, by default, the crowd price) and submits
a probability. Reward is positive-shifted Brier against the real outcome.
"""

from prime_forecast.env import ForecastEnv, load_environment

__all__ = ["ForecastEnv", "load_environment"]
