"""
Risk analytics: alpha/beta decomposition, factor exposure, drawdown analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from statarb.utils import get_logger

logger = get_logger(__name__)


class RiskAnalytics:
    """
    Risk decomposition and factor analysis for strategy returns.

    Methods:
    - Alpha / beta vs a single benchmark (BTC or equal-weighted market)
    - Multi-factor regression (CAPM-style, Fama-French inspired)
    - Rolling alpha / beta estimation
    - Tail risk analysis
    - Correlation regime analysis
    """

    def __init__(self, strategy_returns: pd.Series, periods_per_year: int = 252):
        """
        Args:
            strategy_returns: daily strategy return series
            periods_per_year: annualisation factor
        """
        self.returns = strategy_returns.dropna()
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Alpha / Beta
    # ------------------------------------------------------------------

    def alpha_beta(
        self,
        benchmark_returns: pd.Series,
        annualise: bool = True,
    ) -> Tuple[float, float, float, float]:
        """
        OLS regression of strategy returns on benchmark returns.

        Returns:
            (alpha, beta, r_squared, p_value_alpha)
            alpha is annualised if annualise=True.
        """
        bench = benchmark_returns.reindex(self.returns.index).dropna()
        strat = self.returns.reindex(bench.index).dropna()
        common = strat.index.intersection(bench.index)
        y = strat.loc[common].values
        x = bench.loc[common].values

        if len(y) < 10:
            return (0.0, 0.0, 0.0, 1.0)

        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        alpha = intercept
        if annualise:
            alpha = alpha * self.periods_per_year
        return float(alpha), float(slope), float(r_value ** 2), float(p_value)

    def rolling_alpha_beta(
        self,
        benchmark_returns: pd.Series,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Rolling alpha and beta over a sliding window.

        Args:
            benchmark_returns: benchmark return series
            window: rolling window in periods

        Returns:
            DataFrame with columns ['alpha', 'beta', 'r_squared'].
        """
        bench = benchmark_returns.reindex(self.returns.index).fillna(0)
        result = pd.DataFrame(
            np.nan,
            index=self.returns.index,
            columns=["alpha", "beta", "r_squared"],
        )

        for i in range(window, len(self.returns) + 1):
            y = self.returns.iloc[i - window: i].values
            x = bench.iloc[i - window: i].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < window // 2:
                continue
            slope, intercept, r_value, _, _ = stats.linregress(x[mask], y[mask])
            result.iloc[i - 1] = [
                intercept * self.periods_per_year,
                slope,
                r_value ** 2,
            ]

        return result

    # ------------------------------------------------------------------
    # Multi-factor regression
    # ------------------------------------------------------------------

    def factor_regression(
        self,
        factors: pd.DataFrame,
        add_constant: bool = True,
    ) -> Dict[str, float]:
        """
        Regress strategy returns on multiple factor return series.

        Args:
            factors: DataFrame where each column is a factor return series
            add_constant: include intercept (alpha)

        Returns:
            Dict of factor_name → coefficient, plus 'alpha', 'r_squared'.
        """
        import statsmodels.api as sm

        common_idx = self.returns.index.intersection(factors.index)
        y = self.returns.loc[common_idx]
        X = factors.loc[common_idx]

        if add_constant:
            X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X, missing="drop").fit()
        except Exception as exc:
            logger.error("Factor regression failed: %s", exc)
            return {}

        coef_dict = dict(model.params)
        if "const" in coef_dict:
            alpha_coef = coef_dict.pop("const")
            coef_dict["alpha"] = alpha_coef * self.periods_per_year
        coef_dict["r_squared"] = model.rsquared
        coef_dict["adj_r_squared"] = model.rsquared_adj
        return coef_dict

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    def tail_risk_summary(self) -> Dict[str, float]:
        """
        Summary of tail risk characteristics.

        Returns dict with:
        - max_daily_loss: worst single-period return
        - worst_5_avg: average of worst 5 periods
        - tail_ratio: 95th pct / 5th pct of abs(returns) (>1 = more upside tail)
        - downside_capture (vs benchmark if provided separately)
        """
        sorted_ret = self.returns.sort_values()
        worst_5 = sorted_ret.head(5).mean()
        p95 = np.percentile(self.returns.abs(), 95)
        p5 = np.percentile(self.returns.abs(), 5)

        return {
            "max_daily_loss": float(sorted_ret.iloc[0]),
            "max_daily_gain": float(sorted_ret.iloc[-1]),
            "worst_5_avg": float(worst_5),
            "best_5_avg": float(sorted_ret.tail(5).mean()),
            "tail_ratio": float(p95 / p5) if p5 > 0 else float("inf"),
        }

    # ------------------------------------------------------------------
    # Regime analysis
    # ------------------------------------------------------------------

    def conditional_performance(
        self,
        condition: pd.Series,
        label_high: str = "high",
        label_low: str = "low",
    ) -> Dict[str, Dict[str, float]]:
        """
        Performance breakdown across two regimes defined by a boolean / binary condition.

        Example use: condition = (btc_vol > median_vol) to split into
        high-vol and low-vol regimes.

        Args:
            condition: boolean/binary Series aligned to returns
            label_high: label for condition == True
            label_low: label for condition == False

        Returns:
            Dict of regime → {mean_return, annualised_return, volatility, sharpe}
        """
        cond = condition.reindex(self.returns.index).fillna(False).astype(bool)

        def _stats(rets: pd.Series) -> Dict[str, float]:
            if len(rets) < 5:
                return {}
            ann_ret = float(rets.mean() * self.periods_per_year)
            ann_vol = float(rets.std() * np.sqrt(self.periods_per_year))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            return {
                "n_periods": len(rets),
                "annualised_return": ann_ret,
                "annualised_volatility": ann_vol,
                "sharpe_ratio": sharpe,
            }

        return {
            label_high: _stats(self.returns[cond]),
            label_low: _stats(self.returns[~cond]),
        }

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling annualised Sharpe ratio."""
        roll_mean = self.returns.rolling(window=window, min_periods=window // 2).mean()
        roll_std = self.returns.rolling(window=window, min_periods=window // 2).std()
        return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(self.periods_per_year)

    # ------------------------------------------------------------------
    # Correlation and diversification
    # ------------------------------------------------------------------

    def strategy_correlation(
        self, other_returns: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        Pairwise correlation of this strategy with other strategies.

        Args:
            other_returns: dict of strategy_name → returns Series

        Returns:
            Series of correlations indexed by strategy name.
        """
        corrs = {}
        for name, rets in other_returns.items():
            aligned = pd.concat([self.returns, rets], axis=1).dropna()
            if len(aligned) < 10:
                corrs[name] = np.nan
            else:
                corrs[name] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        return pd.Series(corrs)
