"""Research visualization layer — read-only, matplotlib-based plotting utilities."""

from src.visualization.backtest_plots import (
    plot_drawdown,
    plot_equity_and_drawdown,
    plot_equity_curve,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)
from src.visualization.comparison_plots import (
    plot_metric_comparison,
    plot_metrics_table,
    plot_strategy_drawdowns,
    plot_strategy_equity_curves,
)
from src.visualization.diagnostics import (
    compute_concentration_metrics,
    compute_turnover,
    rolling_average_correlation,
)
from src.visualization.distribution_plots import (
    plot_monthly_return_heatmap,
    plot_return_distribution,
)
from src.visualization.ml_plots import (
    plot_coefficient_stability,
    plot_information_coefficient,
    plot_prediction_distribution,
    plot_prediction_vs_actual,
    plot_signal_turnover,
    plot_split_metric_stability,
)
from src.visualization.portfolio_plots import (
    plot_asset_contribution,
    plot_asset_correlation,
    plot_rolling_correlation,
    plot_rolling_weights,
    plot_turnover,
    plot_weight_concentration,
    plot_weight_heatmap,
    plot_weight_lines,
    plot_weights,
)
from src.visualization.render_profiles import (
    ARCHETYPE_BAR,
    ARCHETYPE_CONTRIBUTION,
    ARCHETYPE_DEFAULT,
    ARCHETYPE_HEATMAP,
    ARCHETYPE_ROLLING,
    ARCHETYPE_STACKED,
    ARCHETYPE_TIMELINE,
    RENDER_PROFILES,
    apply_render_profile,
    get_active_profile,
    get_archetype_hint,
    get_dpi_save,
    get_figsize_scale,
    get_line_width_scale,
    get_render_profile,
    set_active_profile,
)
from src.visualization.signal_plots import (
    plot_position_state,
    plot_sma_strategy,
)
from src.visualization.styles import apply_research_style, research_style_context
from src.visualization.typography import (
    TypographyScale,
    get_typography,
    heatmap_cell_fontsize,
    scale_dynamic_fontsize,
)
from src.visualization.utils import save_figure, to_datetime_index
from src.visualization.validation_plots import (
    plot_metric_stability,
    plot_split_sharpes,
    plot_train_vs_test,
    plot_walk_forward_equity,
    plot_walk_forward_stitched,
)

__all__ = [
    # backtest
    "plot_equity_curve",
    "plot_drawdown",
    "plot_equity_and_drawdown",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    # signal
    "plot_sma_strategy",
    "plot_position_state",
    # distribution
    "plot_return_distribution",
    "plot_monthly_return_heatmap",
    # style
    "apply_research_style",
    "research_style_context",
    # portfolio
    "plot_weights",
    "plot_weight_lines",
    "plot_rolling_weights",
    "plot_asset_correlation",
    "plot_weight_heatmap",
    "plot_turnover",
    "plot_weight_concentration",
    "plot_asset_contribution",
    "plot_rolling_correlation",
    # diagnostics
    "compute_turnover",
    "compute_concentration_metrics",
    "rolling_average_correlation",
    # comparison
    "plot_strategy_equity_curves",
    "plot_strategy_drawdowns",
    "plot_metric_comparison",
    "plot_metrics_table",
    # validation
    "plot_walk_forward_equity",
    "plot_walk_forward_stitched",
    "plot_split_sharpes",
    "plot_metric_stability",
    "plot_train_vs_test",
    # utils
    "save_figure",
    "to_datetime_index",
    # typography
    "TypographyScale",
    "get_typography",
    "heatmap_cell_fontsize",
    "scale_dynamic_fontsize",
    # render profiles
    "apply_render_profile",
    "get_active_profile",
    "get_archetype_hint",
    "get_dpi_save",
    "get_figsize_scale",
    "get_line_width_scale",
    "get_render_profile",
    "set_active_profile",
    "ARCHETYPE_BAR",
    "ARCHETYPE_CONTRIBUTION",
    "ARCHETYPE_DEFAULT",
    "ARCHETYPE_HEATMAP",
    "ARCHETYPE_ROLLING",
    "ARCHETYPE_STACKED",
    "ARCHETYPE_TIMELINE",
    "RENDER_PROFILES",
    # ml diagnostics
    "plot_prediction_vs_actual",
    "plot_prediction_distribution",
    "plot_information_coefficient",
    "plot_split_metric_stability",
    "plot_coefficient_stability",
    "plot_signal_turnover",
]
