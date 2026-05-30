"""Schema constants and field definitions for the LLM context JSON.

Defines the canonical field names used in LLMContext so all summarizers
stay consistent without circular imports.
"""

from __future__ import annotations

CONTEXT_VERSION = "1.0"

# Severity levels for failure modes
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# Performance interpretation thresholds
SHARPE_EXCELLENT = 1.0
SHARPE_GOOD = 0.5
SHARPE_WEAK = 0.0

DRAWDOWN_SEVERE = -0.40
DRAWDOWN_ELEVATED = -0.20

OOS_HIT_RATE_STRONG = 0.70
OOS_HIT_RATE_WEAK = 0.50

IC_MEANINGFUL = 0.03
IC_STRONG = 0.07

DA_STRONG = 0.55
DA_WEAK = 0.48
