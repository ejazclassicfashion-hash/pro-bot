"""
Combined SMC + ICT + VSA Strategy
SMC provides the structure and setup zones; VSA confirms with volume.
Confluence = both must agree for a trade to be taken.
"""

import pandas as pd
from dataclasses import dataclass
from .smc_ict import run_smc_analysis, SMCParams, SMCAnalysis
from .vsa import analyze_vsa, VSAParams, vsa_confirms_smc


@dataclass
class CombinedParams:
    smc: SMCParams = None
    vsa: VSAParams = None
    require_vsa_confirm: bool = False  # VSA as optional filter; turn on after tuning
    vsa_lookback: int = 3             # candles before signal to check VSA
    max_risk_per_trade_pct: float = 1.0  # max 1% account risk per trade

    def __post_init__(self):
        if self.smc is None:
            self.smc = SMCParams()
        if self.vsa is None:
            self.vsa = VSAParams()


def run_combined(
    df: pd.DataFrame,
    params: CombinedParams | None = None,
) -> tuple[SMCAnalysis, pd.DataFrame, list[dict]]:
    """
    Returns:
    - smc: full SMC analysis object (for visualization)
    - df_vsa: dataframe with VSA columns appended
    - filtered_signals: only signals that pass VSA confluence filter
    """
    if params is None:
        params = CombinedParams()

    smc = run_smc_analysis(df, params.smc)
    df_vsa = analyze_vsa(df, params.vsa)

    if not params.require_vsa_confirm:
        return smc, df_vsa, smc.signals

    filtered = []
    for sig in smc.signals:
        if vsa_confirms_smc(df_vsa, sig["idx"], sig["direction"], params.vsa_lookback):
            sig["vsa_confirmed"] = True
            filtered.append(sig)
        else:
            sig["vsa_confirmed"] = False

    return smc, df_vsa, filtered
