import arviz
import get_input
import pandas as pd
import paper_analyses

# Dictionary of times at which to print effective sample sizes for each outbreak
OUTBREAK_ESS_PRINT_TIMES = {
    "Ebola_Likati": [30, 50, 70, 85, 100],
    "Ebola_Equateur": [25, 130, 150, 170],
    "sim1": [4, 6, 8],
    "sim2": [4, 6, 8],
    "sim3": [30, 70, 90, 120],
    "sim4": [40, 80, 160, 210],
}


def _get_effective_sample_size(outbreak):
    print(f"Outbreak: {outbreak}")
    outbreak_descr, options = get_input.get_input(outbreak, run_type="local")
    end_outbreak_analyses = paper_analyses.EndOutbreakAnalyses(outbreak_descr, options)
    end_outbreak_analyses.load_results()
    mcmc_detail_df = end_outbreak_analyses._output_mcmc_dfs["detail"]
    eop_post_df = mcmc_detail_df.xs("End-of-outbreak-probability", level=1, axis=1).xs(
        True, level="After burn-in"
    )
    ess_df = pd.DataFrame(columns=eop_post_df.columns)
    for t in eop_post_df.columns:
        ess = arviz.ess(eop_post_df[t].to_numpy())
        ess_df[t] = [ess]
    print(
        f"Mean ESS = {ess_df.mean(axis=1).values[0]},"
        + f" min ESS = {ess_df.min(axis=1).values[0]},"
        + f" max ESS = {ess_df.max(axis=1).values[0]} (SS = {len(eop_post_df)})"
    )
    for t in OUTBREAK_ESS_PRINT_TIMES[outbreak]:
        ess = ess_df[str(t)].values[0]
        print(f"Time {t}: ESS = {ess} (SS = {len(eop_post_df)})")


if __name__ == "__main__":
    for outbreak_curr in OUTBREAK_ESS_PRINT_TIMES:
        _get_effective_sample_size(outbreak_curr)
