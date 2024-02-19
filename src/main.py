"""
Main script for running analyses and producing figures.

Behaviour is controlled by the following constants, which can be modified below:
OUTBREAKS_DEFAULT: Default outbreaks to run analyses for, if not specified as a command
    line argument. String or list of strings with possible values 'Ebola_Likati',
    'Ebola_Equateur', 'sim1', 'sim2', 'sim3', 'sim4' or 'schematic'.
RUN_TYPE_DEFAULT: Default run type, if not specified as a command line argument. One of
    'cluster' (for running all analyses on a SLURM cluster), 'local' (for running all
    analyses), 'short' (for running a smaller number of MCMC iterations and/or
    simulations), 'test' (for running a very small number of MCMC iterations and/or
    simulations).
PLOT_ONLY_DEFAULT: If True, by default make plots using previously saved results without
    running any analyses, unless the "plot_only" command-line argument is set. Analyses
    must have been run previously and saved in the results directory using the same run
    type as specified here by RUN_TYPE_DEFAULT or as a command line argument.
SHOW_PLOTS: If True, show plots interactively. This will block the script between
    outbreaks until the plot is closed if multiple outbreaks are run (overridden for
    cluster runs).

Different outbreaks and run types from the defaults can be specified by passing command
line arguments (run `python main.py --help` for details).
"""

import argparse

import get_input
import matplotlib.pyplot as plt
import paper_analyses

OUTBREAKS_DEFAULT = [
    "Ebola_Likati",
    "Ebola_Equateur",
    "sim1",
    "sim2",
    "sim3",
    "sim4",
    "schematic",
]
RUN_TYPE_DEFAULT = "local"
PLOT_ONLY_DEFAULT = True
SHOW_PLOTS = True


def run_eo_analyses(outbreak, run_type, plot_only):
    """
    Function for running analyses and producing plots for a single outbreak.

    Parameters
    ----------
    outbreak: str or list of str
        Outbreak(s) to run analyses for. Possible values are: 'Ebola_Likati',
        'Ebola_Equateur', 'sim1', 'sim2', 'sim3', 'sim4' or 'schematic' (if specifying
        multiple outbreaks, the names should be space-separated).
    run_type: str
        Type of run. One of 'cluster', 'local', 'short' or 'test'.
    plot_only: bool
        If True, make plots using previously saved results without running any
        analyses. Analyses must have been run previously and saved in the results
        directory using the same value of run_type.

    Returns
    -------
    None
    """
    assert run_type in [
        "cluster",
        "local",
        "short",
        "test",
    ], (
        f"Invalid run type '{run_type}' specified. Possible values are: 'cluster',"
        " 'local', 'short', or 'test'."
    )
    if isinstance(outbreak, list):
        for outbreak_curr in outbreak:
            run_eo_analyses(outbreak_curr, run_type, plot_only)
    else:
        assert outbreak in [
            "Ebola_Likati",
            "Ebola_Equateur",
            "sim1",
            "sim2",
            "sim3",
            "sim4",
            "schematic",
        ], (
            f"Invalid outbreak '{outbreak}' specified. Possible values are:"
            " 'Ebola_Likati', 'Ebola_Equateur', 'sim1', 'sim2', 'sim3', 'sim4' or"
            " 'schematic', or a list of one or more of these."
        )
        outbreak_descr, options = get_input.get_input(outbreak, run_type=run_type)
        end_outbreak_analyses = paper_analyses.EndOutbreakAnalyses(
            outbreak_descr, options
        )
        if plot_only:
            end_outbreak_analyses.load_results()
        else:
            end_outbreak_analyses.run_analyses()
        end_outbreak_analyses.make_figures()
        if SHOW_PLOTS and run_type != "cluster":
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(prog="main.py", description="Run main analyses.")
    parser.add_argument(
        "--outbreaks",
        nargs="*",
        default=OUTBREAKS_DEFAULT,
        help=(
            "Outbreaks to run analyses for. One or more of"
            " 'Ebola_Likati', 'Ebola_Equateur', 'sim1', 'sim2', 'sim3', 'sim4' or"
            " 'schematic' (space-separated)."
        ),
    )
    parser.add_argument(
        "--run_type",
        default=RUN_TYPE_DEFAULT,
        help="Type of run. One of 'cluster', 'local', 'short' or 'test'.",
    )
    parser.add_argument(
        "-p",
        "--plot_only",
        action="store_true",
        help=(
            "If set, make plots using previously saved results without running any"
            " analyses. Analyses must have been run previously and saved in the results"
            " directory using the same value of run_type (or default RUN_TYPE_DEFAULT"
            " if run_type is not set)."
        ),
    )
    parser.add_argument(
        "-r",
        "--run_analyses",
        action="store_true",
        help=(
            "If set, run analyses and save results. If not set, behaviour depends on"
            " whether --plot_only is set. At most one of --run_analyses and"
            " --plot_only can be set."
        ),
    )
    args = parser.parse_args()
    if args.run_analyses and args.plot_only:
        raise ValueError("At most one of --run_analyses and --plot_only can be set.")
    if not args.run_analyses:
        args.plot_only = PLOT_ONLY_DEFAULT
    # Run analyses
    run_eo_analyses(args.outbreaks, args.run_type, args.plot_only)
