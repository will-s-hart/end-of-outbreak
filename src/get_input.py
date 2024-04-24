"""
Module for obtaining inputs for a given outbreak and run type.
"""

import os

import ebola_parameters
import endoutbreak
import numpy as np
import pandas as pd
from scipy.stats import gamma


def get_input(outbreak, run_type):
    """
    Function for obtaining inputs for a given outbreak and run type.

    Parameters
    ----------
    outbreak: str
        Outbreak to run analyses for. Possible values are: 'Ebola_Likati',
        'Ebola_Equateur', 'Nipah', 'sim1', 'sim2', 'sim3', 'sim4' or 'schematic'. See
        main.py for details.
    run_type: str
        Type of run. One of 'cluster', 'local', 'short' or 'test'. See main.py for
        details.

    Returns
    -------
    outbreak_descr: endoutbreak.OutbreakDescription
        Object containing information about the outbreak.
    options: dict
        Dictionary containing options for analyses to pass to
        paper_analyses.EndOutbreakAnalyses.
    """
    # Directories to save results and figures into
    curr_dir = os.path.dirname(__file__)
    results_dir = os.path.join(curr_dir, "../results", outbreak)
    figure_dir = os.path.join(curr_dir, "../figures", outbreak)

    # Data
    transmission_data = _get_transmission_data(outbreak, results_dir)

    # Distributions
    offspring_distrib = _get_offspring_distrib()
    serial_interval_distrib = _get_serial_interval_distrib(outbreak, si_dir=None)

    # Times to estimate end-of-outbreak probabilities at
    t_vec = np.arange(
        np.max(transmission_data.day_reported)
        + int(serial_interval_distrib.ppf(0.9999))
    )

    # Options for which analyses to run
    if outbreak in ["sim1", "sim2"]:
        run_sim = True
        run_enumerate = True
    elif outbreak == "schematic":
        run_sim = False
        run_enumerate = True
    else:
        run_sim = False
        run_enumerate = False

    # Options for MCMC and sim methods depending on run type
    options_mcmc, options_sim = _get_options_mcmc_sim(run_type)

    # Options for plotting
    options_plot = _get_options_plot(outbreak)

    # Format inputs
    outbreak_descr = endoutbreak.OutbreakDescription(
        transmission_data, offspring_distrib, serial_interval_distrib
    )
    options = {
        "results_dir": results_dir,
        "figure_dir": figure_dir,
        "t_vec": t_vec,
        "options_mcmc": options_mcmc,
        "options_sim": options_sim,
        "options_plot": options_plot,
        "run_sim": run_sim,
        "run_enumerate": run_enumerate,
    }
    return outbreak_descr, options


def _get_transmission_data(outbreak, formatted_data_dir):
    if outbreak == "schematic":
        transmission_data = endoutbreak.TransmissionDataset(
            day_reported=np.array([0, 20, 20, 29]),
            imported=np.array([True, False, False, False]),
        )
    else:
        formatted_data_path = os.path.join(formatted_data_dir, "formatted_data.csv")
        transmission_data = endoutbreak.load_outbreak_dataset(formatted_data_path)
    return transmission_data


def _get_offspring_distrib():
    reproduction_number = ebola_parameters.REPRODUCTION_NUMBER
    offspring_dispersion_param = ebola_parameters.OFFSPRING_DISPERSION_PARAM
    offspring_distrib = endoutbreak.OffspringDistribution(
        reproduction_number, offspring_dispersion_param
    )
    return offspring_distrib


def _get_serial_interval_distrib(outbreak, si_dir=None):
    if si_dir is not None:
        serial_interval_distrib = endoutbreak.load_discr_serial_interval_distrib(
            os.path.join(si_dir, "serial_interval.csv")
        )
    else:
        si_mean = ebola_parameters.SI_MEAN
        si_sd = ebola_parameters.SI_SD
        if outbreak in ["sim1", "sim2"]:
            # Simulated datasets using weekly data
            si_mean = si_mean / 7
            si_sd = si_sd / 7
        serial_interval_distrib_cont = gamma(
            a=(si_mean / si_sd) ** 2, scale=si_sd**2 / si_mean
        )
        serial_interval_distrib = endoutbreak.discretise_serial_interval(
            serial_interval_distrib_cont
        )
    return serial_interval_distrib


def _get_options_mcmc_sim(run_type):
    if run_type == "cluster":
        n_jobs = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    else:
        n_jobs = os.cpu_count() // 2
    if run_type in ["cluster", "local"]:
        options_mcmc = {
            "no_iterations": 10000000,
            "burn_in": 2000000,
            "thinning": 1000,
            "parallel": True,
            "n_jobs": n_jobs,
            "print_progress": True,
            "rng_seed": 1,
        }
        options_sim = {
            "no_matches": 10000,
            "parallel": True,
            "n_jobs": n_jobs,
            "print_progress": True,
            "rng_seed": 1,
        }
    elif run_type == "short":
        options_mcmc = {
            "no_iterations": 100000,
            "burn_in": 20000,
            "thinning": 100,
            "parallel": True,
            "n_jobs": n_jobs,
            "print_progress": True,
            "rng_seed": 1,
        }
        options_sim = {
            "no_matches": 1000,
            "parallel": True,
            "n_jobs": n_jobs,
            "print_progress": True,
            "rng_seed": 1,
        }
    elif run_type == "test":
        parallel_test = False
        options_mcmc = {
            "no_iterations": 100,
            "burn_in": 2,
            "thinning": 1,
            "parallel": parallel_test,
            "n_jobs": n_jobs,
            "print_progress": True,
            "rng_seed": 1,
        }
        options_sim = {
            "no_matches": 10,
            "parallel": parallel_test,
            "n_jobs": n_jobs,
            "print_progress": True,
            "rng_seed": 1,
        }
    return options_mcmc, options_sim


def _get_options_plot(outbreak):
    options_plot = {
        "options_offspring": {"offspring_max": 10},
        "options_serial_interval": {},
        "options_data_eop": {},
        "options_declaration": {},
        "options_mcmc_log_lik_trace": {},
        "options_mcmc_eop_trace": {},
        "options_mcmc_eop_hist": {},
    }
    if outbreak == "Ebola_Likati":
        options_plot["options_data_eop"]["t_max"] = 100
        options_plot["options_data_eop"]["xlim_left"] = -1
        options_plot["options_data_eop"]["eop_methods"] = ["MCMC", "Traced"]
        options_plot["options_data_eop"]["legend_kwargs"] = {"loc": (0.55, 0.02)}
        declaration_day = (
            pd.to_datetime("2017-07-02") - pd.to_datetime("2017-03-27")
        ).days
        options_plot["options_data_eop"]["declaration_day"] = declaration_day
        for key in ["eop_methods", "declaration_day"]:
            options_plot["options_declaration"][key] = options_plot["options_data_eop"][
                key
            ]
        options_plot["options_declaration"]["ylim"] = (20, 70)
        options_plot["options_mcmc_log_lik_trace"]["t_vec_plot"] = [30, 50, 70]
        options_plot["options_mcmc_log_lik_trace"]["y_lim_vals"] = [
            [-15, -10],
            [-40, -20],
            [-40, -20],
        ]
        options_plot["options_mcmc_eop_trace"]["t_vec_plot"] = [30, 50, 70]
        options_plot["options_mcmc_eop_hist"]["t_vec_plot"] = [30, 50, 70]
    if outbreak == "Ebola_Equateur":
        options_plot["options_data_eop"]["t_max"] = 200
        options_plot["options_data_eop"]["xlim_left"] = -1
        options_plot["options_data_eop"]["xticks"] = [0, 40, 80, 120, 160, 200]
        options_plot["options_data_eop"]["eop_methods"] = ["MCMC"]
        options_plot["options_data_eop"]["legend_kwargs"] = {"loc": "upper center"}
        declaration_day = (
            pd.to_datetime("2020-11-18") - pd.to_datetime("2020-05-09")
        ).days
        options_plot["options_data_eop"]["declaration_day"] = declaration_day
        for key in ["eop_methods", "declaration_day"]:
            options_plot["options_declaration"][key] = options_plot["options_data_eop"][
                key
            ]
        options_plot["options_declaration"]["ylim"] = (20, 70)
        options_plot["options_declaration"]["legend_kwargs"] = {"loc": "lower right"}
        options_plot["options_mcmc_log_lik_trace"]["t_vec_plot"] = [25, 130, 150]
        options_plot["options_mcmc_log_lik_trace"]["y_lim_vals"] = [
            [-50, -25],
            [-550, -350],
            [-550, -350],
        ]
        options_plot["options_mcmc_eop_trace"]["t_vec_plot"] = [25, 130, 150]
        options_plot["options_mcmc_eop_hist"]["t_vec_plot"] = [25, 130, 150]
    if outbreak == "sim1":
        options_plot["options_data_eop"]["t_max"] = 12
        options_plot["options_data_eop"]["xlabel"] = "Week of outbreak"
        options_plot["options_serial_interval"]["xlabel"] = (
            "Weekly serial interval (weeks)"
        )
    if outbreak == "sim2":
        options_plot["options_data_eop"]["t_max"] = 12
        options_plot["options_data_eop"]["xlabel"] = "Week of outbreak"
        options_plot["options_data_eop"]["show_legend"] = False
        options_plot["options_serial_interval"]["xlabel"] = (
            "Weekly serial interval (weeks)"
        )
    if outbreak == "sim3":
        options_plot["options_data_eop"]["t_max"] = 120
        options_plot["options_data_eop"]["xlim_left"] = -1
        options_plot["options_data_eop"]["legend_kwargs"] = {
            "borderpad": 0.4,
            "handlelength": 1.4,
            "handletextpad": 0.4,
            "borderaxespad": 0.4,
        }
    if outbreak == "sim4":
        options_plot["options_data_eop"]["t_max"] = 210
        options_plot["options_data_eop"]["xlim_left"] = -1
        options_plot["options_data_eop"]["show_legend"] = False
    if outbreak == "schematic":
        options_plot["options_data_eop"]["eop_methods"] = ["MCMC"]
        options_plot["options_data_eop"]["include_quantile"] = False
        options_plot["options_data_eop"]["declaration_day"] = 40
        options_plot["options_data_eop"]["t_max"] = 70
        options_plot["options_data_eop"]["show_legend"] = False
    return options_plot
