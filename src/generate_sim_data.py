"""
Script for generating simulated outbreak data. This script will generate a number of
simulated outbreaks, and save the output of four selected simulations".
"""

import os

import numpy as np
from scipy.stats import gamma

import ebola_parameters
import endoutbreak
import endoutbreak.sim

NO_SIMS = 400
SIM1_ITERATION = 43
SIM2_ITERATION = 102
SIM3_ITERATION = 206
SIM4_ITERATION = 301


def _generate_sim_data():
    # Helper function for generating simulated outbreak data.

    # Set up directories
    curr_dir = os.path.dirname(__file__)
    sim1_data_dir = os.path.join(curr_dir, "../results/sim1/")
    sim2_data_dir = os.path.join(curr_dir, "../results/sim2/")
    sim3_data_dir = os.path.join(curr_dir, "../results/sim3/")
    sim4_data_dir = os.path.join(curr_dir, "../results/sim4/")
    schematic_data_dir = os.path.join(curr_dir, "../results/schematic/")
    os.makedirs(sim1_data_dir, exist_ok=True)
    os.makedirs(sim2_data_dir, exist_ok=True)
    os.makedirs(sim3_data_dir, exist_ok=True)
    os.makedirs(sim4_data_dir, exist_ok=True)
    os.makedirs(schematic_data_dir, exist_ok=True)
    # Inputs to branching process model.
    reproduction_number = ebola_parameters.REPRODUCTION_NUMBER
    offspring_dispersion_param = ebola_parameters.OFFSPRING_DISPERSION_PARAM
    offspring_distrib = endoutbreak.OffspringDistribution(
        reproduction_number, offspring_dispersion_param
    )
    si_mean_daily = ebola_parameters.SI_MEAN
    si_sd_daily = ebola_parameters.SI_SD
    serial_interval_distrib_cont_daily = gamma(
        a=(si_mean_daily / si_sd_daily) ** 2, scale=si_sd_daily**2 / si_mean_daily
    )
    serial_interval_distrib_daily = endoutbreak.discretise_serial_interval(
        serial_interval_distrib_cont_daily
    )
    si_mean_weekly = si_mean_daily / 7
    si_sd_weekly = si_sd_daily / 7
    serial_interval_distrib_cont_weekly = gamma(
        a=(si_mean_weekly / si_sd_weekly) ** 2, scale=si_sd_weekly**2 / si_mean_weekly
    )
    serial_interval_distrib_weekly = endoutbreak.discretise_serial_interval(
        serial_interval_distrib_cont_weekly
    )
    # Other options
    day_stop = 420
    rng_seed = 7
    # Set random number generator seeds for each simulation (using different seeds for
    # each simulation ensures that the value of t_stop does not affect the results).
    seed_sequence = np.random.SeedSequence(rng_seed)
    child_seeds = seed_sequence.spawn(NO_SIMS)

    # Run simulations
    for iteration in range(NO_SIMS):
        if iteration < NO_SIMS / 2:
            serial_interval_distrib = serial_interval_distrib_weekly
            t_stop = day_stop // 7
        else:
            serial_interval_distrib = serial_interval_distrib_daily
            t_stop = day_stop
        rng = np.random.default_rng(seed=child_seeds[iteration])
        transmission_data, outbreak_over = endoutbreak.sim.OutbreakSimulationTraced(
            offspring_distrib,
            serial_interval_distrib,
            t_stop=t_stop,
            rng=rng,
        ).run_sim()
        day_reported = transmission_data.day_reported
        if outbreak_over and len(day_reported) > 3:
            print(
                "Iteration: " + str(iteration),
                "day_reported =",
                day_reported,
            )
        # Save output for selected iterations
        if iteration == SIM1_ITERATION:
            transmission_data.to_csv(sim1_data_dir + "formatted_data.csv")
        if iteration == SIM2_ITERATION:
            transmission_data.to_csv(sim2_data_dir + "formatted_data.csv")
        if iteration == SIM3_ITERATION:
            transmission_data.to_csv(sim3_data_dir + "formatted_data.csv")
        if iteration == SIM4_ITERATION:
            transmission_data.to_csv(sim4_data_dir + "formatted_data.csv")


if __name__ == "__main__":
    _generate_sim_data()
