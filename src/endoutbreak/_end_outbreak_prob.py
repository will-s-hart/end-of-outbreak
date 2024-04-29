"""
Module for running end-of-outbreak probability calculations.
"""

import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import endoutbreak
import endoutbreak.mcmc
import endoutbreak.sim


def calc_end_outbreak_prob(outbreak_descr, t, method, options):
    """
    Function for running end-of-outbreak probability calculations. Creates an
    EndOutbreakProbCalculator object of the appropriate type and calls its calculate
    method. See endoutbreak.OutbreakDescription.calc_end_outbreak_prob for details.

    Parameters
    ----------
    outbreak_descr : endoutbreak.OutbreakDescription
    t : int or numpy array
    method : str
    options : dict

    Returns
    -------
    end_outbreak_prob : float or numpy array
    output : dict
    """
    if method == "traced":
        end_outbreak_calculator = EndOutbreakProbCalculatorTraced(
            outbreak_descr, t, options
        )
    elif method == "nishiura":
        end_outbreak_calculator = EndOutbreakProbCalculatorNishiura(
            outbreak_descr, t, options
        )
    elif method == "mcmc":
        end_outbreak_calculator = EndOutbreakProbCalculatorMCMC(
            outbreak_descr, t, options
        )
    elif method == "enumerate":
        end_outbreak_calculator = EndOutbreakProbCalculatorEnumerate(
            outbreak_descr, t, options
        )
    elif method == "sim":
        end_outbreak_calculator = EndOutbreakProbCalculatorSim(
            outbreak_descr, t, options
        )
    else:
        raise ValueError("Invalid method")
    end_outbreak_calculator.calculate()
    return end_outbreak_calculator.end_outbreak_prob, end_outbreak_calculator.output


class EndOutbreakProbCalculator:
    """
    General class for controlling end-of-outbreak probability calculations. Intended to
    be subclassed for a specific calculation method.

    Parameters
    ----------
    outbreak_descr : OutbreakDescription
        Outbreak description.
    t : int or numpy array
        Time point(s) to calculate probability for.
    options : dict
        Dictionary of options supplied to the calculation method. See
        OutbreakDescription.calc_end_outbreak_prob for details.

    Attributes
    ----------
    end_outbreak_prob : float or numpy array
        Probability that the outbreak has ended by time t. Has value None if the
        "calculate" method has not been run.
    output : dict
        Dictionary of additional output from the calculation. Is empty if the
        "calculate" method has not been run.
    """

    def __init__(self, outbreak_descr, t, options):
        self.end_outbreak_prob = None
        self.output = {}
        self._outbreak_descr = outbreak_descr
        self._t = t
        if np.isscalar(t):
            self._vector_t = False
        elif isinstance(t, np.ndarray):
            self._vector_t = True
        else:
            raise ValueError("Invalid t")
        self._options = options
        self._start_time = None

    def calculate(self):
        """
        Main method for running end-of-outbreak probability calculation. Calls either
        _calculate_scalar or _calculate_vector depending on whether t is scalar or
        vector.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._start_time = time.time()
        if self._vector_t:
            self._calculate_vector()
        else:
            self._calculate_scalar()
        time_taken = time.time() - self._start_time
        self.output["time_taken"] = time_taken
        if self._options["print_progress"]:
            print("Calculation completed. Time elapsed =", time_taken, "seconds")

    def _calculate_scalar(self):
        # Placeholder method for calculating the end-of-outbreak probability for a
        # single t. Should set self.end_outbreak_prob and optionally self.output.
        # Should deal with the case where t is less than the maximum time in the data.
        # Should be defined by subclass.
        raise NotImplementedError

    def _calculate_vector(self):
        # Method for calculating the end-of-outbreak probability for a vector of t
        # values using joblib.Parallel. Creates a pool of one or more workers (depending
        # on the supplied options) within a context manager, then calls _run_pool_calc
        # to run the main calculation using the pool.
        options = self._options
        if options["parallel"]:
            n_jobs = options["n_jobs"]
            if options["print_progress"]:
                print("Computing in parallel using", n_jobs, "processes")
        else:
            n_jobs = 1
        seed_sequence = np.random.SeedSequence(options["rng_seed"])
        with Parallel(n_jobs, verbose=0, batch_size=32) as par:
            self._run_pool_calc(par, seed_sequence)

    def _run_pool_calc(self, par, seed_sequence):
        # Called by _calculate_vector to run the main end-of-outbreak probability
        # calculation using joblib.Parallel. Sets self.end_outbreak_prob and
        # self.output.
        options = self._options
        outbreak_descr = self._outbreak_descr

        def _inner_fun(t_curr, rng_seed_curr):
            # Inner function called by the parallel pool to calculate the
            # end-of-outbreak probability for a single t value by creating a new
            # EndOutbreakProbCalculator object and calling its calculate method.
            options_curr = dict(options, rng_seed=rng_seed_curr, print_progress=False)
            calculator = type(self)(outbreak_descr, t_curr, options_curr)
            calculator.calculate()
            return calculator.end_outbreak_prob, calculator.output

        t = self._t
        # Spawn a random number seed for each t value
        child_seeds = seed_sequence.spawn(len(t))
        # Run the calculation for each t value using the parallel pool
        par_calc_results = par(
            delayed(_inner_fun)(t[i], child_seeds[i]) for i in range(len(t))
        )
        # Extract the end-of-outbreak probabilities and output from the results
        self.end_outbreak_prob = np.array(
            [par_calc_results[i][0] for i in range(len(t))]
        )
        output_keys = list(par_calc_results[0][1].keys())
        output_vals = [
            [par_calc_results[j][1][output_keys[i]] for j in range(len(t))]
            for i in range(len(output_keys))
        ]
        output = dict(zip(output_keys, output_vals))
        self.output = output


class EndOutbreakProbCalculatorTraced(EndOutbreakProbCalculator):
    """
    Class for end-of-outbreak probability calculations using the traced method, which
    gives the exact end-of-outbreak probability when the transmission tree is known
    (i.e. when the infectors of all cases are known).

    Inherits from EndOutbreakProbCalculator.
    """

    def _calculate_scalar(self):
        # Calls OutbreakDescription.calc_end_outbreak_prob_traced to calculate the
        # end-of-outbreak probability for a single t value.
        t = self._t
        end_outbreak_prob = self._outbreak_descr.calc_end_outbreak_prob_traced(t)
        self.end_outbreak_prob = end_outbreak_prob


class EndOutbreakProbCalculatorNishiura(EndOutbreakProbCalculator):
    """
    Class for end-of-outbreak probability calculations using the Nishiura method, which
    gives an approximation to the end-of-outbreak probability given line list case
    data.

    Inherits from EndOutbreakProbCalculator.
    """

    def _calculate_scalar(self):
        # Calls OutbreakDescription.calc_end_outbreak_prob_nishiura to calculate the
        # end-of-outbreak probability for a single t value.
        t = self._t
        end_outbreak_prob = self._outbreak_descr.calc_end_outbreak_prob_nishiura(t)
        self.end_outbreak_prob = end_outbreak_prob


class EndOutbreakProbCalculatorMCMC(EndOutbreakProbCalculator):
    """
    Class for end-of-outbreak probability calculations using the MCMC method, which
    involves using  data augmentation MCMC to sample from the posterior distribution of
    possible transmission trees (given line list case data) and averaging over the end-
    of-outbreak probabilities given each sampled tree (calculated using the traced
    method).

    Inherits from EndOutbreakProbCalculator.
    """

    def calculate(self):
        """
        Extends EndOutbreakProbCalculator.calculate to print a method-specific progress
        update. See EndOutbreakProbCalculator.calculate for details.
        """
        options = self._options
        if options["print_progress"]:
            print(
                "Running",
                options["no_iterations"],
                "MCMC iterations per time point",
            )
        super().calculate()

    def _calculate_scalar(self):
        # Uses MCMC to calculate the end-of-outbreak probability for a single t value.
        # An endoutbreak.mcmc.OutbreakDescriptionAugmented object is used to store and
        # update augmented transmission data (which includes imputed infector indices
        # for each case) and for calculating the log-likelihood and end-of-outbreak
        # probability (using the traced method) given the augmented data.
        t = self._t
        options = self._options
        no_iterations = options["no_iterations"]
        burn_in = options["burn_in"]
        thinning = options["thinning"]
        # Set up augmented transmission data
        rng = np.random.default_rng(seed=options["rng_seed"])
        outbreak_descr_augmented = endoutbreak.mcmc.OutbreakDescriptionAugmented(
            self._outbreak_descr, t, rng=rng
        )
        # Set up arrays for storing output
        iterations_kept = np.arange(0, no_iterations, thinning)
        after_burn_in_kept_mask = iterations_kept >= burn_in
        no_iterations_kept = iterations_kept.size
        iteration_kept_curr = -1
        end_outbreak_prob_post = np.zeros(no_iterations_kept)
        infector_index_post = np.zeros(
            (
                no_iterations_kept,
                outbreak_descr_augmented.transmission_data.no_hosts,
            ),
            dtype=int,
        )
        log_lik_post = np.zeros(no_iterations_kept)
        acceptance_vec = np.full(no_iterations_kept, False)
        # Main MCMC loop
        for iteration in range(no_iterations):
            # Update augmented transmission data (Metropolis step)
            outbreak_descr_augmented.update_infectors()
            if iteration in iterations_kept:
                iteration_kept_curr += 1
                # Calculate and store the end-of-outbreak probability and log-likelihood
                end_outbreak_prob_post[iteration_kept_curr] = (
                    outbreak_descr_augmented.end_outbreak_prob
                )
                log_lik_post[iteration_kept_curr] = outbreak_descr_augmented.log_lik
                acceptance_vec[iteration_kept_curr] = (
                    outbreak_descr_augmented.accepted_last_update
                )
                # Store the current infector indices
                infector_index_post[iteration_kept_curr, :] = (
                    outbreak_descr_augmented.transmission_data.infector_index
                )
        # Process output
        end_outbreak_prob = np.mean(end_outbreak_prob_post[after_burn_in_kept_mask])
        output = {}
        detail_df = pd.DataFrame(
            data={
                "MCMC iteration": iterations_kept + 1,
                "After burn-in": after_burn_in_kept_mask,
                "Log-likelihood": log_lik_post,
                "End-of-outbreak probability": end_outbreak_prob_post,
            }
        )
        detail_df.set_index(["MCMC iteration", "After burn-in"], inplace=True)
        detail_df.columns = pd.MultiIndex.from_product([[str(t)], detail_df.columns])
        output["detail"] = detail_df
        output["acceptance_rate"] = np.mean(acceptance_vec[after_burn_in_kept_mask])
        # output['infector_index_post'] = infector_index_post
        # [Could add in functionality to track probabilities of different possible
        # infectors]
        self.end_outbreak_prob = end_outbreak_prob
        self.output = output

    def _calculate_vector(self):
        # Extends EndOutbreakProbCalculator._calculate_vector to provide method-
        # specific output processing.
        # See EndOutbreakProbCalculator._calculate_vector
        super()._calculate_vector()
        self.output["detail"] = pd.concat(self.output["detail"], axis=1)
        self.output["acceptance_rate"] = pd.DataFrame(
            data={
                "Day of outbreak": self._t,
                "Acceptance rate": self.output["acceptance_rate"],
            }
        ).set_index("Day of outbreak")


class EndOutbreakProbCalculatorEnumerate(EndOutbreakProbCalculator):
    """
    Class for end-of-outbreak probability calculations using the enumerate method,
    which involves enumerating all possible transmission trees (given line list case
    data) and calculating a likelihood-weighted sum of the end-of-outbreak probabilities
    (calculated using the traced method) given each tree.

    Inherits from EndOutbreakProbCalculator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not np.any(
            self._outbreak_descr.transmission_data.imported[1:]
        ), "The 'enumerate' method is only implemented with a single imported case"

    def _calculate_scalar(self):
        # Calculates the end-of-outbreak probability for a single t value using the
        # enumerate method.
        t = self._t
        day_reported = self._outbreak_descr.transmission_data.truncate(t).day_reported
        # Create a list of all possible infector indices for each case
        poss_infectors = [np.array([-1])] + [
            np.arange(np.max(np.nonzero(day_reported < day_reported[i])) + 1)
            for i in range(1, len(day_reported))
        ]
        no_poss_combs = np.prod(
            [len(poss_infectors[i]) for i in range(len(poss_infectors))]
        )
        # Create an iterator over all possible combinations of infector indices
        infector_index_possibilities = itertools.product(*poss_infectors)
        # Set up arrays for storing output
        ll_possibilities = np.zeros(no_poss_combs)
        end_outbreak_prob_possibilities = np.zeros(no_poss_combs)
        # Set up an endoutbreak.mcmc.OutbreakDescriptionAugmented object for calculating
        # the end-of-outbreak probability and likelihood for each possible combination
        # of infector indices
        outbreak_descr_augmented = endoutbreak.mcmc.OutbreakDescriptionAugmented(
            self._outbreak_descr, t
        )
        # Main loop over all possible combinations of infector indices
        for i in range(no_poss_combs):
            infector_index_curr = np.array(next(infector_index_possibilities))
            # Update the augmented transmission data
            outbreak_descr_augmented.transmission_data = (
                endoutbreak.TransmissionDatasetTraced(day_reported, infector_index_curr)
            )
            # Calculate and store the end-of-outbreak probability and likelihood
            end_outbreak_prob_possibilities[i] = (
                outbreak_descr_augmented.end_outbreak_prob
            )
            ll_possibilities[i] = outbreak_descr_augmented.log_lik
        # Calculate the end-of-outbreak probability as a likelihood-weighted sum of the
        # values calculated for each possible combination of infector indices
        rel_prob_possibilities = np.exp(ll_possibilities - np.max(ll_possibilities))
        rel_prob_possibilities = rel_prob_possibilities / np.sum(rel_prob_possibilities)
        end_outbreak_prob = np.sum(
            rel_prob_possibilities * end_outbreak_prob_possibilities
        )
        # Process output
        self.end_outbreak_prob = end_outbreak_prob


class EndOutbreakProbCalculatorSim(EndOutbreakProbCalculator):
    """
    Class for end-of-outbreak probability calculations using the sim method, which
    involves repeatedly simulating the branching process model until a specified number
    of simulations have been found that match the line list case data, and then
    calculating the proportion of these simulations that end by the time, t, at which
    the end-of-outbreak probability is being calculated.

    Note that this method is only implemented for the case where there is a single
    imported case. Additionally, only scalar t values or vectors with strictly
    increasing entries are currently supported.

    Inherits from EndOutbreakProbCalculator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not np.any(
            self._outbreak_descr.transmission_data.imported[1:]
        ), "The 'sim' method is only implemented with a single imported case"
        if self._vector_t:
            assert np.all(
                np.diff(self._t) > 0
            ), "t should be strictly increasing for the 'sim' method"

    def calculate(self):
        # Extends EndOutbreakProbCalculator.calculate to print a method-specific
        # progress update. See EndOutbreakProbCalculator.calculate for details.
        options = self._options
        if options["print_progress"]:
            print(
                "Searching for",
                options["no_matches"],
                "matched simulations per time point",
            )
        return super().calculate()

    def _calculate_scalar(self):
        # Calculates the end-of-outbreak probability for a single t value via a thin
        # wrapper around the _calculate_vector method (the _run_pool_calc method from
        # the base class is overloaded so that _calculate_vector does not itself call
        # _calculate_scalar via calculate, as is the case in the base class).
        self._t = np.array([self._t])
        self._calculate_vector()
        self.end_outbreak_prob = self.end_outbreak_prob[0]

    def _run_pool_calc(self, par, seed_sequence):
        # Overloads EndOutbreakProbCalculator._run_pool_calc. Unlike the base class
        # method, this method does not simply calculate the end-of-outbreak probability
        # for each t value in parallel. Instead, it exploits the fact that matched
        # simulations for different t values can be found together.
        t = self._t
        options = self._options
        no_matches = options["no_matches"]
        offspring_distrib = self._outbreak_descr.offspring_distrib
        serial_interval_distrib = self._outbreak_descr.serial_interval_distrib
        # Obtain the daily case counts from the line list data that will be matched
        case_counts_match = (
            self._outbreak_descr.transmission_data.get_daily_case_counts_to(
                t_max=np.max(t)
            )
        )

        def _inner_fun(t_curr, rng_seed_curr):
            # Inner function called by the parallel pool to repeatedly simulate the
            # branching process model until a simulation has been found that matches the
            # line list data at least up to time t_curr. Simulations are run using the
            # endoutbreak.sim.OutbreakSimulationMatched class.
            # Returns
            # -------
            # t_match_vec_curr: numpy array
            #   Vector of times for which the simulation actually matches the data
            #   (between time 0 and the maximum time up to which the end-of-outbreak
            #   probability is to be calculated).
            # outbreak_over_vec_curr: numpy array
            #   Boolean vector indicating whether the outbreak is over at each time in
            #   t_match_vec_curr.
            # no_sims_to_match: int
            #   Number of simulations run until a match was found.
            rng = np.random.default_rng(seed=rng_seed_curr)
            match_found = False
            no_sims_to_match = 0
            while not match_found:
                (
                    match_found,
                    t_match_vec_curr,
                    outbreak_over_vec_curr,
                ) = endoutbreak.sim.OutbreakSimulationMatched(
                    offspring_distrib,
                    serial_interval_distrib,
                    case_counts_match=case_counts_match,
                    t_match_min=t_curr,
                    rng=rng,
                ).run_sim()
                no_sims_to_match += 1
            return t_match_vec_curr, outbreak_over_vec_curr, no_sims_to_match

        # Set up arrays for storing output
        matches_found_vec = np.zeros(len(t), dtype=int)  # number of matches found
        # for each t value
        outbreak_over_mat = np.full((len(t), no_matches), False)  # whether the outbreak
        # is over for each match found at each t value
        no_total_sims_vec = np.zeros(len(t), dtype=int)
        # Main loop over t values
        for i in range(len(t)):  # pylint: disable=consider-using-enumerate
            # Use the parallel pool to run model simulations until the required number
            # of matches have been found for the current t value (including matches
            # found for previous t values that actually match up to the current t value)
            matches_to_find = no_matches - matches_found_vec[i]
            child_seeds = seed_sequence.spawn(matches_to_find)
            results = par(
                delayed(_inner_fun)(t[i], child_seeds[j])
                for j in range(matches_to_find)
            )
            # Extract the results from par(delayed) and store them in the output arrays
            for j in range(matches_to_find):
                (
                    t_match_vec_curr,
                    outbreak_over_vec_curr,
                    no_sims_to_match,
                ) = results[j]
                matched = (t >= t[i]) & (t <= np.max(t_match_vec_curr))
                outbreak_over_mat[matched, matches_found_vec[matched]] = (
                    outbreak_over_vec_curr[t[matched]]
                )
                matches_found_vec[matched] += 1
                no_total_sims_vec[i:] += no_sims_to_match
            # Print progress update, using matches found for the final t value to
            # estimate overall progress
            if options["print_progress"]:
                print(
                    "Matches obtained for",
                    f"{i+1}/{len(t)}",
                    "time points. Estimated progress =",
                    f"{(matches_found_vec[-1]/no_matches):.1%}",
                    ", time elapsed =",
                    f"{(time.time()-self._start_time):.1f}",
                    "seconds",
                )
        # Calculate vector of end-of-outbreak probabilities and store output
        self.end_outbreak_prob = np.mean(outbreak_over_mat, axis=1)
        self.output["proportion_sims_matched"] = pd.DataFrame(
            data={
                "Day of outbreak": t,
                "Proportion of simulations matched": no_matches / no_total_sims_vec,
            }
        ).set_index("Day of outbreak")
