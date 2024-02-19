"""
Module for outbreak simulations.
"""

import numpy as np

import endoutbreak


class OutbreakSimulation:
    """
    Class for simulating an outbreak using a branching process model.

    Parameters
    ----------
    offspring_distrib : endoutbreak.OffspringDistribution
        Distribution of offspring per case.
    serial_interval_distrib : endoutbreak.SerialIntervalDistribution
        Distribution of serial intervals.
    t_stop : int, optional
        Cutoff time at which to stop simulation if the outbreak has not already ended.
        Defaults to 1000.
    rng : numpy.random.Generator, optional
        Random number generator. The default is None, in which case a new generator is
        created.
    """

    def __init__(
        self,
        offspring_distrib,
        serial_interval_distrib,
        t_stop=1000,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        self._offspring_distrib = offspring_distrib
        self._serial_interval_distrib = serial_interval_distrib
        self._t_stop = t_stop
        self._sim_complete = False
        self._outbreak_truncated = False
        self._day_reported_curr = None
        self._day_reported_prev = None
        self._day_reported = None
        self._day_reported_prev = None
        self._infectee_index_prev = None
        self._offspring_prev_all = None
        self._before_t_stop_mask = None
        self._case_counts = None
        self._t_matches_to = None

    def run_sim(self):
        """
        Method to run the simulation.

        Parameters
        ----------
        None

        Returns
        -------
        transmission_data : endoutbreak.TransmissionDataset
            Simulated transmission data.
        outbreak_over : bool
            Whether or not the outbreak is over by the cutoff time, t_stop.
        """
        # Initialise simulated outbreak
        self._initialise_outbreak()
        # Run transmission generations until no more cases occur or the cutoff time is
        # reached
        while not self._sim_complete:
            # Run next transmission generation
            self._sim_next_gen()
        # Return simulation output
        return self._get_output()

    def _initialise_outbreak(self):
        # Initialises simulated transmission data
        self._day_reported_curr = np.array([0], dtype=int)
        self._day_reported = self._day_reported_curr

    def _sim_next_gen(self):
        # Simulates a single transmission generation
        # Update variables tracking previous generation
        day_reported_prev = self._day_reported_curr
        self._day_reported_prev = day_reported_prev
        self._infectee_index_prev = np.arange(
            len(self._day_reported) - len(day_reported_prev),
            len(self._day_reported),
        )
        # Simulate offspring from previous generation
        offspring_prev_all = self._offspring_distrib.rvs(
            size=len(day_reported_prev), random_state=self._rng
        )
        self._offspring_prev_all = offspring_prev_all
        # Simulate reporting dates of current generation
        day_reported_curr_all = np.repeat(
            day_reported_prev, offspring_prev_all
        ) + self._serial_interval_distrib.rvs(
            size=np.sum(offspring_prev_all), random_state=self._rng
        )
        # Only track cases that occur up to time t_stop
        t_stop = self._t_stop
        before_t_stop_mask = day_reported_curr_all <= t_stop
        self._before_t_stop_mask = before_t_stop_mask
        if not np.all(before_t_stop_mask):
            # Record that the simulation has been truncated before the outbreak's end
            self._outbreak_truncated = True
        if not np.any(before_t_stop_mask):
            # Simulation completed as no new cases up to cutoff time
            self._sim_complete = True
            self._day_reported_curr = np.array([], dtype=int)
        else:
            self._day_reported_curr = day_reported_curr_all[before_t_stop_mask]
            self._day_reported = np.concatenate(
                (self._day_reported, self._day_reported_curr)
            )
            if self._outbreak_truncated and np.all(self._day_reported_curr == t_stop):
                # No more cases can occur up to and including the cutoff time, and we
                # have already recorded that the outbreak has been truncated, so no need
                # to simulate further transmission generations
                self._sim_complete = True

    def _get_output(self):
        # Returns simulated transmission data and whether or not the outbreak is over by
        # the cutoff time
        transmission_data = self._get_ordered_transmission_data()
        outbreak_over = not self._outbreak_truncated
        return transmission_data, outbreak_over

    def _get_ordered_transmission_data(self):
        # Returns a transmission dataset structure with transmission data ordered by
        # reporting date
        day_reported = np.sort(self._day_reported)
        imported = np.full(len(day_reported), False)
        imported[0] = True
        return endoutbreak.TransmissionDataset(day_reported, imported)


class OutbreakSimulationTraced(OutbreakSimulation):
    """
    Extended class for simulating an outbreak with the transmission tree recorded.
    See OutbreakSimulation for parameter descriptions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._infector_index = None

    def _initialise_outbreak(self):
        # Extends parent class initialisation to record that the initial case was
        # imported
        super()._initialise_outbreak()
        self._infector_index = np.array([-1], dtype=int)

    def _sim_next_gen(self):
        # Extends parent class method to record infector indices of individuals
        # infected in the current generation
        super()._sim_next_gen()
        infectee_index_prev = self._infectee_index_prev
        offspring_prev_all = self._offspring_prev_all
        before_t_stop_mask = self._before_t_stop_mask
        if any(before_t_stop_mask):
            # Record infector indices of individuals in current generation infected
            # before the cutoff time
            infector_index_curr_all = np.repeat(infectee_index_prev, offspring_prev_all)
            infector_index_curr = infector_index_curr_all[before_t_stop_mask]
            self._infector_index = np.concatenate(
                (self._infector_index, infector_index_curr)
            )

    def _get_ordered_transmission_data(self):
        # Overrides parent class method to return a transmission dataset structure
        # including infector indices.
        day_reported = self._day_reported
        infectee_index = np.arange(len(day_reported))
        infector_index = self._infector_index
        # Reorder by reporting date
        ind = np.argsort(day_reported)
        day_reported = day_reported[ind]
        infectee_index = infectee_index[ind]
        infector_index = infector_index[ind]
        # Relabel so infectee_index in increasing order
        ind1 = np.argsort(infectee_index)
        infectee_index = ind1[infectee_index]
        infector_index[1:] = ind1[infector_index[1:]]
        # Return transmission dataset structure
        return endoutbreak.TransmissionDatasetTraced(day_reported, infector_index)


class OutbreakSimulationMatched(OutbreakSimulation):
    """
    Extended class for simulating an outbreak that looks for a simulated outbreak that
    matches specified case counts up to at least a specified time point. If a match
    found, it is determined whether or not the simulated outbreak is over at matched
    by matched time points.
    """

    def __init__(
        self,
        offspring_distrib,
        serial_interval_distrib,
        case_counts_match,
        t_match_min,
        rng=None,
    ):
        super().__init__(offspring_distrib, serial_interval_distrib, rng=rng)
        self._case_counts_match = case_counts_match
        self._t_stop = len(case_counts_match) - 1  # Stop simulation at end of matched
        # data
        self._t_match_min = t_match_min
        self._case_counts = None

    def _initialise_outbreak(self):
        # Extends parent class initialisation to calculate case counts and check
        # consistency with matched values
        super()._initialise_outbreak()
        self._case_counts = np.zeros(self._t_stop + 1, dtype=int)
        self._case_counts[0] = 1
        self._check_matching()

    def _sim_next_gen(self):
        # Extends parent class method to calculate case counts and check consistency
        # with matched case values
        super()._sim_next_gen()
        before_t_stop_mask = self._before_t_stop_mask
        if any(before_t_stop_mask):
            # Count daily new cases
            (t_vals, count_vals) = np.unique(
                self._day_reported_curr, return_counts=True
            )
            self._case_counts[t_vals] += count_vals
        self._check_matching()

    def _check_matching(self):
        # Checks consistency with matched case counts

        # Time up to which case data is entirely determined (i.e. up to which no
        # further cases can occur in future transmission generations)
        if len(self._day_reported_curr) > 0:
            t_determined_to = np.min(self._day_reported_curr)
        else:
            t_determined_to = self._t_stop
        # Difference between simulated case counts and matched case counts
        case_count_diffs = self._case_counts - self._case_counts_match
        # Days on which case counts differ from matched case counts
        count_diff_nonzero = case_count_diffs != 0
        # Conditional next steps
        if np.any(count_diff_nonzero[: (t_determined_to + 1)]):
            # Simulated case counts differ from matched case counts at some time point
            # up to which the simulated case data is entirely determined (so matching
            # will never be restored at that time point). Stop the simulation and record
            # the time up to which datasets match.
            self._sim_complete = True
            self._t_matches_to = np.argmax(count_diff_nonzero) - 1
            if len(self._day_reported_curr) > 0:
                # Outbreak has been truncated at the time matching is lost
                self._outbreak_truncated = True
        elif self._sim_complete:
            # Simulated case counts match the matched case counts entirely (since the
            # simulated data is entirely determined up to the maximum matched time
            # point). Record the time up to which datasets match.
            assert np.all(case_count_diffs == 0), (
                "This branch should not be reached if an exact match has not been"
                " found. This is a bug."
            )
            self._t_matches_to = self._t_stop
        else:
            # Simulated case counts match the matched case counts up to the time point
            # up to which the simulated case data is entirely determined.
            # Determine the times at which the simulated case counts exceed the matched
            # case counts.
            count_diff_positive = case_count_diffs > 0
            if np.any(count_diff_positive):
                # Simulated case counts exceed matched case counts at one or more time
                # points (so regardless of future transmission generations, the
                # simulated case counts cannot match the matched case counts beyond one
                # day before the first of these time points). Therefore, no need to
                # simulate beyond one day before the first of these time points.
                t_stop_new = np.argmax(count_diff_positive) - 1
                assert t_determined_to <= t_stop_new <= self._t_stop, (
                    "t_stop_new should be between t_determined_to and t_stop. This is a"
                    " bug."
                )
                # Record that the outbreak has been truncated.
                self._outbreak_truncated = True
                if t_stop_new < self._t_match_min:
                    # We will never find a match up to the minimum time point, so stop
                    # the simulation.
                    self._sim_complete = True
                    self._t_matches_to = t_determined_to  # may actually match beyond
                    # t_determined_to, but this isn't important here
                else:
                    # We may find a match up to the minimum time point, but not beyond
                    # t_stop_new. Update the simulation to only keep looking for a
                    # match up to t_stop_new.
                    self._t_stop = t_stop_new
                    self._case_counts_match = self._case_counts_match[
                        : (t_stop_new + 1)
                    ]
                    self._day_reported_curr = self._day_reported_curr[
                        self._day_reported_curr <= t_stop_new
                    ]
                    self._case_counts = self._case_counts[: (t_stop_new + 1)]
                    assert len(self._day_reported_curr) > 0, (
                        "This branch should only be reached if there are cases in the"
                        " current generation up to t_stop_new. This is a bug."
                    )
                    if np.all(self._day_reported_curr == t_stop_new):
                        # If all retained cases in the current generation occur at
                        # t_stop_new, then no more cases can occur up to t_stop_new, so
                        # the simulated data match the matched data exactly up to
                        # t_stop_new.
                        assert all(self._case_counts == self._case_counts_match), (
                            "This branch should only be reached if the simulated case"
                            " counts match the matched case counts exactly up to"
                            " t_stop_new (since the simulated case data are entirely"
                            " determined up to this time). This is a bug."
                        )
                        self._sim_complete = True
                        self._t_matches_to = t_stop_new
            else:
                # Nothing to do, as no new information about matching has been gained.
                pass

    def _get_output(self):
        # Overrides parent class to return whether a match has been found, and if so,
        # the times up to which the simulated and matched case counts match, and whether
        # or not the outbreak is over at the matched time points.
        if self._t_matches_to >= self._t_match_min:
            match_found = True
            t_match_vec = np.arange(self._t_matches_to + 1)
            outbreak_over_vec = np.full(self._t_matches_to + 1, False)
            if not self._outbreak_truncated:
                # Outbreak has not been truncated, so it is over from the day of the
                # last case.
                t_last_case = np.nonzero(self._case_counts)[0][-1]
                outbreak_over_vec[t_last_case:] = True
            else:
                # Outbreak has been truncated, so has not finished by any of the matched
                # time points.
                pass
        else:
            match_found = False
            t_match_vec = None
            outbreak_over_vec = None
        return match_found, t_match_vec, outbreak_over_vec
