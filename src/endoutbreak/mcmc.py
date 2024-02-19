"""
Module providing classes for storing and updating augmented transmission data in the
MCMC method for calculating the end-of-outbreak probability.
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import rv_discrete, uniform

import endoutbreak


class OutbreakDescriptionAugmented:
    """
    Class for storing and updating augmented transmission data alongside the serial
    interval and offspring distributions and calculating the log-likelihood and
    end-of-outbreak probability given the augmented data.

    Parameters
    ----------
    outbreak_descr_recorded : endoutbreak.OutbreakDescription
        Outbreak description containing recorded transmission data.
    t : int
        Time to which data are assumed to be available when calculating likelihood and
        end-of-outbreak probability. The provided data are automatically truncated to
        this time.
    rng : numpy.random.Generator, optional
        Random number generator used in Metropolis updates. The default is None, in
        which case a new generator is created.

    Attributes
    ----------
    accepted_last_update : bool
        Whether the last update to the infectors was accepted.
    """

    def __init__(self, outbreak_descr_recorded, t, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self._transmission_data_recorded = (
            outbreak_descr_recorded.transmission_data.truncate(t)
        )
        self._offspring_distrib = outbreak_descr_recorded.offspring_distrib
        self._serial_interval_distrib = outbreak_descr_recorded.serial_interval_distrib
        self._t = t
        self._rng = rng
        self._constants_in_formulae = None
        self._log_constants_in_formulae = None
        self._infectee_sample_distrib = None
        self._infector_sample_distribs = None
        self._transmission_data = None
        self._log_lik = None
        self._end_outbreak_prob = None
        self.accepted_last_update = None

    @property
    def constants_in_formulae(self):
        """
        Vector of quantities (one entry for each case in the dataset) only dependent on
        known recorded data/distributions and t. The i^th entry appears in the i^th term
        of the product in the formulae for the likelihood and end-of-outbreak
        probability.
        """
        if self._constants_in_formulae is None:
            t = self._t
            day_reported = self._transmission_data_recorded.day_reported
            offspring_prob_fail = self._offspring_distrib.prob_fail
            serial_interval_distrib = self._serial_interval_distrib
            serial_interval_cdf_curr_vals = serial_interval_distrib.cdf(
                t - day_reported
            )
            self._constants_in_formulae = 1 - offspring_prob_fail * (
                1 - serial_interval_cdf_curr_vals
            )
        return self._constants_in_formulae

    @property
    def log_constants_in_formulae(self):
        """
        Natural logarithm of constants_in_formulae.
        """
        if self._log_constants_in_formulae is None:
            self._log_constants_in_formulae = np.log(self.constants_in_formulae)
        return self._log_constants_in_formulae

    @property
    def infectee_sample_distrib(self):
        """
        scipy.statsrv_discrete object for selecting which infectee to resample the
        infector of during Metropolis updates. Each locally infected case is sampled
        with equal probability
        """
        if self._infectee_sample_distrib is None:
            no_hosts = self._transmission_data_recorded.no_hosts
            imported = self._transmission_data_recorded.imported
            if no_hosts > 1:
                self._infectee_sample_distrib = rv_discrete(
                    values=(np.arange(no_hosts), (~imported) / np.sum(~imported))
                )
        return self._infectee_sample_distrib

    @property
    def infector_sample_distribs(self):
        """
        List of scipy.stats.rv_discrete objects for selecting which infector to assign
        to the corresponding infectee during Metropolis updates. For a locally infected
        case, the probability of selecting a particular infector is proportional to the
        corresponding serial interval probability. For imported cases, the distribution
        is a constant distribution with value -1.
        """
        if self._infector_sample_distribs is None:
            no_hosts = self._transmission_data_recorded.no_hosts
            day_reported = self._transmission_data_recorded.day_reported
            imported = self._transmission_data_recorded.imported
            # Create grid of possible serial intervals for each case (axis 0 of
            # poss_serial_interval_grid and serial_interval_prob_grid indexes the
            # infector, and axis 1 the infectee)
            poss_serial_interval_grid = day_reported[np.newaxis].T - day_reported
            serial_interval_prob_grid = self._serial_interval_distrib.pmf(
                poss_serial_interval_grid
            )
            serial_interval_prob_grid[imported, :] = 0
            # Normalise the vector of possible serial interval probabilities for each
            # infectee
            s = np.sum(serial_interval_prob_grid, axis=1)
            s[imported] = 1
            infector_sample_prob_grid = serial_interval_prob_grid / (s[np.newaxis].T)
            # Create list of distributions for selecting infector for each infectee.
            infector_sample_distribs = [rv_discrete(values=(-1, 1))] * no_hosts
            for infectee_curr in (i for i in range(no_hosts) if not imported[i]):
                infector_sample_distribs[infectee_curr] = rv_discrete(
                    values=(
                        np.arange(no_hosts),
                        infector_sample_prob_grid[infectee_curr, :],
                    )
                )
            self._infector_sample_distribs = infector_sample_distribs
        return self._infector_sample_distribs

    @property
    def transmission_data(self):
        """
        Gets or sets the augmented transmission data (TransmissionDatasetTraced
        object).
        """
        if self._transmission_data is None:
            no_hosts = self._transmission_data_recorded.no_hosts
            day_reported = self._transmission_data_recorded.day_reported
            infector_index = np.array(
                [
                    self.infector_sample_distribs[infectee_curr].rvs(
                        random_state=self._rng
                    )
                    for infectee_curr in range(no_hosts)
                ]
            )
            self.transmission_data = endoutbreak.TransmissionDatasetTraced(
                day_reported, infector_index
            )
        return self._transmission_data

    @transmission_data.setter
    def transmission_data(self, transmission_data_in):
        if len(transmission_data_in.day_reported) != len(
            self._transmission_data_recorded.day_reported
        ):
            raise ValueError(
                """Length of provided transmission data must match length of recorded
                transmission data up to time t. Use the truncate method to truncate
                provided data to time t."""
            )
        self._transmission_data = transmission_data_in
        self._log_lik = None
        self._end_outbreak_prob = None
        self.accepted_last_update = None

    @property
    def log_lik(self):
        """
        The log-likelihood of the augmented transmission data.
        """
        if self._log_lik is None:
            day_reported = self.transmission_data.day_reported
            imported = self.transmission_data.imported
            infector_index = self.transmission_data.infector_index
            no_transmissions = self.transmission_data.no_transmissions
            offspring_distrib = self._offspring_distrib
            offspring_dispersion_param = offspring_distrib.dispersion_param
            serial_interval_distrib = self._serial_interval_distrib
            log_constants_in_formulae = self.log_constants_in_formulae
            ll_transmission_contrib1 = np.sum(
                offspring_distrib.logpmf(no_transmissions)
            )
            ll_transmission_contrib2 = np.sum(
                -(offspring_dispersion_param + no_transmissions)
                * log_constants_in_formulae
            )
            ll_transmission_contrib3 = np.sum(gammaln(no_transmissions + 1))
            ll_si_contrib = np.sum(
                serial_interval_distrib.logpmf(
                    day_reported[~imported] - day_reported[infector_index[~imported]]
                )
            )
            self._log_lik = (
                ll_transmission_contrib1
                + ll_transmission_contrib2
                + ll_transmission_contrib3
                + ll_si_contrib
            )
        return self._log_lik

    @property
    def end_outbreak_prob(self):
        """
        The end-of-outbreak probability given the augmented transmission data
        (calculated using the traced method).
        """
        if self._end_outbreak_prob is None:
            offspring_dispersion_param = self._offspring_distrib.dispersion_param
            no_transmissions = self.transmission_data.no_transmissions
            log_constants_in_formulae = self.log_constants_in_formulae
            self._end_outbreak_prob = np.exp(
                np.sum(
                    (offspring_dispersion_param + no_transmissions)
                    * log_constants_in_formulae
                )
            )
        return self._end_outbreak_prob

    def update_infectors(self):
        """
        Method implementing a Metropolis update of the infector of a single case in
        the dataset.
        """
        if self.transmission_data.no_hosts <= 1:
            # Update is trivially accepted
            self.accepted_last_update = True
            return
        rng = self._rng
        # Select infectee to update and new infector
        update_infectee = self.infectee_sample_distrib.rvs(random_state=rng)
        infector_old = self.transmission_data.infector_index[update_infectee]
        infector_prop = self.infector_sample_distribs[update_infectee].rvs(
            random_state=rng
        )
        if infector_prop == infector_old:
            # Update is trivially accepted
            self.accepted_last_update = True
            return
        # Calculate acceptance probability
        no_transmissions = self.transmission_data.no_transmissions
        offspring_dispersion_param = self._offspring_distrib.dispersion_param
        constants_in_formulae = self.constants_in_formulae
        acceptance_prob = (
            (offspring_dispersion_param + no_transmissions[infector_prop])
            * constants_in_formulae[infector_old]
        ) / (
            (offspring_dispersion_param + no_transmissions[infector_old] - 1)
            * constants_in_formulae[infector_prop]
        )
        # Accept or reject update
        if uniform.rvs(random_state=rng) < acceptance_prob:
            # If accepted, update infectee's infector in the augmented data
            self.transmission_data.swap_infector(update_infectee, infector_prop)
            self.accepted_last_update = True
            # Set values underlying properties to None so that they are recalculated
            # when needed
            self._log_lik = None
            self._end_outbreak_prob = None
        else:
            self.accepted_last_update = False
