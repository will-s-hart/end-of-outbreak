"""
Module containing base classes and functions for the endoutbreak package.
"""

import functools

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import nbinom, rv_discrete

from ._end_outbreak_prob import calc_end_outbreak_prob as calc_end_outbreak_prob_func

# Data classes


class TransmissionDataset:
    """
    Class for storing line list transmission data.

    Parameters
    ----------
    day_reported : numpy array
        Time of reporting of each case.
    imported : numpy array
        Boolean array indicating whether each case was imported or not.

    Attributes
    ----------
    no_hosts : int
        Number of hosts in the transmission data.
    """

    def __init__(self, day_reported, imported):
        self.day_reported = day_reported
        self.imported = imported
        self.no_hosts = np.size(day_reported)
        self._daily_case_counts = None

    def __getitem__(self, mask):
        """
        Returns a new TransmissionDataset object containing only cases indicated by the
        provided mask.

        Parameters
        ----------
        mask : numpy array
            Boolean array indicating which cases to keep.

        Returns
        -------
        TransmissionDataset
            New TransmissionDataset containing transmission data.
        """
        return TransmissionDataset(self.day_reported[mask], self.imported[mask])

    @property
    def daily_case_counts(self):
        """
        Gets daily total case counts between times 0 and the maximum time in the
        transmission data.
        """
        if self._daily_case_counts is None:
            t_max = np.max(self.day_reported)
            daily_case_counts = np.zeros(t_max + 1, dtype=int)
            (t_vals, count_vals) = np.unique(self.day_reported, return_counts=True)
            daily_case_counts[t_vals] = count_vals
            self._daily_case_counts = daily_case_counts
        return self._daily_case_counts

    def get_daily_case_counts_to(self, t_max):
        """
        Gets daily total case counts between times 0 and t_max.

        Parameters
        ----------
        t_max : int
            Maximum time to calculate daily case counts up to.

        Returns
        -------
        daily_case_counts : numpy array
            Daily case counts between times 0 and t_max.
        """
        t_max_data = np.max(self.day_reported)
        if t_max <= t_max_data:
            return self.daily_case_counts[: (t_max + 1)]
        return np.concatenate(
            (self.daily_case_counts, np.zeros(t_max - t_max_data, dtype=int))
        )

    def truncate(self, t_trunc):
        """
        Returns a truncated transmission dataset including only cases reported up to
        time t_trunc.

        Parameters
        ----------
        t_trunc : int
            Time to truncate data to.

        Returns
        -------
        TransmissionDataset
            New TransmissionDataset containing transmission data.
        """
        kept_indicator = self.day_reported <= t_trunc
        return self[kept_indicator]

    def to_csv(self, csv_path):
        """
        Saves transmission data to csv file.

        Parameters
        ----------
        csv_path : str
            Path to csv file to save transmission data to.

        Returns
        -------
        None
        """
        df = pd.DataFrame(
            {
                "day_reported": self.day_reported,
                "imported": self.imported,
            }
        )
        df.to_csv(csv_path, index=False)


class TransmissionDatasetTraced(TransmissionDataset):
    """
    Class for storing line list transmission data with known infectors.

    Inherits from TransmissionDataset.

    Parameters
    ----------
    day_reported : numpy array
        Time of reporting of each case.
    infector_index : numpy array
        Index of infector for each case. Imported cases have infector index -1.

    Attributes
    ----------
    imported : numpy array
        Boolean array indicating whether each case was imported or not.
    no_hosts : int
        Number of hosts in the transmission data.
    """

    def __init__(self, day_reported, infector_index):
        imported = infector_index == -1
        TransmissionDataset.__init__(self, day_reported, imported)
        self.infector_index = infector_index
        self._no_transmissions = None

    def __getitem__(self, mask):
        """
        Returns a new TransmissionDatasetTraced object containing only cases indicated
        by the provided mask.

        Overrides TransmissionDataset.__getitem__.

        Parameters
        ----------
        mask : numpy array
            Boolean array indicating which cases to keep.

        Returns
        -------
        TransmissionDatasetTraced
            New TransmissionDatasetTraced containing transmission data.
        """
        return TransmissionDatasetTraced(
            self.day_reported[mask], self.infector_index[mask]
        )

    @property
    def no_transmissions(self):
        """
        The total number of transmissions from each host.
        """
        if self._no_transmissions is None:
            (hosts, freqs) = np.unique(self.infector_index, return_counts=True)
            no_transmissions = np.zeros(self.no_hosts, dtype=int)
            no_transmissions[hosts[1:]] = freqs[1:]  # excludes imported cases
            self._no_transmissions = no_transmissions
        return self._no_transmissions

    def to_csv(self, csv_path):
        """
        Saves transmission data to csv file.

        Overrides TransmissionDataset.to_csv.

        Parameters
        ----------
        csv_path : str
            Path to csv file to save transmission data to.

        Returns
        -------
        None
        """
        df = pd.DataFrame(
            {
                "day_reported": self.day_reported,
                "infector_index": self.infector_index,
            }
        )
        df.to_csv(csv_path, index=True)

    def swap_infector(self, infectee, infector_new):
        """
        Updates the infector of a case in the dataset.

        Parameters
        ----------
        infectee : int
            Index of infectee to update.
        infector_new : int
            Index of new infector.

        Returns
        -------
        None
        """
        infector_old = self.infector_index[infectee]
        self.infector_index[infectee] = infector_new
        self._no_transmissions[infector_old] -= 1
        self._no_transmissions[infector_new] += 1


class OutbreakDescription:
    """
    Class for storing the complete description of an outbreak, characterised by a
    transmission dataset (which may or may not be traced), offspring distribution and
    serial interval distribution.

    Methods for calculating the end-of-outbreak probability are defined on this class.

    Parameters
    ----------
    transmission_data : TransmissionDataset or TransmissionDatasetTraced
        Transmission dataset.
    offspring_distrib : OffspringDistribution
        Negative binomial offspring distribution.
    serial_interval_distrib : SerialIntervalDistribution
        Discrete serial interval distribution, assumed to take strictly positive values.
    """

    def __init__(self, transmission_data, offspring_distrib, serial_interval_distrib):
        self.transmission_data = transmission_data
        self.offspring_distrib = offspring_distrib
        self.serial_interval_distrib = serial_interval_distrib

    def truncate(self, t_trunc):
        """
        Returns a new OutbreakDescription object with transmission data truncated at
        time t_trunc.

        See TransmissionDataset.truncate.

        Parameters
        ----------
        t_trunc : int
            Time to truncate transmission data to.

        Returns
        -------
        OutbreakDescription
            New OutbreakDescription object with truncated transmission data.
        """
        return OutbreakDescription(
            self.transmission_data.truncate(t_trunc),
            self.offspring_distrib,
            self.serial_interval_distrib,
        )

    def calc_end_outbreak_prob(self, t, method, options=None):
        """
        Uses a specified method to calculate the probability that the outbreak has ended
        by time t given the provided data.

        Parameters
        ----------
        t : int or numpy array
            Time point(s) to calculate probability for.
        method : str
            Method to use for calculating probability. One of "traced", "nishiura",
            "mcmc", "enumerate" or "sim".
        options : dict, optional
            Dictionary of options for the specified method. Default options are used for
            any options not provided. Available option keys and defaults are as follows:
            "parallel": Whether to calculate probabilities at different time points in
                parallel. Default is False.
            "n_jobs": Number of processes to use if parallel is True. Default is 1.
            "print_progress": Whether to print progress updates. Default is False.
            "rng_seed": Seed for random number generator. Default is None.
            "no_iterations": Number of MCMC iterations to run per time point ("mcmc"
                method only). Default is 1000.
            "burn_in": Number of MCMC iterations to discard as burn-in ("mcmc" method
                only). Default is 200.
            "thinning": Only one MCMC iteration is kept out of every number of
            iterations specified by this option following the burn-in period ("mcmc"
                method only). Default is 1.
            "no_matches": Number of matched simulations to find per time point ("sim"
                method only). Default is 1000.

        Returns
        -------
        end_outbreak_prob : float or numpy array
            Probability that the outbreak has ended by time t.
        output : dict
            Dictionary of additional output from the calculation.
        """
        # Check that the method is valid
        assert method in [
            "traced",
            "nishiura",
            "mcmc",
            "enumerate",
            "sim",
        ], "Invalid method"
        # Parse options
        options_in = options or {}
        options = {
            "parallel": False,
            "n_jobs": 1,
            "print_progress": False,
            "rng_seed": None,
            "no_iterations": 1000,
            "burn_in": 200,
            "thinning": 1,
            "no_matches": 1000,
        }
        options.update(options_in)
        # Run calculation using the calc_end_outbreak_prob function from the
        # _end_outbreak_prob module
        end_outbreak_prob, output = calc_end_outbreak_prob_func(
            self, t, method, options
        )
        return end_outbreak_prob, output

    def calc_end_outbreak_prob_traced(self, t):
        """
        Calculates the probability that the outbreak has ended by scalar time t given
        the provided data using the traced method.

        Parameters
        ----------
        t : int
            Time point to calculate probability at.

        Returns
        -------
        end_outbreak_prob : float
            Probability that the outbreak has ended by time t.
        """
        # Check that transmission data is traced
        assert isinstance(
            self.transmission_data, TransmissionDatasetTraced
        ), "Transmission tree must be provided to use the 'traced' method"
        # Run calculation on truncated data if t is less than the maximum time in the
        # data
        day_reported = self.transmission_data.day_reported
        if t < np.max(day_reported):
            return self.truncate(t).calc_end_outbreak_prob_traced(t)
        # Main calculation
        offspring_dispersion_param = self.offspring_distrib.dispersion_param
        offspring_prob_fail = self.offspring_distrib.prob_fail
        serial_interval_distrib = self.serial_interval_distrib
        serial_interval_cdf_vec = serial_interval_distrib.cdf(t - day_reported)
        no_transmissions = self.transmission_data.no_transmissions
        end_outbreak_prob = np.prod(
            (1 - offspring_prob_fail * (1 - serial_interval_cdf_vec))
            ** (offspring_dispersion_param + no_transmissions)
        )
        return end_outbreak_prob

    def calc_end_outbreak_prob_nishiura(self, t):
        """
        Calculates the probability that the outbreak has ended by scalar time t given
        the provided data using the Nishiura method.

        Parameters
        ----------
        t : int
            Time point to calculate probability at.

        Returns
        -------
        end_outbreak_prob : float
            Probability that the outbreak has ended by time t.
        """
        # Run calculation on truncated data if t is less than the maximum time in the
        # data
        day_reported = self.transmission_data.day_reported
        if t < np.max(day_reported):
            return self.truncate(t).calc_end_outbreak_prob_nishiura(t)
        # Main calculation
        offspring_dispersion_param = self.offspring_distrib.dispersion_param
        offspring_prob_fail = self.offspring_distrib.prob_fail
        serial_interval_distrib = self.serial_interval_distrib
        serial_interval_cdf_vec = serial_interval_distrib.cdf(t - day_reported)
        end_outbreak_prob = np.prod(
            (
                (1 - offspring_prob_fail)
                / (1 - offspring_prob_fail * serial_interval_cdf_vec)
            )
            ** offspring_dispersion_param
        )
        return end_outbreak_prob


# Classes for offspring and serial interval distributions


class OffspringDistribution:
    """
    Class for negative binomial offspring distributions. Provides a thin wrapper around
    the scipy.stats.nbinom class.

    Attributes
    ----------
    reproduction_number : float
        Reproduction number of offspring distribution.
    dispersion_param : float
        Dispersion parameter of offspring distribution.
    prob_fail : float
        Parameter of negative binomial distribution (no obvious epidemiological
        interpretation).
    """

    def __init__(self, reproduction_number, dispersion_param):
        prob_success = dispersion_param / (reproduction_number + dispersion_param)
        self._obj = nbinom(dispersion_param, prob_success)
        self.reproduction_number = reproduction_number
        self.dispersion_param = dispersion_param
        self.prob_fail = 1 - prob_success

    def pmf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.nbinom.pmf.
        """
        return self._obj.pmf(*args, **kwargs)

    def logpmf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.nbinom.logpmf.
        """
        return self._obj.logpmf(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.nbinom.cdf.
        """
        return self._obj.cdf(*args, **kwargs)

    def ppf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.nbinom.ppf.
        """
        return self._obj.ppf(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.nbinom.rvs.
        """
        return self._obj.rvs(*args, **kwargs)


class SerialIntervalDistribution:
    """
    Class for serial interval distributions. Provides a thin wrapper around the
    scipy.stats.rv_discrete class.
    """

    def __init__(self, vals, probs):
        if np.any(vals <= 0):
            raise ValueError(
                "Serial interval distribution must take strictly positive values"
            )
        if not np.issubdtype(vals.dtype, np.integer):
            raise ValueError("Serial interval distribution must take integer values")
        self._obj = rv_discrete(values=(vals, probs))
        self.vals = vals
        self.probs = probs

    def pmf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.rv_discrete.pmf.
        """
        return self._obj.pmf(*args, **kwargs)

    def logpmf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.rv_discrete.logpmf.
        """
        return self._obj.logpmf(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.rv_discrete.cdf.
        """
        return self._obj.cdf(*args, **kwargs)

    def ppf(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.rv_discrete.ppf.
        """
        return self._obj.ppf(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        """
        Wrapper for scipy.stats.rv_discrete.rvs.
        """
        return self._obj.rvs(*args, **kwargs)

    def to_csv(self, csv_path):
        """
        Saves serial interval distribution to csv file.

        Parameters
        ----------
        csv_path : str
            Path to csv file to save serial interval distribution to.

        Returns
        -------
        None
        """
        df = pd.DataFrame(
            {
                "serial_interval": self.vals,
                "probability": self.probs,
            }
        )
        df.to_csv(csv_path, index=False)


# Data import function


def load_outbreak_dataset(data_path, data_format="csv", imported_infector_id=None):
    """
    Function for loading line list transmission data from a csv or excel file.

    The csv file should have colums (("day_reported" or "date_reported") and optionally
    ("imported" or "infector_index" or ("id" and "infector_id"))), as described below:
    day_reported: Time of reporting of each case (integer).
    date_reported: Time of reporting of each case (date string in format recognised by
        pandas).
    imported: Boolean indicator of whether each case was imported or not.
    infector_index: Index of infector for each case. Cases should be ordered by time of
        reporting, and infectors should then by indexed in the order they appear in the
        csv file, with the first case having index 0; imported cases should have
        infector index -1.
    id: ID of each case (string).
    infector_id: ID of infector for each case (string). If the "imported" column is not
        supplied, imported cases should have infector ID specified by the
        imported_infector_id argument (which should be supplied in this case).

    If none of "imported", "infector_index" or "infector_id" are present, only the
    first reported case (or cases, if multiple cases occurred on the first case date)
    is assumed to be imported.

    Parameters
    ----------
    data_path : str
        Path to csv file containing transmission data.
    imported_id : str, optional
        Value of "infector_id" used to signify imported cases (only required if "id" and
        "infector_id" columns are used).

    Returns
    -------
    TransmissionDataset or TransmissionDatasetTraced
        Transmission dataset object.
    """
    # Load data as pandas dataframe
    dtypes_in = {"id": "string", "infector_id": "string", "imported": "bool"}
    if data_format == "csv":
        df = pd.read_csv(data_path, dtype=dtypes_in)
    elif data_format == "excel":
        df = pd.read_excel(data_path, dtype=dtypes_in)
    else:
        raise ValueError("Invalid data format. Supported formats are 'csv' and 'excel'")
    # Convert dates to days since first case if necessary
    if "day_reported" not in df.columns and "date_reported" in df.columns:
        df["date_reported"] = pd.to_datetime(df["date_reported"])
        df["day_reported"] = (df["date_reported"] - df["date_reported"].min()).dt.days
    else:
        df["day_reported"] = df["day_reported"] - df["day_reported"].min()
    # Find IDs of imported cases if not provided
    if "imported" in df.columns or "infector_index" in df.columns:
        pass  # imported column already present or not needed
    elif "infector_id" in df.columns:
        assert imported_infector_id is not None, (
            "Need to specify" + "'imported_infector_id'"
        )
        df["imported"] = df["infector_id"] == imported_infector_id
    else:
        df["imported"] = df["day_reported"] == df["day_reported"].min()
    # Order data by time of reporting and then by imported status
    if "infector_index" not in df.columns:
        df.sort_values(
            ["day_reported", "imported"], ascending=[True, False], inplace=True
        )
        df.reset_index(inplace=True, drop=True)
    # Create infector index column if not present but infector ID column is
    if "infector_index" not in df.columns and "infector_id" in df.columns:
        id_index_mapping = dict(zip(df["id"].to_list(), df.index.to_list()))
        id_index_mapping[imported_infector_id] = -1
        df["infector_index"] = [
            id_index_mapping[x] for x in df["infector_id"].to_list()
        ]
    # Create TransmissionDataset or TransmissionDatasetTraced object
    if "infector_index" in df.columns:
        return TransmissionDatasetTraced(
            df["day_reported"].to_numpy(), df["infector_index"].to_numpy()
        )
    return TransmissionDataset(df["day_reported"].to_numpy(), df["imported"].to_numpy())


# Functions for importing serial interval distributions and for discretising continuous
# serial interval distributions


def load_discr_serial_interval_distrib(si_path, data_format="csv"):
    """
    Function for loading a discrete serial interval distribution from a csv file.

    The csv file should have columns "serial_interval" and "probability", as described
    below:
    serial_interval: Serial interval (days). Only positive values are allowed, and if
        zero is included it must have probability zero.
    probability: Probability of corresponding serial interval.

    Parameters
    ----------
    si_path : str
        Path to csv file containing serial interval distribution.
    data_format : str, optional
        Format of data file. One of "csv" or "excel". Default is "csv".

    Returns
    -------
    SerialIntervalDistribution
        Serial interval distribution object.
    """
    if data_format == "csv":
        df = pd.read_csv(si_path)
    elif data_format == "excel":
        df = pd.read_excel(si_path)
    else:
        raise ValueError("Invalid data format. Supported formats are 'csv' and 'excel'")
    vals = df["serial_interval"].to_numpy()
    probs = df["probability"].to_numpy()
    if 0 in vals:
        if probs[vals == 0] != 0:
            raise ValueError(
                "Serial interval distribution must take strictly positive values"
            )
        probs = probs[vals != 0]
        vals = vals[vals != 0]
    return SerialIntervalDistribution(vals, probs)


def discretise_serial_interval(serial_interval_dist_cont):
    """
    Function for discretising a continuous serial interval distribution using the method
    described in https://doi.org/10.1093/aje/kwt133 (web appendix 11).
    """

    def _integrand_fun(x, y):
        # To get probability mass function at time x, need to integrate this expression
        # with respect to y between y=x-1 and and y=x+1
        return (1 - abs(x - y)) * serial_interval_dist_cont.pdf(y)

    # Set up vector of x values and pre-allocate vector of probabilities
    x_max = int(serial_interval_dist_cont.ppf(0.9999))
    x_vec = np.arange(0, x_max + 1)
    p_vec = np.zeros(len(x_vec))
    # Calculate probability mass function at each x value
    for i in range(len(x_vec)):  # pylint: disable=consider-using-enumerate
        x = x_vec[i]
        integrand = functools.partial(_integrand_fun, x)
        p_vec[i] = integrate.quad(
            integrand,
            x - 1,
            x + 1,
        )[0]
    # Assign mass from 0 to 1
    x_vec = x_vec[1:]
    p_vec[1] = p_vec[1] + p_vec[0]
    p_vec = p_vec[1:]
    # Assign residual mass to x_max
    p_vec[-1] = p_vec[-1] + 1 - np.sum(p_vec)
    return SerialIntervalDistribution(x_vec, p_vec)
