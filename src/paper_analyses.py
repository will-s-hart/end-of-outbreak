"""
Module for running analyses for the paper using the endoutbreak package. This module
defines the EndOutbreakAnalyses class, which can be used to run analyses for a single
outbreak and produce plots.
"""

import os

import endoutbreak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

rc_params = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
    "axes.autolimit_mode": "round_numbers",
    "savefig.transparent": True,
    "savefig.format": "pdf",
    "svg.fonttype": "none",
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "lines.linewidth": 2,
    "lines.markersize": 10,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
}
sns.set_theme(style="ticks", rc=rc_params)


class EndOutbreakAnalyses:
    """
    Class for running analyses and producing plots for a single outbreak.

    Parameters
    ----------
    outbreak_descr: endoutbreak.OutbreakDescription
        Object containing outbreak data, offspring distribution and serial interval
        distribution.
    options: dict
        Dictionary of options for running analyses. Possible keys are:
        "results_dir": str
            Directory to save results to. Must be provided.
        "figure_dir": str
            Directory to save figures to. Must be provided.
        "t_vec": np.ndarray
            Times to estimate end-of-outbreak probability at. Must be provided.
        "run_sim": bool
            Whether to run the sim method for estimating the end-of-outbreak
            probability. Defaults to False.
        "run_enumerate": bool
            Whether to run the enumerate method for estimating the end-of-outbreak
            probability. Defaults to False.
        "options_mcmc": dict
            Options to pass to endoutbreak.OutbreakDescription.calc_end_outbreak_prob
            when using the MCMC method. Defaults to {}.
        "options_sim": dict or None
            Options to pass to endoutbreak.OutbreakDescription.calc_end_outbreak_prob
            when using the simulation method. Defaults to {}. Ignored if "run_sim" is
            False.
        "options_plot": dict
            Options for plotting. See make_figures method for details. Defaults to {}.

    """

    def __init__(self, outbreak_descr, options):
        self._outbreak_descr = outbreak_descr
        if isinstance(
            outbreak_descr.transmission_data,
            endoutbreak.TransmissionDatasetTraced,
        ):
            self._traced = True
        else:
            self._traced = False
        assert "results_dir" in options and "figure_dir" in options, (
            "'Results' and figure directories ('results_dir' and 'figure_dir') must be"
            " specified in provided options."
        )
        options_in = options
        options = {
            "run_sim": False,
            "run_enumerate": False,
            "options_mcmc": {},
            "options_sim": {},
            "options_plot": {},
        }
        options.update(options_in)
        self._options = options
        self._end_outbreak_prob_df = None
        self._output_mcmc_dfs = None
        self._output_sim_dfs = None

    def run_analyses(self):
        """
        Method for generating end-of-outbreak probability estimates using the MCMC,
        traced (if the transmission tree is available), Nishiura, sim (if the "run_sim"
        option is set to True) and enumerate (if the "run_enumerate" option is set to
        True) methods.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        t_vec = self._options["t_vec"]
        self._end_outbreak_prob_df = pd.DataFrame(
            data={"Day of outbreak": t_vec}
        ).set_index("Day of outbreak")
        # MCMC method
        (
            end_outbreak_prob_mcmc,
            output_mcmc,
        ) = self._outbreak_descr.calc_end_outbreak_prob(
            t_vec, method="mcmc", options=self._options["options_mcmc"]
        )
        self._output_mcmc_dfs = {
            x: output_mcmc[x] for x in ["detail", "acceptance_rate"]
        }
        self._end_outbreak_prob_df["MCMC"] = end_outbreak_prob_mcmc
        # Traced method (if transmission tree available)
        if self._traced:
            (
                end_outbreak_prob_traced,
                _,
            ) = self._outbreak_descr.calc_end_outbreak_prob(t_vec, method="traced")
            self._end_outbreak_prob_df["Traced"] = end_outbreak_prob_traced
        # Nishiura method
        (
            end_outbreak_prob_nishiura,
            _,
        ) = self._outbreak_descr.calc_end_outbreak_prob(t_vec, method="nishiura")
        self._end_outbreak_prob_df["Nishiura"] = end_outbreak_prob_nishiura
        # Enumerate method (if included)
        if self._options["run_enumerate"]:
            (
                end_outbreak_prob_enumerate,
                _,
            ) = self._outbreak_descr.calc_end_outbreak_prob(t_vec, method="enumerate")
            self._end_outbreak_prob_df["Enumerate"] = end_outbreak_prob_enumerate
        # Simulation method (if included)
        if self._options["run_sim"]:
            (
                end_outbreak_prob_sim,
                output_sim,
            ) = self._outbreak_descr.calc_end_outbreak_prob(
                t_vec, method="sim", options=self._options["options_sim"]
            )
            self._output_sim_dfs = {
                x: output_sim[x] for x in ["proportion_sims_matched"]
            }
            self._end_outbreak_prob_df["Simulation"] = end_outbreak_prob_sim
        # Save results
        self._save_results()

    def _save_results(self):
        # Save results to csv files.
        results_dir = self._options["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        eop_path = results_dir + "/end_outbreak_probabilities.csv"
        self._end_outbreak_prob_df.to_csv(eop_path)
        mcmc_detail_path = results_dir + "/mcmc_detail.csv"
        mcmc_acc_path = results_dir + "/mcmc_acc.csv"
        self._output_mcmc_dfs["detail"].to_csv(mcmc_detail_path)
        self._output_mcmc_dfs["acceptance_rate"].to_csv(mcmc_acc_path)
        if self._options["run_sim"]:
            sim_prop_matched_path = results_dir + "/sim_prop_matched.csv"
            self._output_sim_dfs["proportion_sims_matched"].to_csv(
                sim_prop_matched_path
            )

    def load_results(self):
        """
        Method for loading end-of-outbreak probability estimates and output from the
        MCMC and (if included) simulation methods from csv files.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        results_dir = self._options["results_dir"]
        eop_path = results_dir + "/end_outbreak_probabilities.csv"
        self._end_outbreak_prob_df = pd.read_csv(eop_path, index_col="Day of outbreak")
        mcmc_detail_path = results_dir + "/mcmc_detail.csv"
        mcmc_acc_path = results_dir + "/mcmc_acc.csv"
        mcmc_detail_df = pd.read_csv(mcmc_detail_path, header=[0, 1], index_col=[0, 1])
        mcmc_acc_df = pd.read_csv(mcmc_acc_path, index_col=0)
        self._output_mcmc_dfs = {
            "detail": mcmc_detail_df,
            "acceptance_rate": mcmc_acc_df,
        }
        if self._options["run_sim"]:
            sim_prop_matched_path = results_dir + "/sim_prop_matched.csv"
            self._output_sim_dfs = {
                "proportion_sims_matched": pd.read_csv(
                    sim_prop_matched_path, index_col=0
                )
            }

    def make_figures(self):
        """
        Method for producing figures for the paper and saving them to pdf and svg files.

        Plots are produced showing (i) the offspring distribution, (ii) the serial
        interval distribution, (iii) the case data and end-of-outbreak probabilities,
        and, optionally (iv) MCMC output consisting of trace plots of the log-likelihood
        and end-of-outbreak probability, and histograms of end-of-outbreak probability
        across MCMC iterations at each time point.

        Options for plotting can be specified as a dictionary in the "options_plot" key
        of the options dictionary passed to the EndOutbreakAnalyses constructor. The
        following options are available in options["options_plot"]:
        "options_offspring": dict
            Options for plotting the offspring distribution passed to
            _make_offspring_fig as keyword arguments. Defaults to {}.
        "options_serial_interval": dict
            Options for plotting the serial interval distribution passed to
            _make_serial_interval_fig as keyword arguments. Defaults to {}.
        "options_data_eop": dict
            Options for plotting the case data and end-of-outbreak probabilities passed
            to _make_data_eop_figs as keyword arguments. Defaults to {}.
        "options_declaration": dict
            Options for plotting potential end-of-outbreak declaration days passed to
            _make_declaration_figs as keyword arguments. Defaults to {}.
        "options_mcmc_log_lik_trace": dict
            Options for MCMC log-likelihood trace plots passed to
            _make_mcmc_log_lik_trace_figs as keyword arguments. Defaults to {}.
        "options_mcmc_eop_trace": dict
            Options for MCMC end-of-outbreak probability trace plots passed to
            _make_mcmc_eop_trace_figs as keyword arguments. Defaults to {}.
        "options_mcmc_eop_hist": dict
            Options for MCMC end-of-outbreak probability histogram plots passed to
            _make_mcmc_eop_hist_figs as keyword arguments. Defaults to {}.

        Either run_analyses or load_results should be called before this method to
        generate or load data to plot.
        """
        os.makedirs(self._options["figure_dir"], exist_ok=True)
        options_plot = self._options.get("options_plot", {})
        options_offspring = options_plot.get("options_offspring", {})
        options_serial_interval = options_plot.get("options_serial_interval", {})
        options_data_eop = options_plot.get("options_data_eop", {})
        options_data = dict(options_data_eop, plot_eop=False, file_name="data")
        options_declaration = options_plot.get("options_declaration", {})
        options_mcmc_log_lik_trace = options_plot.get("options_mcmc_log_lik_trace", {})
        options_mcmc_eop_trace = options_plot.get("options_mcmc_eop_trace", {})
        options_mcmc_eop_hist = options_plot.get("options_mcmc_eop_hist", {})
        self._make_offspring_fig(fig_handle=1, **options_offspring)
        self._make_serial_interval_fig(fig_handle=2, **options_serial_interval)
        self._make_data_eop_figs(fig_handle=3, **options_data_eop)
        self._make_data_eop_figs(fig_handle=4, **options_data)
        self._make_declaration_figs(fig_handle=5, **options_declaration)
        self._make_mcmc_trace_figs(
            plot_var="Log-likelihood", **options_mcmc_log_lik_trace
        )
        self._make_mcmc_trace_figs(
            plot_var="End-of-outbreak-probability", **options_mcmc_eop_trace
        )
        self._make_mcmc_eop_hist_figs(**options_mcmc_eop_hist)

    def _make_offspring_fig(
        self, fig_handle=None, display_fig=False, keep_open=True, offspring_max=None
    ):
        # Method for plotting the offspring distribution.
        if offspring_max is None:
            offspring_max = int(self._outbreak_descr.offspring_distrib.ppf(0.99))
        figure_dir = self._options["figure_dir"]
        x = np.arange(offspring_max + 1)
        p = self._outbreak_descr.offspring_distrib.pmf(x)
        fig = plt.figure(fig_handle)
        ax = fig.add_subplot(box_aspect=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10]))
        ax.bar(x, p, width=1, alpha=0.5, edgecolor="k")
        ax.set_xlim(left=-0.5, right=offspring_max + 0.5)
        ax.set_xlabel("Number of offspring")
        ax.set_ylabel("Probability")
        offspring_fig_path = figure_dir + "/offspring_distribution.pdf"
        plt.savefig(offspring_fig_path)
        plt.savefig(offspring_fig_path.replace("pdf", "svg"))
        if display_fig:
            plt.show()
        if not keep_open:
            plt.close()

    def _make_serial_interval_fig(
        self,
        fig_handle=None,
        display_fig=False,
        keep_open=True,
        t_max=None,
        xlabel="Serial interval (days)",
    ):
        # Method for plotting the serial interval distribution.
        if t_max is None:
            t_max = int(self._outbreak_descr.serial_interval_distrib.ppf(0.999))
        figure_dir = self._options["figure_dir"]
        t = np.arange(t_max + 1)
        p = self._outbreak_descr.serial_interval_distrib.pmf(t)
        fig = plt.figure(fig_handle)
        ax = fig.add_subplot(box_aspect=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10]))
        ax.bar(t, p, width=1, alpha=0.5, edgecolor="k")
        ax.set_xlim(left=0.5, right=t_max + 0.5)
        ax.set_xticks(np.unique(np.concatenate((np.array([1]), plt.xticks()[0]))))
        ax.set_xlim(left=0.5, right=t_max + 0.5)  # ensures no change to xlim from ticks
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability")
        si_fig_path = figure_dir + "/serial_interval.pdf"
        plt.savefig(si_fig_path)
        plt.savefig(si_fig_path.replace("pdf", "svg"))
        if display_fig:
            plt.show()
        if not keep_open:
            plt.close()

    def _make_data_eop_figs(
        self,
        fig_handle=None,
        display_fig=False,
        keep_open=True,
        file_name="data_end_outbreak_probabilities",
        plot_data=True,
        plot_eop=True,
        include_quantile=True,
        eop_methods=None,
        declaration_day=None,
        t_max=None,
        xlim_left=None,
        xticks=None,
        xlabel="Day of outbreak",
        show_legend=True,
        legend_kwargs=None,
    ):
        # Method for plotting the case data and end-of-outbreak probabilities.
        if t_max is None:
            t_max = np.max(self._options["t_vec"])
        legend_kwargs_in = legend_kwargs or {}
        legend_kwargs = {"loc": "lower right"}
        legend_kwargs.update(legend_kwargs_in)
        figure_dir = self._options["figure_dir"]
        fig = plt.figure(fig_handle)
        if plot_data:
            # Plot case data
            t = np.arange(t_max + 1)
            cases = self._outbreak_descr.transmission_data.get_daily_case_counts_to(
                t_max
            )
            ax1 = fig.add_subplot(box_aspect=1)
            ax1.xaxis.set_major_locator(
                MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10])
            )
            ax1.yaxis.set_major_locator(
                MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10])
            )
            if xlim_left is None:
                xlim_left = -0.5
            ax1.set_xlim(left=xlim_left, right=t_max)
            ax1.set_ylim(0, np.max(cases) * 1)
            if xticks is not None:
                ax1.set_xticks(xticks)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel("Number of cases")
            ax1.bar(
                t,
                cases,
                width=1,
                alpha=0.25,
                color="k",
                edgecolor="k",
                linewidth=0.1,
                clip_on=False,
            )
        if plot_eop:
            # Plot end-of-outbreak probabilities
            if plot_data:
                # Plot alongside case data
                ax2 = ax1.twinx()
                ax2.set_box_aspect(1)
                ax2.spines["right"].set_visible(True)
                ax2.spines["right"].set_color("tab:blue")
                ax2.tick_params(axis="y", colors="tab:blue")
                ax2.set_ylabel("", color="tab:blue")
            else:
                # Plot separately
                fig = plt.figure(fig_handle)
                ax2 = fig.add_subplot(box_aspect=1)
                if xlim_left is None:
                    xlim_left = 0
                ax2.set_xlim(left=xlim_left, right=t_max)
                if xticks is not None:
                    ax2.set_xticks(xticks)
                ax2.set_xlabel(xlabel)
            style = {
                "MCMC": "-",
                "Traced": "k--",
                "Nishiura": "-",
                "Enumerate": "ro",
                "Simulation": "y^",
            }
            plot_df = self._end_outbreak_prob_df
            if eop_methods is not None:
                for method in plot_df.columns:
                    if method not in eop_methods:
                        plot_df = plot_df.drop(columns=method)
            t_vec = self._options["t_vec"]
            t_vec_incl_mask = t_vec <= t_max
            plot_df[t_vec_incl_mask].plot(ax=ax2, style=style, clip_on=False)
            if include_quantile:
                # Compute and plot 95% confidence bounds from MCMC output
                mcmc_detail_df = self._output_mcmc_dfs["detail"]
                eop_post_df = mcmc_detail_df.xs(
                    "End-of-outbreak-probability", level=1, axis=1
                ).xs(True, level="After burn-in")
                eop_mcmc_lower = eop_post_df.quantile(0.025)
                eop_mcmc_lower.index = eop_mcmc_lower.index.astype(int)
                eop_mcmc_upper = eop_post_df.quantile(0.975)
                eop_mcmc_upper.index = eop_mcmc_upper.index.astype(int)
                ax2.fill_between(
                    t_vec[t_vec_incl_mask],
                    eop_mcmc_lower[t_vec_incl_mask],
                    eop_mcmc_upper[t_vec_incl_mask],
                    alpha=0.5,
                    clip_on=False,
                )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("End-of-outbreak-probability")
            plt.legend(**legend_kwargs).set_visible(show_legend)
        if declaration_day is not None:
            try:
                ax = ax1
            except NameError:
                ax = ax2
            ax.plot(
                [declaration_day, declaration_day],
                ax.get_ylim(),
                ":",
                color="tab:orange",
                # linewidth=0.5,
                # clip_on=False,
            )
        data_eop_fig_path = figure_dir + "/" + file_name + ".pdf"
        plt.savefig(data_eop_fig_path)
        plt.savefig(data_eop_fig_path.replace("pdf", "svg"))
        if display_fig:
            plt.show()
        if not keep_open:
            plt.close()

    def _make_declaration_figs(
        self,
        fig_handle=None,
        display_fig=False,
        keep_open=True,
        eop_methods=None,
        declaration_day=None,
        t_rel_max=None,
        ylim=None,
        legend_kwargs=None,
    ):
        last_case_day = np.max(self._outbreak_descr.transmission_data.day_reported)
        if t_rel_max is None:
            t_rel_max = np.max(self._options["t_vec"]) - last_case_day
        legend_kwargs_in = legend_kwargs or {}
        legend_kwargs = {"loc": "upper right"}
        legend_kwargs.update(legend_kwargs_in)
        figure_dir = self._options["figure_dir"]
        thresholds = 1.0 - 0.01 * np.array([5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2])
        if ylim is None:
            ylim = (0, t_rel_max)
        fig = plt.figure(fig_handle)
        ax = fig.add_subplot(box_aspect=1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10]))
        ax.set_ylim(ylim)
        ax.set_xlabel("Threshold risk of additional cases (%)")
        ax.set_ylabel("Days from final case to declaration")
        style = {
            "MCMC": "o",
            "Traced": "kx",
            "Nishiura": "o",
        }
        t_vec = self._options["t_vec"]
        t_vec_incl_mask = (t_vec >= last_case_day) & (
            t_vec <= last_case_day + t_rel_max
        )
        # Get dataframe of the first day the end-of-outbreak probability exceeds each
        # threshold for each method
        eop_df = self._end_outbreak_prob_df[t_vec_incl_mask]
        eop_df = eop_df.drop(columns=["Simulation", "Enumerate"], errors="ignore")
        if eop_methods is not None:
            for method in eop_df.columns:
                if method not in eop_methods:
                    eop_df = eop_df.drop(columns=method)
        first_exceed_day_rel_df = pd.DataFrame(index=thresholds, columns=eop_df.columns)
        for threshold in thresholds:
            first_exceed_days_rel_curr = (eop_df > threshold).idxmax() - last_case_day
            first_exceed_day_rel_df.loc[threshold] = first_exceed_days_rel_curr
        # Get dataframe of posterior samples of the first day the MCMC method
        # end-of-outbreak probability exceeds each threshold
        mcmc_eop_post_df = (
            self._output_mcmc_dfs["detail"]
            .xs("End-of-outbreak-probability", level=1, axis=1)
            .xs(True, level="After burn-in")
            .loc[:, t_vec_incl_mask]
        )
        mcmc_eop_post_df.columns = mcmc_eop_post_df.columns.astype(int)
        first_exceed_day_rel_post_df = pd.DataFrame(
            index=mcmc_eop_post_df.index, columns=thresholds, dtype=int
        )
        for threshold in thresholds:
            first_exceed_days_rel_post_curr = (mcmc_eop_post_df > threshold).idxmax(
                axis=1
            ) - last_case_day
            first_exceed_day_rel_post_df.loc[:, threshold] = (
                first_exceed_days_rel_post_curr
            )
        sns.pointplot(
            data=first_exceed_day_rel_post_df,
            ax=ax,
            errorbar=("pi", 95),
            linestyle="none",
            marker="none",
            color="tab:blue",
            markersize=7.5,
            capsize=0.3,
            linewidth=1.5,
        )
        xticks = ax.get_xticks()
        xticklabels = [f"{x:0.1g}" for x in 100 * (1 - thresholds)]
        ax.set_xticks(xticks, labels=xticklabels)
        first_exceed_day_rel_df.index = xticks
        first_exceed_day_rel_df.plot(ax=ax, style=style, linewidth=5, markersize=7.5)
        plt.legend(**legend_kwargs)
        if declaration_day is not None:
            xlim = ax.get_xlim()
            ax.set_xlim(xlim)  # ensures no change to xlim
            ax.plot(
                xlim,
                [declaration_day - last_case_day, declaration_day - last_case_day],
                ":",
                color="tab:orange",
                zorder=0,
            )
        declaration_fig_path = figure_dir + "/declaration.pdf"
        plt.savefig(declaration_fig_path)
        plt.savefig(declaration_fig_path.replace("pdf", "svg"))
        if display_fig:
            plt.show()
        if not keep_open:
            plt.close()

    def _make_mcmc_trace_figs(
        self,
        plot_var,
        t_vec_plot=None,
        y_lim_vals=None,
        fig_handle_vec=None,
        display_fig=False,
        keep_open=False,
    ):
        # Method for producing MCMC trace plots. Can be used to plot the
        # log-likelihood (plot_var='Log-likelihood') or end-of-outbreak probability
        # (plot_var='End-of-outbreak-probability').
        assert plot_var in [
            "Log-likelihood",
            "End-of-outbreak-probability",
        ], (
            "Invalid plot_var specified. Must be 'Log-likelihood' or"
            " 'End-of-outbreak-probability'."
        )
        if t_vec_plot is None:
            t_vec_plot = []
        if fig_handle_vec is None:
            fig_handle_vec = [None] * len(t_vec_plot)
        figure_dir = self._options["figure_dir"]
        if plot_var == "Log-likelihood":
            trace_fig_dir = figure_dir + "/mcmc_output/log_lik_trace"
        elif plot_var == "End-of-outbreak-probability":
            trace_fig_dir = figure_dir + "/mcmc_output/eop_trace"
        os.makedirs(trace_fig_dir, exist_ok=True)
        options_mcmc = self._options["options_mcmc"]
        no_iterations = options_mcmc["no_iterations"]
        burn_in = options_mcmc["burn_in"]
        mcmc_detail_df = self._output_mcmc_dfs["detail"]
        plot_df = mcmc_detail_df.xs(plot_var, level=1, axis=1)
        plot_df.index = plot_df.index.get_level_values("MCMC iteration")
        for i in range(len(t_vec_plot)):  # pylint: disable=consider-using-enumerate
            # Make trace plot for current time point
            t = t_vec_plot[i]
            fig_handle = fig_handle_vec[i]
            fig = plt.figure(fig_handle)
            ax = fig.add_subplot(box_aspect=1)
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10])
            )
            plot_df[str(t)].plot(ax=ax)
            ax.set_xlim(0, no_iterations)
            if y_lim_vals is not None:
                ax.set_ylim(y_lim_vals[i])
            elif plot_var == "End-of-outbreak-probability":
                ax.set_ylim(0, 1)
            ax.set_xlabel("MCMC iteration")
            ax.set_ylabel(plot_var)
            bot, top = ax.get_ylim()
            ax.plot(burn_in * np.ones(2), [bot, top], "k--")
            ax.set_ylim(bot, top)
            trace_fig_path = trace_fig_dir + "/day_" + str(t) + ".pdf"
            plt.savefig(trace_fig_path)
            plt.savefig(trace_fig_path.replace("pdf", "svg"))
            if display_fig:
                plt.show()
            if not keep_open:
                plt.close()

    def _make_mcmc_eop_hist_figs(
        self, t_vec_plot=None, fig_handle_vec=None, display_fig=False, keep_open=False
    ):
        # Method for producing histograms of end-of-outbreak probability across MCMC
        # iterations.
        if t_vec_plot is None:
            t_vec_plot = []
        if fig_handle_vec is None:
            fig_handle_vec = [None] * len(t_vec_plot)
        figure_dir = self._options["figure_dir"]
        eop_hist_fig_dir = figure_dir + "/mcmc_output/eop_hist"
        os.makedirs(eop_hist_fig_dir, exist_ok=True)
        mcmc_detail_df = self._output_mcmc_dfs["detail"]
        eop_post_df = mcmc_detail_df.xs(
            "End-of-outbreak-probability", level=1, axis=1
        ).xs(True, level="After burn-in")
        for i in range(len(t_vec_plot)):  # pylint: disable=consider-using-enumerate
            t = t_vec_plot[i]
            fig_handle = fig_handle_vec[i]
            fig = plt.figure(fig_handle)
            ax = fig.add_subplot(box_aspect=1)
            ax.yaxis.set_major_locator(
                MaxNLocator(integer=True, steps=[1, 2, 3, 4, 5, 10])
            )
            sns.histplot(
                data=eop_post_df[str(t)],
                ax=ax,
                stat="density",
                binrange=(0, 1),
                binwidth=0.025,
            )
            ax.set_xlim(0, 1)
            # eop_post_df[str(t)].hist(density=True,ax=ax,grid=False)
            ax.set_xlabel("End-of-outbreak probability")
            ax.set_ylabel("Density")
            # ax.grid(False)
            eop_hist_fig_path = eop_hist_fig_dir + "/day_" + str(t) + ".pdf"
            plt.savefig(eop_hist_fig_path)
            plt.savefig(eop_hist_fig_path.replace("pdf", "svg"))
            if display_fig:
                plt.show()
            if not keep_open:
                plt.close()
