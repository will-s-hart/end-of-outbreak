"""
Script for formatting outbreak data csv files into the format used by endoutbreak.
For each outbreak specified in OUTBREAKS, this script will load the raw data from
"../data/outbreak/raw_data.xlsx" using endoutbreak.load_outbreak_dataset, and save the
formatted dataset to "../results/outbreak/formatted_data.csv".
"""

import os

import endoutbreak

OUTBREAKS = ["Ebola_Likati", "Ebola_Equateur"]


def _run_formatting(outbreak):
    # Helper function for running data formatting for specified outbreak.
    curr_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(curr_dir, "../data/" + outbreak + "/raw_data.xlsx")
    formatted_data_dir = os.path.join(curr_dir, "../results/" + outbreak)
    os.makedirs(formatted_data_dir, exist_ok=True)
    formatted_data_path = os.path.join(formatted_data_dir + "/formatted_data.csv")
    transmission_data = endoutbreak.load_outbreak_dataset(
        raw_data_path, data_format="excel", imported_infector_id="0"
    )
    transmission_data.to_csv(formatted_data_path)


if __name__ == "__main__":
    for outbreak_curr in OUTBREAKS:
        _run_formatting(outbreak_curr)
