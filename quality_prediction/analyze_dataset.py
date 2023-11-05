import pandas as pd
from helpers.analyze_df import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import normalize
from math import floor

input_data_path = r"C:\Users\hamza\Desktop\AlarmPrediction\sorted_shifted_df_rm_cols3.csv"
target_data_path = r"C:\Users\hamza\Desktop\AlarmPrediction\Lab Verisi - Plt 1650 DizelSarj2.xlsx"

df_target = pd.read_excel(target_data_path, verbose=True)
df_input = pd.read_csv(input_data_path)
df_input["TimeStamp"] = pd.to_datetime(df_input["TimeStamp"])

breakpoint()

input_na_mean_dict = identify_columns_with_most_nans(df_input) # inputs are all clear
target_na_mean_dict = identify_columns_with_most_nans(df_target) # target contains nan values for some columns

visualize_nan_values(df_input)
visualize_nan_values(df_target)

# check all outliers in target: 
[check_outlier(df_target, col) for col in df_target.columns if col not in ["Tarih", "Vardiya", "DHP DİZEL SARJ Renk -"]]
[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, False]

# check all outliers in target: 
[check_outlier(df_input, col) for col in df_input.columns if col not in ["TimeStamp"]]
[True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True]

# outlier by indexes
indexes = grab_outliers(df_input, df_input.columns[1], index=True)