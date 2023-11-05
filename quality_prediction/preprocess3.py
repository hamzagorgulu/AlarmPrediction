import pandas as pd
from helpers.analyze_df import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import normalize
from math import floor

# this script show how the input-target pairs are created.

input_data_path = r"C:\Users\hamza\Desktop\AlarmPrediction\sorted_shifted_df_rm_cols3.csv"
target_data_path = r"C:\Users\hamza\Desktop\AlarmPrediction\Lab Verisi - Plt 1650 DizelSarj2.xlsx"

df_target = pd.read_excel(target_data_path, verbose=True)
df_input = pd.read_csv(input_data_path)
df_input["TimeStamp"] = pd.to_datetime(df_input["TimeStamp"])

#print(check_df(df_input))
#print(check_df(df_target))

#cat_cols, num_cols, cat_but_car = grab_col_names(df_target, cat_th=10, car_th=20)
#target_values = df_target["DHP DİZEL SARJ wt% 90 °C"].values

input_chunks = chunk_dataframe_by_time(df_input)  # all inputs have the same shape
paired_data = pair_input_output(df_target, input_chunks)

inputs = np.array([paired_data[i][0] for i in range(len(paired_data))])
output = np.array([paired_data[i][1] for i in range(len(paired_data))])

f = open("process_inputs.npy", "wb")
np.save(f, inputs)

f = open("quality_targets.npy", "wb")
np.save(f, output)