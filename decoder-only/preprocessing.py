import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from helpers import *

data_path = "datasets/final-all-months-alarms-with-day-filtered.csv"
save_path = "datasets/tupras_alarm.txt"

df = pd.read_csv(data_path)
print("Data is loaded")
#['MachineName', 'SourceName', 'Message', 'Condition', 'MessageType', 'StartTime', 'EndTime', 'EndMessage', 'TimeDelta', 'Year-Month', 'Day']

#["Text_Before_PV", "SourceName_Identifier", "All_Values_Except_PV", "StartTime", "EndTime", "TimeDelta"]
df_alarm = df[["SourceName", "EndTime", "StartTime", "TimeDelta"]]

# Convert the 'Timestamp' column to a Pandas datetime format
df_alarm['StartTime'] = pd.to_datetime(df['StartTime'])
df_alarm['EndTime'] = pd.to_datetime(df['EndTime'])

alarm_dict_list = df_alarm.to_dict(orient="records") 
print("Dict list is created")

filtered_alarm_list = remove_chattering_alarms(alarm_dict_list, column_name = "SourceName", count_threshold = 3)

alarm_sequence = sequence_segmentation(filtered_alarm_list, time_delta = 30 * 60) # 30 min

alarm_sequence_lst = sequence_lst(alarm_sequence, alarm_definition = "SourceName")

with open(save_path, 'w') as output:
    for row in alarm_sequence_lst:
        output.write(str(row) + '\n')



