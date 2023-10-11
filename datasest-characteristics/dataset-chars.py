import pandas as pd

data_path = "../decoder-only/datasets/final-all-months-alarms-with-day-filtered.csv"

df = pd.read_csv(data_path)

unique_alarms = set(df["SourceName"].values)
num_of_distinct_alarms = len(unique_alarms)

print(f"Number of distinct alarms is {num_of_distinct_alarms}")

with open("unique-alarms.txt", "w") as f:
    for alarm in unique_alarms:
        f.write(alarm+" ")



