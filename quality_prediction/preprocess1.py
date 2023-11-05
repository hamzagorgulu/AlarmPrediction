import pandas as pd
from glob import glob
import os

data_path = r"C:\Users\hamza\Desktop\AlarmPrediction\datasets\Kırıkkale"

process_files = ['ocak2022', 'subat2022', 'mart2022', 'nisan2022', 'mayıs2022', 'haziran2022', 'temmuz2022', 'agustos2022', 'eylül2022', 'ekim2022', 'kasım2022', 'aralık2022']

all_files = glob(os.path.join(data_path, "*.xlsx"))

breakpoint()
for idx, file in enumerate(all_files):
    if file.split("\\")[-1][:-5] in process_files:
        sub_df = pd.read_excel(file, verbose=True)

        if idx == 0:
            all_df = sub_df
        else:
            all_df = pd.concat([all_df, sub_df], ignore_index=True)
            all_df.to_csv("all_df.csv")