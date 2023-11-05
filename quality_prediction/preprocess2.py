import os
import sys
import pandas as pd
from helpers.analyze_df import *

data_path = r"C:\Users\hamza\Desktop\AlarmPrediction\all_df13.csv"

all_df = pd.read_csv(data_path)

breakpoint()

print(check_df(all_df))
cat_cols, num_cols, cat_but_car = grab_col_names(all_df, cat_th=10, car_th=20)

# explore categorical columns
print(f"cat_cols are: {cat_cols}")
for cat_col in cat_cols:
    cat_summary(all_df, cat_col)

# explore numerical columns
print(f"num_cols are: {num_cols}")
for num_col in num_cols:
    num_summary(all_df, num_col, plot=True)

# explore categorical but cardinal columns
print(f"cat_but_car cols are: {cat_but_car}")
print("Types of the cat_but_car columns are:")
print(set([str(all_df[cat_but_car[i]].dtypes) for i in range(len(cat_but_car))])) # they put in this categories bc they are all object type.

# there are 2 cat_cols and there are onyl 1 unique value in both of them. Delete those columns.
cat_cols_to_be_deleted = [cat_col for cat_col in cat_cols]

# Heatmap of NaN locations
sns.heatmap(all_df.isna(), cbar=False)
plt.show()

# Apply the function to each row
shifted_df = all_df.apply(shift_values_left, axis=1)
