import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

def check_df(dataframe, head=8):
    print("##### Shape #####")
    print(dataframe.shape)
    print("##### Types #####")
    print(dataframe.dtypes)
    print("##### Tail #####")
    print(dataframe.tail(head))
    print("##### Head #####")
    print(dataframe.head(head))
    print("##### Null Analysis #####")
    print(dataframe.isnull().sum())
    print("##### Quantiles #####")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name):
    return print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                               "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

def cat_sum_plot(dataframe, col_name, plot=False, hue=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        if hue:
            sns.countplot(x=dataframe[col_name], hue=col_name, data=dataframe)
            plt.show(block=True)
        else:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

    print("\n\n")

def target_sum_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

def target_sum_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = num_but_cat + cat_cols
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(f"Numerical Columns: {len(num_cols)}")
    print(f"Categoric but Cardinal: {len(cat_but_car)}")
    print(f"Numeric but Categoric: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outliers(dataframe, col_name, index=False):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])  # concat the columns
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])
    return temp_df

# gives you the percentage of the nan values in every column
def identify_columns_with_most_nans(dataframe):
    #na_columns = [col for col in dataframe.columns if dataframe[col].isnull().mean() > threshold_percentage]
    na_mean_dict = {}
    for col in dataframe.columns:
        na_mean_dict[col] = dataframe[col].isnull().mean()
    return na_mean_dict

# Function to shift non-NaN values to the left
def shift_values_left(row):
    non_nans = row.dropna().values
    return pd.Series(np.concatenate([non_nans, np.full(len(row) - len(non_nans), np.nan)]))

def visualize_nan_values(df):
    # Use Seaborn's heatmap to visualize the NaN values
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis')
    plt.title('NaN Values Visualization')
    plt.show()

def chunk_dataframe_by_time(df):
    chunks = []
    for day in df['TimeStamp'].dt.date.unique():
        day_df = df[df['TimeStamp'].dt.date == day]
        chunk_1 = day_df[day_df['TimeStamp'].dt.hour < 8]
        chunk_2 = day_df[(day_df['TimeStamp'].dt.hour >= 8) & (day_df['TimeStamp'].dt.hour < 16)]
        chunk_3 = day_df[day_df['TimeStamp'].dt.hour >= 16]
        chunks.extend([chunk_1, chunk_2, chunk_3])
    return chunks

# Function to pair input chunks with corresponding output values based on matching dates and time intervals
def pair_input_output(df_target, input_chunks):
    paired_data = []
    not_found_lst = []
    for idx, chunk in enumerate(input_chunks):
        chunk_time = chunk['TimeStamp'].iloc[0]
        chunk_date = chunk_time.date()
        if 0 <= chunk_time.hour < 8:
            matching_row = df_target[(df_target['Tarih'].dt.date == chunk_date) & (df_target['Vardiya'].str.contains('V1_24/08'))]
        elif 8 <= chunk_time.hour < 16:
            matching_row = df_target[(df_target['Tarih'].dt.date == chunk_date) & (df_target['Vardiya'].str.contains('V2_08/16'))]
        elif 16 <= chunk_time.hour <= 23:
            matching_row = df_target[(df_target['Tarih'].dt.date == chunk_date) & (df_target['Vardiya'].str.contains('V3_16/24'))]
        if not matching_row.empty:
            output_value = matching_row['DHP DİZEL SARJ wt% 90 °C'].values  # Replace the columns with your actual output column
            chunk_values = chunk.drop("TimeStamp", axis=1).values
            paired_data.append((chunk_values, output_value))
        else:
            print(f"matching_row is not found at input chunk idx: {idx}")
            not_found_lst.append(idx)
    print(f"Not found idx list: {not_found_lst}")
    return paired_data

# Not found idx list: [0, 1, 2, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 164, 280, 281, 282, 283, 284, 285, 372, 373, 374, 375, 376, 377, 378, 467, 468, 469, 496, 555, 556, 557, 558, 559, 560, 561, 593, 594, 595, 596, 597, 598, 688, 689, 711, 712, 775, 776]