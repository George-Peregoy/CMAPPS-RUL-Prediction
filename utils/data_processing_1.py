import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_data():
    pd.set_option('display.max_columns', None) # some columns are truncated without this


    column_names = ["unit", "time", "op1", "op2", "op3"] + [f"sensor {i}" for i in range(1,22)]
    df = pd.read_csv("../data/raw/CMAPSSData/train_FD001.txt", sep=r'\s+', header = None, names = column_names)
    df_test_raw = pd.read_csv("../data/raw/CMAPSSData/test_FD001.txt", sep=r'\s+', header = None, names = column_names)
    y_test = pd.read_csv("../data/raw/CMAPSSData/RUL_FD001.txt", sep=r'\s+', header = None, names = ["RUL"])

    low_var_col = [col for col in df.columns if df[col].std() <= 1e-4]

    df = df.drop(columns= low_var_col)
    df_test_raw = df_test_raw.drop(columns=low_var_col)


    df = df.drop(columns = "sensor 14")
    df_test_raw = df_test_raw.drop(columns = "sensor 14")

    # find Remaining useful life 

    df['max cycle'] = df.groupby("unit")["time"].transform("max") # max cycle

    # used as the y value for prediction
    df["RUL"] = df["max cycle"] - df["time"] # remaining time until last cycle

    units = np.random.permutation(df['unit'].unique())

    train_units = units[:int(len(units) * 0.7)]
    val_units = units[int(len(units) * 0.7):]

    # units were shuffled but rows were not
    # .sample to shuffle all rows, 1 is 100%
    # reset index so original indeces not attached 
    df_train = df[df['unit'].isin(train_units)].sample(frac = 1, random_state=42).reset_index(drop=True)
    df_val = df[df['unit'].isin(val_units)].sample(frac = 1, random_state=42).reset_index(drop=True)

    y_train = df_train["RUL"]
    y_val = df_val["RUL"]

    dropped_columns = ["unit", "RUL", "max cycle"] 
    feature_columns = [col for col in df.columns if col not in dropped_columns]

    x_train = df_train[feature_columns]
    x_val = df_val[feature_columns]

    df_unit_dropped = df_test_raw.drop(columns="unit")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train) # converted to a np array
    x_val_scaled = scaler.transform(x_val) # converted to a np array
    x_test_scaled = scaler.transform(df_unit_dropped)

    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns= feature_columns)
    x_test_scaled_df = pd.concat([df_test_raw["unit"].reset_index(drop = True), pd.DataFrame(x_test_scaled, columns= feature_columns).reset_index(drop=True)], axis = 1)
    df_test_last = x_test_scaled_df.sort_values(["unit", "time"]).groupby("unit").tail(1)
    x_test_final = df_test_last.drop(columns=["unit"])

    df_training_data = pd.concat([pd.DataFrame(x_train_scaled, columns=feature_columns), y_train.reset_index(drop=True)], axis = 1)
    df_validation_data = pd.concat([pd.DataFrame(x_val_scaled, columns=feature_columns), y_val.reset_index(drop=True)], axis = 1)
    df_testing_data = pd.concat([pd.DataFrame(x_test_final, columns=feature_columns).reset_index(drop=True), y_test.reset_index(drop = True)], axis = 1)

    df_training_data.to_csv("../data/processed/data_set_1/train_data_1.csv", index = False)
    df_validation_data.to_csv("../data/processed/data_set_1/validation_data_1.csv", index = False)
    df_testing_data.to_csv("../data/processed/data_set_1/testing_data_1.csv", index = False)

if __name__=="__main__":
    process_data()
