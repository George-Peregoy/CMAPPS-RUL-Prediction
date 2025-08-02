from models.model_class.model import * 
import pandas as pd

def test():
    df_testing_data = pd.read_csv("data/processed/data_set_1/testing_data_1.csv")

    y = df_testing_data["RUL"].to_numpy()
    x = df_testing_data.drop(columns=["RUL"]).to_numpy()

    model = Model.load("models/saved_models/feedforward_1.pkl")
    model.test(x, y)

if __name__=="__main__":
    test()