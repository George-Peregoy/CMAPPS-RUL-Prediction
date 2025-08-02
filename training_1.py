import pandas as pd
import numpy as np
from models.model_class.model import *

def train():
    # Data is processed in utils/data_processing_1 and stored in data/processed
    df_training_data = pd.read_csv("data/processed/data_set_1/train_data_1.csv")
    df_validation_data = pd.read_csv("data/processed/data_set_1/validation_data_1.csv")

    np.random.seed(42) # for debugging 

    y_train_df = df_training_data["RUL"]
    X_train_df = df_training_data.drop(columns = "RUL")

    y_val_df = df_validation_data["RUL"]
    X_val_df = df_validation_data.drop(columns = "RUL")

    y_train_np = y_train_df.to_numpy() # NN expects np arrays
    X_train_np = X_train_df.to_numpy()

    y_val_np = y_val_df.to_numpy()
    X_val_np = X_val_df.to_numpy()

    # Define layers, activation, loss function, and optimizer
    layer_size = [17, 64, 32, 1] # 17 features in input data
    model = Model(layer_size=layer_size, dropout_rate=0.3)
    model.set(activation=ReLU(), loss_function=Loss_MSE(), 
            optimizer=Optimizer_Adam(learning_rate=1e-3, beta_1=0.9,beta_2=.999))
    model.build()

    model.train(X_train_np, y_train_np, epochs=2000, batch_size=128)
    model.validate(X_val_np, y_val_np)
    model.plot_loss()

    model.save("models/saved_models/feedforward_1.pkl")

if __name__=="__main__":
    train()


        

    

