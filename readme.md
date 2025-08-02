# RUL estimation for turbofans from CMAPPS dataset

This project uses a feedforward neural network to estimate the 
Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPPS dataset. The neural network uses ReLU activation
MSE loss, Adam optimizer, and layer dropout.

## How to run 

python version - 3.11.9
pip install -r requirements.txt

## project structure

data
    processed
        data_set_1 
            testing_data_1.csv
            train_data_1.csv
            validation_data_1.csv

    raw
        CMAPSSData 
            readme.txt
            RUL_FD001.txt
            test_FD001.txt
            train_FD001.txt

models
    model_class - model class code
        model.py
    
    saved_models - fully trained model
        feedforward_1.pkl

utils - holds the code used to process raw data
    data_processing_1.py

requirements.txt

test_1.py

training_1.py

readme.md

## How to use

Train the model:
    training_1.py 

Test the model:
    test_1.py

## results

Training loss: 1604.0145

Validation loss: 842.8497

Test loss: 24.6239

The test loss is reported as RMSE (square root of MSE) to match the units of the original data.
The average max cycle in the training data was 212.
24.62 / 212 ≈ 0.1161 → ~11.6% margin of error.

## Author

George Peregoy
