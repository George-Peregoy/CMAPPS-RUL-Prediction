# RUL estimation for turbofans from C-MAPPS dataset

This project uses a feedforward neural network to estimate the 
Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPPS dataset. The neural network uses ReLU activation
MSE loss, Adam optimizer, and layer dropout.

## How to run 

python version - 3.11.9
pip install -r requirements.txt

## project structure

```
project_root/
│
├── data/
│   ├── raw/
│   │   └── CMAPSSData/
│   │       ├── readme.txt
│   │       ├── RUL_FD001.txt
│   │       ├── test_FD001.txt
│   │       └── train_FD001.txt
│   └── processed/
│       └── data_set_1/
│           ├── train_data_1.csv
│           ├── validation_data_1.csv
│           └── testing_data_1.csv
│
├── models/
│   ├── model_class/
│   │   └── model.py
│   └── saved_models/
│       └── feedforward_1.pkl
│
├── utils/
│   └── data_processing_1.py
│
├── requirements.txt
├── training_1.py
├── test_1.py
├── README.md
```

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

## Author

George Peregoy
