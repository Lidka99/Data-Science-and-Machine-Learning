import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_wine() -> np.ndarray:
    return pd.read_csv(r'C:\Users\Lidka\Desktop\DS i MUM\zad4\data\wine-clustering.csv').iloc[:, :].to_numpy()


def read_heart() -> np.ndarray:
    return pd.read_csv(r'C:\Users\Lidka\Desktop\DS i MUM\zad4\data\heart_disease_patients.csv').iloc[:, 1:11].to_numpy()


def read_customers() -> np.ndarray:
    df = pd.read_csv(r'C:\Users\Lidka\Desktop\DS i MUM\zad4\data\customers.csv')
    df.drop('Segmentation', axis=1).to_numpy()

    df.iloc[:, 1] = LabelEncoder().fit_transform(df.iloc[:, 1])
    df.iloc[:, 5] = LabelEncoder().fit_transform(df.iloc[:, 5])
    df.iloc[:, 7] = LabelEncoder().fit_transform(df.iloc[:, 7])
    df.iloc[:, 9] = LabelEncoder().fit_transform(df.iloc[:, 9])

    print()

    return df.iloc[:, 1:9].to_numpy()
