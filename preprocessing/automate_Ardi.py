
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Melakukan preprocessing dataset:
    - Scaling semua fitur numerik
    - Mengembalikan X (fitur), y (target) dan scaler

    Parameters:
    df (DataFrame): Dataset asli

    Returns:
    X (DataFrame): Fitur yang telah diproses
    y (Series): Target (Grades)
    scaler (StandardScaler): Objek scaler yang digunakan
    """
    X = df.drop(columns=["Grades"])
    y = df["Grades"]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler
