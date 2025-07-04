import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def remove_outliers_iqr(df):
    """
    Menghapus outlier dari DataFrame menggunakan metode IQR.
    Diterapkan hanya pada kolom numerik.

    Parameters:
    df (DataFrame): Dataset

    Returns:
    DataFrame tanpa outlier
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include='number').columns

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

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
    df = df.dropna()

    df = df.drop_duplicates()

    df = remove_outliers_iqr(df)

    X = df.drop(columns=["Grades"])
    y = df["Grades"]

    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    dir_preprocess_data = "Predict_Student_Performance_preprocessing"

    df_train.to_csv(f"{dir_preprocess_data}/train.csv", index=False)
    df_test.to_csv(f"{dir_preprocess_data}/test.csv", index=False)

    return X, y, scaler

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_preprocess.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    X, y, scaler = preprocess_data(df)

    print("=== Scaled Features ===")
    print(X)

    print("\n=== Target ===")
    print(y)
