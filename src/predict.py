import numpy as np
from tensorflow.keras.models import load_model
from src.data_preprocessing import create_sequences
from src.utils import inverse_scale


def predict_lstm(model_path, df, scaler, seq_length=7):
    """
    df: dataframe procesado con columna 'sales_scaled'
    scaler: MinMaxScaler usado en preprocessing
    """
    # Crear secuencias
    X, y = create_sequences(df, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Cargar modelo
    model = load_model(model_path)

    # Predecir
    y_pred = model.predict(X)

    # Revertir escalado
    y_pred_inv = inverse_scale(scaler, y_pred)
    y_true_inv = inverse_scale(scaler, y)

    return y_true_inv, y_pred_inv
