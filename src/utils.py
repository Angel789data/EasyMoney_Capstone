from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, recall_score, roc_curve, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import pickle
import numpy as np  
import json
import joblib
from datetime import datetime
import os
os.chdir(r"C:\Users\angel\Desktop\Proyectos_privados\PROYECTOS\EasyMoney_Capstone")
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency



def eliminar_valores(lista, valores):
    """
    Elimina de 'lista' todos los elementos que estén en 'valores'.
    """
    return [x for x in lista if x not in valores]


def eliminar_columnas(df, columnas):
    """
    Elimina de 'lista' todos los elementos que estén en un dataframe.
    """
    return df.drop(columns=columnas, errors='ignore', inplace= True)


def describe_columns(df, columns=None):
    """
    Imprime el listado de columnas y sus descripciones.

    """
    if columns is None:
        columns = df.columns.tolist()

    print("Listado de columnas y sus descripciones:\n")
    for i, col in enumerate(columns, start=1):
        description = df.attrs.get("descriptions", {}).get(
            col, "No hay descripción disponible.")
        print(f"{i}. {col}: {description}\n")


def copy_meta(df) :
    """
    Crea una copia profunda de un DataFrame incluyendo sus metadatos (df.attrs).
    """
    new_df = df.copy(deep=True)
    new_df.attrs = df.attrs.copy()

    return new_df

def obtener_lista_variables(dataset, target):
    """
    crea 3 listas indicando cuales son de las columnas de dataset sin contar el taget , cuales son boleanas , numericas y categoricas.
    """

    lista_numericas = []
    lista_boolean = []
    lista_categoricas = []

    for i in dataset:
        if (dataset[i].dtype.kind == "f" or dataset[i].dtype.kind == "i") and len(dataset[i].unique()) != 2 and (i not in target):
            lista_numericas.append(i)
        elif (dataset[i].dtype.kind == "f" or dataset[i].dtype.kind == "i") and len(dataset[i].unique()) == 2 and (i not in target):
            lista_boolean.append(i)
        elif (dataset[i].dtype.kind == "O") and i not in target:
            lista_categoricas.append(i)

    return lista_numericas, lista_boolean, lista_categoricas


def umbral_columnas_nulos(dataset, columna, umbral):

    """
    crea una lista de aquellas columnas que tienen un porcentaje muy alto de nulos por encima del umbral 
    """
    lista_umnbral = []
    porcentaje_nulos_n = (
        dataset[columna].isnull().sum() / dataset.shape[0]) * 100

    for i in range(0, len(porcentaje_nulos_n)):
        if (porcentaje_nulos_n.iloc[i]) > umbral:
            lista_umnbral.append(porcentaje_nulos_n.index[i])
            print(porcentaje_nulos_n.index[i], ":", porcentaje_nulos_n.iloc[i] , "% nulos")


    return lista_umnbral


def listas_balanceadas(df, lista, umbral):
    """
    Genera 2 listas indicando cuales son las columnas que se encuentran desbalanceadas , es decir tiene una cantidad una grande de un valor por encima del umbral(90% aprox)
    """
    resultados = []
    for col in lista:
        # Calculamos el porcentaje de cada valor único
        porcentaje = df[col].value_counts(normalize=True) * 100

        # Filtrar valores que superan el umbral
        mayores_80 = porcentaje[porcentaje > umbral]

        if not mayores_80.empty:
            for valor, pct in mayores_80.items():
                print(
                    f"Columna: {col}, Valor: '{valor}', Participación: {round(pct, 2)}%")
                resultados.append((col))
    resultados = list(set(resultados))
    resultados_balencedado = eliminar_valores(lista, resultados)
    return resultados_balencedado, resultados


def segmentacion_categorias_menor_umbral(df, lista, umbral):
    """
    crea dos listas separando aquellas columnas que tienen muchos valores unicos por encima del umbral , es decir si tiene mas de 15 valores unicos por ejemplo
    """
    categorias_mayor_umbral = []
    categorias_menor_umbral = []
    for i in lista:
        if len(df[i].unique()) >= umbral:
            categorias_mayor_umbral.append(i)
        else:
            categorias_menor_umbral.append(i)
            print(i, " : ", len(df[i].unique()))
    return categorias_mayor_umbral, categorias_menor_umbral


# Filtrar correlaciones mayores al umbral (excluyendo la diagonal y duplicados)
def corr_umbral(corr, limite):
    """
    Crea un dataframe que me muestra segun el corr cual de ellos tiene un porcentaje muy alto , es decir son columnas que una puede ser eliminada porque explican lo mismo
    """
    umbral = limite/100

    corr_pairs = (
        # Mantener solo la parte superior de la matriz
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # Convierte a Series con pares (col1, col2)
            .stack()
            .reset_index()                                         # Pasar a DataFrame
    )
    corr_pairs.columns = ['Columna1', 'Columna2', 'Correlacion']

    # Filtrar los pares que superen el umbral
    corr_filtrado = corr_pairs[corr_pairs['Correlacion'].abs() > umbral]
    return corr_filtrado

def pareto_por_columna(df, cols, umbral: float = 0.8):
    """
    Realiza un análisis Pareto en las columnas indicadas.
    
    Para cada columna:
      1. Calcula las frecuencias de cada valor.
      2. Ordena de mayor a menor frecuencia.
      3. Calcula el porcentaje acumulado.
      4. Devuelve los valores que acumulan hasta el umbral.

    """

    resultados = []

    for col in cols:
        if col not in df.columns:
            print(f"⚠️ La columna '{col}' no está en el DataFrame. Se omite.")
            continue

        # Conteo y porcentaje
        freq = df[col].value_counts(dropna=False)
        total = freq.sum()
        porcentaje = (freq / total) * 100

        # Acumulado
        porcentaje_acumulado = porcentaje.cumsum()

        # Filtrar hasta el umbral
        filtrados = freq.index[porcentaje_acumulado <= umbral * 100]

        for valor in filtrados:
            resultados.append({
                "Columna": col,
                "Valor": valor,
                "Frecuencia": freq[valor],
                "Porcentaje": round(porcentaje[valor], 2),
                "Acumulado": round(porcentaje_acumulado[valor], 2)
            })
    tabla_resultados = pd.DataFrame(resultados)
    conteo_valores = tabla_resultados.groupby('Columna')[
        'Valor'].nunique()
    diccionario = tabla_resultados.groupby(
        'Columna')['Valor'].apply(list).to_dict()

    print(conteo_valores)

    return tabla_resultados, conteo_valores, diccionario


def chi2_categoricas(df: pd.DataFrame, columnas: list):
    """
    Calcula el test de Chi-cuadrado para todas las combinaciones de variables categóricas.

    """
    resultados = []

    for col1, col2 in combinations(columnas, 2):
        # Crear tabla de contingencia
        tabla = pd.crosstab(df[col1], df[col2])

        # Test Chi-cuadrado
        chi2, p, dof, expected = chi2_contingency(tabla)

        resultados.append({
            'Variable1': col1,
            'Variable2': col2,
            'Chi2': chi2,
            'p_valor': p,
            'dof': dof
        })

    # Convertimos a DataFrame y ordenamos por p-valor
    df_result = pd.DataFrame(resultados)
    df_result = df_result.sort_values('p_valor').reset_index(drop=True)

    return df_result


def save_metrics(metrics, comentario , path):

    # Agregar timestamp
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics["Comentario"] = comentario

    # Si el archivo existe, lo leemos
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Agregamos el nuevo registro
    data.append(metrics)

    # Guardamos nuevamente
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# --- Función para calcular guardar metricas a cvs---


def json_to_dataframe():

    path = "results/metrics/logistic_metrics.json"

    with open(path, "r") as f:
        data = json.load(f)

    rows = []

    # Transformar estructura
    for experiment in data:
        for proceso, metrics_dict in experiment["METRICS"].items():
            for metrica, valor in metrics_dict.items():
                rows.append({
                    "Nombre_Modelo": experiment["Nombre_Modelo"],
                    "timestamp": experiment.get("timestamp"),
                    "PROCESO": proceso,
                    "METRICA": metrica,
                    "VALOR": valor,
                    "Comentario": experiment.get("Comentario")
                })

    # Crear DataFrame
    df = pd.DataFrame(rows)
    return df


# --- Función para calcular métricas ---
def calcular_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    return acc, pre, rec, f1, roc, cm


def evaluate_model(model, X_train, X_test, val_df_X, Y_train, Y_test, val_df_y, comentario , threshold=0.5):
    # --- Definir threshold ---
    # ajustable para mejorar recall/F1

    # --- Predicciones con probabilidades ---
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_valid_proba = model.predict_proba(val_df_X)[:, 1]

    # --- Predicciones ajustadas según threshold ---
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)
    y_valid_pred = (y_valid_proba >= threshold).astype(int)

    # --- Métricas TRAIN ---
    acc_train, pre_train, rec_train, f1_train, roc_train, cm_train = calcular_metrics(
        Y_train, y_train_pred, y_train_proba)

    # --- Métricas TEST ---
    acc_test, pre_test, rec_test, f1_test, roc_test, cm_test = calcular_metrics(
        Y_test, y_test_pred, y_test_proba)

    # --- Métricas VALID ---
    acc_val, pre_val, rec_val, f1_val, roc_val, cm_val = calcular_metrics(
        val_df_y, y_valid_pred, y_valid_proba)
    

    # --- Mostrar resultados ---
    print(
        f"\nTRAIN -> Accuracy: {acc_train:.3f}, Precision: {pre_train:.3f}, Recall: {rec_train:.3f}, F1: {f1_train:.3f}, ROC AUC: {roc_train:.3f}")
    print("Matriz de confusión:")
    print(cm_train, "\n")

    print(
        f"TEST  -> Accuracy: {acc_test:.3f}, Precision: {pre_test:.3f}, Recall: {rec_test:.3f}, F1: {f1_test:.3f}, ROC AUC: {roc_test:.3f}")
    print("Matriz de confusión:")
    print(cm_test, "\n")

    print(
        f"VALID -> Accuracy: {acc_val:.3f}, Precision: {pre_val:.3f}, Recall: {rec_val:.3f}, F1: {f1_val:.3f}, ROC AUC: {roc_val:.3f}")
    print("Matriz de confusión:")
    print(cm_val, "\n")

    name_model = model.__class__.__name__

    metrics = {
        "Nombre_Modelo": name_model,
        "METRICS": {
            "TRAIN": {
                "Accuracy": round(acc_train, 3),
                "Precision": round(pre_train, 3),
                "Recall": round(rec_train, 3),
                "F1": round(f1_train, 3),
                "ROC_AUC": round(roc_train, 3)
            },
            "TEST": {
                "Accuracy": round(acc_test, 3),
                "Precision": round(pre_test, 3),
                "Recall": round(rec_test, 3),
                "F1": round(f1_test, 3),
                "ROC_AUC": round(roc_test, 3)
            },
            "VALIDACION": {
                "Accuracy": round(acc_val, 3),
                "Precision": round(pre_val, 3),
                "Recall": round(rec_val, 3),
                "F1": round(f1_val, 3),
                "ROC_AUC": round(roc_val, 3)
            }
        }
    }

    save_metrics(metrics, comentario , path="results/metrics/logistic_metrics.json")
    df = json_to_dataframe()

    df.to_csv("results/metrics/model_metrics.csv", index=False)

    # --- Gráfica ROC para TEST ---
    fpr, tpr, thresholds_roc = roc_curve(Y_test, y_test_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=name_model + ' (AUC = %0.2f)' % roc_test)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - TEST')
    plt.legend(loc="lower right")
    plt.savefig(f"results/plots/roc_curve_{name_model}.png", dpi=300)
    plt.show()


def import_dev_val(dev_df, val_df,  target):
    # -

    # ## 1. Carga de data procesada , imputada y limpia

    # -

    # ## 2. En esta parte se divide la parte de validacion con develop

    dev_df_X = dev_df.drop(target, axis=1)
    dev_df_y = dev_df[[target]]

    val_df_X = val_df.drop(target, axis=1)
    val_df_y = val_df[[target]]

    return dev_df_X, dev_df_y, val_df_X, val_df_y

    # -

    # ## 3. Se genera la divicion de train y test


def split_dev(dev_df_X, dev_df_y):
    X_train, X_test, Y_train, Y_test = train_test_split(dev_df_X,
                                                        dev_df_y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test


def pipeline_evaluacion(X_train, X_test, Y_train, Y_test, val_df_X, val_df_y, scoring, model, param_grid, act=True):

    model_test = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=5
    )

    if act == True:
        model_test.fit(X_train, Y_train)

    else:
        model_test.fit(X_train, np.array(Y_train).ravel().astype(int))

    print("Mejores parámetros:", model_test.best_params_)
    print("Mejor F1 promedio:", model_test.best_score_)

    best_params_text = str(model_test.best_params_)

    # -

    # ## 6. Se genera las metricas del modelo

    comentario = best_params_text
    model_best = model_test.best_estimator_
    evaluate_model(model_best, X_train, X_test, val_df_X, Y_train,
                   Y_test, val_df_y, comentario, threshold=0.5)

    # ## 7. Resultados Totales

    # +
    name_model = model_best.__class__.__name__

    df = pd.read_csv("results/metrics/model_metrics.csv",
                     sep=",", low_memory=False)
    df[df['Nombre_Modelo'] == name_model]

    # -

    # ## 8.Guardar modelo

    with open(f"results/model/fit_model/{name_model}.pkl", "wb") as f:
        pickle.dump(model_best, f)



    

