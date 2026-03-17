# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from tqdm import tqdm
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib
import json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, recall_score, roc_curve, precision_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency
import pickle
import os
os.chdir(r"C:\Users\angel\Desktop\Proyectos_privados\PROYECTOS\EasyMoney_Capstone")

pd.set_option('display.max_columns', None)

set_config(transform_output="pandas")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)

# -


def train_model(name_model, scoring, param_grid, target="sales_account", act=True):

    # ## 1. Carga de data procesada , imputada y limpia

    dev_df = pd.read_csv(
        "data/processed/output_data_enriquesida_muestra_limpia_develop.csv", sep="|", low_memory=False)
    val_df = pd.read_csv(
        "data/processed/output_data_enriquesida_muestra_limpia_validacion.csv",  sep="|", low_memory=False)

    dev_df_X, dev_df_y, val_df_X, val_df_y = import_dev_val(
        dev_df, val_df, target)

    # ## 2. Resalizar el split de train y test en develop

    X_train, X_test, Y_train, Y_test = split_dev(dev_df_X, dev_df_y)

    # ## 3. Carga de modelo seleccionado

    # +

    with open(f"results/model/fit_model/{name_model}.pkl", "rb") as f:
        model = pickle.load(f)

    # -

    # ## 3. Hiperparametros , target , optimizador

    # +


    # -

    # ## 4. Evaluacion de metricas del modelo
  
    # +
    pipeline_evaluacion(X_train, X_test, Y_train, Y_test,
                        val_df_X, val_df_y, scoring, model, param_grid, act)

    # +
    df = pd.read_csv("results/metrics/model_metrics.csv",
                     sep=",", low_memory=False)

    df[df["Nombre_Modelo"] == name_model]

    return df[df["Nombre_Modelo"] == name_model]
