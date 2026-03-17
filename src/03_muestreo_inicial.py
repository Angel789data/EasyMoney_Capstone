# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Importar librerias

# %%
import os
os.chdir(r"C:\Users\angel\Desktop\Proyectos_privados\PROYECTOS\EasyMoney_Capstone")
from src.utils import *
from sklearn.calibration import CalibratedClassifierCV
from sklearn import set_config


from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.stats import chi2_contingency
pd.set_option('display.max_columns', None)

set_config(transform_output="pandas")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
target = ["sales_account"]
lista_cod_reg = pd.read_csv("data/raw/codigo_region.csv", sep=";")


# %% [markdown]
# ## 1. carga de datos

# %%
# pk_cid, pk_partition
df_us = pd.read_csv(
    "data/processed/output_data_enriquesida.csv", sep="|", low_memory=False)


# %% [markdown]
# ## 2. Seleccion de muestra y target

# %%

tarjet = "family_product_account"
cod_cip_sales_account = df_us[df_us[tarjet] > 0]['pk_cid'].unique()
df_con_sales_account = df_us[df_us['pk_cid'].isin(cod_cip_sales_account)]

df_con_sales_account["month"] = pd.to_datetime(
    df_con_sales_account["pk_partition"])

purchase_month = (df_con_sales_account[df_con_sales_account[tarjet] > 0]
                  .loc[:, ["pk_cid", "month"]].rename(columns={"month": "purchase_month"}))

df_merged = df_con_sales_account.merge(
    purchase_month, on="pk_cid", how="left")

df_merged["prev_month"] = df_merged["purchase_month"] - \
    pd.DateOffset(months=1)

df_prev_con_sales = df_merged[df_merged["month"]
                              == df_merged["prev_month"]]

eliminar_columnas(df_prev_con_sales, [
    'month',	'purchase_month',	'prev_month'])

# Se seleciona solo las columnas de pk_cid y pk_partition para sobrecargar el proceso
aleatorio1 = df_prev_con_sales[['pk_cid', 'pk_partition']]

# se eligen de forma aleatoria
df_1mes_aleat_con_sales = aleatorio1.groupby("pk_cid").apply(
    lambda x: x.sample(1)).reset_index(drop=True)

# Se genera un inner join con la tabla principal para solo tener los aleatorios
df_1mes_con_venta = df_prev_con_sales.merge(
    df_1mes_aleat_con_sales, on=["pk_cid", "pk_partition"], how="inner")

# Marcador de venta (target)
df_1mes_con_venta['sales_account'] = 1

df_sin_sales_account = df_us[~df_us['pk_cid'].isin(cod_cip_sales_account)]

# Se seleciona solo las columnas de pk_cid y pk_partition para sobrecargar el proceso
aleatorio2 = df_sin_sales_account[['pk_cid', 'pk_partition']]

# se eligen de forma aleatoria
df_1mes_aleat_sin_sales = aleatorio2.groupby("pk_cid").apply(
    lambda x: x.sample(1)).reset_index(drop=True)

# Se genera un inner join con la tabla principal para solo tener los aleatorios
df_1mes_sin_venta = df_sin_sales_account.merge(
    df_1mes_aleat_sin_sales, on=["pk_cid", "pk_partition"], how="inner")

df_1mes_sin_venta['sales_account'] = 0

# Unir y mezclar
df_final = pd.concat([df_1mes_sin_venta, df_1mes_con_venta])

# Columnas que no sirven
eliminar_columnas(df_final, ["pk_cid", "country_id", "deceased"])


# %% [markdown]
# ## 3. guardar muestreo inicial - mes anterior a la compra y mes aleatorio sin compra

# %%
df_final.to_csv("data/processed/output_data_enriquesida_muestra.csv", sep="|", index=False)

