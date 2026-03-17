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
df = pd.read_csv("data/processed/output_data_enriquesida_muestra.csv", sep="|", low_memory=False)


# %% [markdown]
# ## 2. Imputación de nulos

# %%
# Imputación de categorias

df['entry_channel'].fillna(df['entry_channel'].mode()[0], inplace=True)
df['segment'].fillna(df['segment'].mode()[0], inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

# Imputación de números
df["salary"].fillna(df["salary"].median(), inplace=True)

# Imputación de Boleanos
df["payroll"].fillna(df["payroll"].mode()[0], inplace=True)
df["pension_plan"].fillna(df["pension_plan"].mode()[0], inplace=True)


# %% [markdown]
# ## 3. procesamiento a variables numericas

# %%
# variable extra enriquesida
df["rel_age_segment"] = df.groupby(
    "segment")["age"].transform(lambda x: (x - x.mean()) / x.std())

lista_cod_reg = pd.read_csv("data/raw/codigo_region.csv", sep=";")

lista_categoricas = ['pk_partition', 'entry_date', 'region_code',
                     'entry_channel', 'segment', 'gender', 'age_group']

lista_numericas = ['age', 'salary', 'diff_meses', 'cantidad_de_meses', 'suma_acumulada_active',
                   'n_products', 'rel_age_segment', 'pension_x_payroll', 'invest_x_age', 'cards_x_payroll']

lista_boolean = ['active_customer', 'short_term_deposit', 'loans', 'funds', 'securities', 'long_term_deposit', 'credit_card',
                 'payroll_account', 'emc_account', 'debit_card', 'em_acount', 'product_desc_credit_card',
                 'product_desc_debit_card', 'product_desc_em_acount', 'product_desc_emc_account', 'product_desc_funds',
                 'product_desc_long_term_deposit', 'product_desc_payroll', 'product_desc_payroll_account',
                 'product_desc_pension_plan', 'product_desc_securities', 'product_desc_short_term_deposit',
                 'family_product_account', 'family_product_investment', 'family_product_payment_card',
                 'family_product_pension_plan', 'payroll', 'pension_plan', 'has_any_deposit', 'has_investments',
                 'full_payroll_relationship', 'has_cards', 'heavy_user', 'segm_plus_client', 'otra_familia_flag',
                 'payroll_and_pension', 'payroll_and_card']

target = ['sales_account']

total_lista = lista_numericas + lista_boolean + lista_categoricas + target
df = df[total_lista]

# Procesamiento de Numeros
MinMax = MinMaxScaler()  # Definimos la función de sklearn que vamos a usar
df[lista_numericas] = MinMax.fit_transform(df[lista_numericas])

# Procesamiento de Categorias
lista_entry_channel = ['KHE', 'KFC', 'KAT', 'KHQ', 'KHK', 'KHM']

df["entry_channel"] = np.where(df["entry_channel"].isin(
    lista_entry_channel), df["entry_channel"], "otros")

df["region_code"] = df["region_code"].astype(int)

lista_region_code = [28, 8, 46, 41, 30, 15, 29, 3, 11, 36, 33, 50, 6, 35, 47, 18, 45, 37, 10, 14, 21, 2, 39, 12, 13, 7, 17,
                     43, 32, 27, 9, 25, 48, 24, 4, 38, 19]

df["region_code"] = np.where(df["region_code"].isin(
    lista_region_code), df["region_code"], 0)
df["region_code"] = df["region_code"].astype(int)

df = df.merge(lista_cod_reg, how='left', on='region_code')
df['Comunidad_autonoma'] = np.where(
    df['region_code'] == 0, 'Otros', df['Comunidad_autonoma'])

df["pk_partition_year"] = df["pk_partition"].str.split(
    "-", expand=True)[0].astype(object)
df["pk_partition_mes"] = df["pk_partition"].str.split(
    "-", expand=True)[1].astype(object)
df["entry_date_year"] = df["entry_date"].str.split(
    "-", expand=True)[0].astype(object)
df["entry_date_mes"] = df["entry_date"].str.split(
    "-", expand=True)[1].astype(object)

df["gender"] = np.where(df["gender"] == "H", 1, 0)

lista_ohe = ['region_code', 'entry_channel',
             'segment', 'age_group', 'Comunidad_autonoma']


# Transformamos las variables categóricas mediante OHE usando get_dummies
df_ohc = pd.get_dummies(data=df, columns=lista_ohe, dtype=int)

eliminar_columnas(df_ohc, ["pk_partition", "entry_date"])



# %%
df_ohc.to_csv("data/processed/output_data_enriquesida_muestra_limpia.csv", sep="|", index=False)


# %% [markdown]
# ## 2. En esta parte separo la parte de validacion y develop que sera los ultimos meses 

# %%
df2 = df_ohc[~((df_ohc["pk_partition_year"] == '2019')
           & (df_ohc["pk_partition_mes"] == '05'))]

dev_df = df2[~((df2['pk_partition_year'].astype(int) == 2019) & ( df2['pk_partition_mes'].astype(int) >= 3))]  # development = train + test
val_df = df2[((df2['pk_partition_year'].astype(int) == 2019) & ( df2['pk_partition_mes'].astype(int) >= 3))]  # validation

# %%
dev_df.to_csv("data/processed/output_data_enriquesida_muestra_limpia_develop.csv", sep="|", index=False)
val_df.to_csv("data/processed/output_data_enriquesida_muestra_limpia_validacion.csv", sep="|", index=False)

