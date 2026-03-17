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
cca = pd.read_csv("data/raw/customer_commercial_activity.csv")
cp = pd.read_csv("data/raw/customer_products.csv")
csd = pd.read_csv("data/raw/customer_sociodemographics.csv")
sales = pd.read_csv("data/raw/sales.csv")
prod_desc = pd.read_csv("data/raw/product_description.csv")

cca = cca.drop('Unnamed: 0', axis=1)
cp = cp.drop('Unnamed: 0', axis=1)
csd = csd.drop('Unnamed: 0', axis=1)
sales = sales.drop('Unnamed: 0', axis=1)
prod_desc = prod_desc.drop('Unnamed: 0', axis=1)


# %% [markdown]
# ## 2. Unificacion de tablas

# %%
df = cca.merge(cp, on=["pk_cid", "pk_partition"], how="left") \
        .merge(csd, on=["pk_cid", "pk_partition"], how="left")

sales = sales.merge(prod_desc, left_on="product_ID",
                    right_on="pk_product_ID", how="left")
sales_drop = sales.drop('product_ID', axis=1)
eliminar_columnas(sales_drop, ['pk_product_ID', 'pk_sale'])

df_sum = (
    sales_drop.groupby(['cid', 'month_sale'], as_index=False)['net_margin']
    .sum()
    .rename(columns={'net_margin': 'total_net_margin'})
)

df_dummies = pd.get_dummies(sales_drop[['product_desc', 'family_product']],
                            prefix=['product_desc', 'family_product'])
df_encoded = pd.concat(
    [sales_drop[['cid', 'month_sale']], df_dummies], axis=1)
df_hot = (
    df_encoded
    .groupby(['cid', 'month_sale'], as_index=False)
    .sum()
)

sales_cid = df_sum.merge(df_hot, on=["cid", "month_sale"], how="left")
sales_cid['pk_partition'] = sales_cid['month_sale'].astype(
    'string').str[:7]

sales_cid = sales_cid.rename(columns={"cid": "pk_cid"})
# Drop the 'Unnamed: 0' column from sales before merging to avoid conflicts

df_v1 = df.merge(sales_cid, on=["pk_cid", "pk_partition"], how="left")


# %% [markdown]
# ## 3. identificador de columnas

# %%
productos_lista = ['product_desc_em_acount', 'product_desc_emc_account', 'product_desc_payroll',
                   'product_desc_payroll_account',
                   'product_desc_funds', 'product_desc_long_term_deposit', 'product_desc_securities',
                   'product_desc_short_term_deposit',
                   'product_desc_loans', 'product_desc_mortgage', 'product_desc_credit_card', 'product_desc_debit_card',
                   'product_desc_pension_plan']
account = ['product_desc_em_acount', 'product_desc_emc_account',
           'product_desc_payroll', 'product_desc_payroll_account']
investment = ['product_desc_funds', 'product_desc_long_term_deposit',
              'product_desc_securities', 'product_desc_short_term_deposit']
loan = ['product_desc_loans', 'product_desc_mortgage']
payment_card = ['product_desc_credit_card', 'product_desc_debit_card']
pension_plan = ['product_desc_pension_plan']
familias = ['family_product_account', 'family_product_investment',
            'family_product_loan', 'family_product_payment_card',	'family_product_pension_plan']

df_v1[account] = (df_v1[account] > 0).astype(int)
df_v1[investment] = (df_v1[investment] > 0).astype(int)
df_v1[loan] = (df_v1[loan] > 0).astype(int)
df_v1[payment_card] = (df_v1[payment_card] > 0).astype(int)
df_v1[pension_plan] = (df_v1[pension_plan] > 0).astype(int)
df_v1[productos_lista] = (df_v1[productos_lista] > 0).astype(int)
df_v1[familias] = (df_v1[familias] > 0).astype(int)


# %% [markdown]
# ## 4.Enriquesimiento de la data

# %%
df_v1 = df_v1[df_v1['age'] >= 17]

df_v1["entry_date_date"] = pd.to_datetime(
    df_v1["entry_date"], format="%Y-%m")
df_v1["pk_partition_date"] = pd.to_datetime(
    df_v1["pk_partition"], format="%Y-%m")
df_v1["diff_meses"] = (df_v1["pk_partition_date"].dt.year - df_v1["entry_date_date"].dt.year) * 12 + \
    (df_v1["pk_partition_date"].dt.month -
     df_v1["entry_date_date"].dt.month)

df_v1["cantidad_de_meses"] = df_v1.groupby(
    "pk_cid")["pk_partition_date"].rank(method="first", ascending=True)

# Acumulados de actividad
df_v1 = df_v1.sort_values(
    ["pk_cid", "pk_partition_date"], ascending=[True, True])
df_v1["suma_acumulada_active"] = df_v1.groupby(
    "pk_cid")["active_customer"].cumsum()


bool_cols = ['loans', 'funds', 'securities', 'long_term_deposit',
             'credit_card', 'payroll_account', 'emc_account', 'debit_card', 'em_acount', 'payroll', 'pension_plan']

familias = ['family_product_investment',
            'family_product_payment_card', 'family_product_pension_plan']

df_v1[bool_cols] = (df_v1[bool_cols] > 0).astype(int)

df_v1["n_products"] = df_v1[bool_cols].sum(axis=1)

df_v1["has_any_deposit"] = df_v1["short_term_deposit"] | df_v1["long_term_deposit"]
df_v1["has_investments"] = df_v1["funds"] | df_v1["securities"]
df_v1["full_payroll_relationship"] = df_v1["payroll"] & df_v1["payroll_account"]
df_v1["has_cards"] = df_v1["credit_card"] | df_v1["debit_card"]

df_v1["heavy_user"] = (df_v1["n_products"] >= 3).astype(int)
df_v1["segm_plus_client"] = df_v1["entry_channel"].isin(
    ["KCB", "KAE", "KHO", "KHN", "KHL", "KDT"]).astype(int)
df_v1['otra_familia_flag'] = (df_v1[familias].sum(axis=1) > 0).astype(int)

df_v1["payroll_and_pension"] = df_v1["payroll"] & df_v1["pension_plan"]
df_v1['payroll_and_card'] = (
    df_v1['payroll'] == 1) & (df_v1['has_cards'] == 1)
df_v1['payroll_and_card'] = df_v1['payroll_and_card'].astype(int)

df_v1["pension_x_payroll"] = df_v1["payroll"] * df_v1["pension_plan"]
df_v1["invest_x_age"] = df_v1["has_investments"] * df_v1["age"]
df_v1["cards_x_payroll"] = df_v1["has_cards"] * df_v1["payroll"]

bins = [16, 25, 35, 45, 55, 65, 200]
labels = ['17-24', '25-34', '35-44', '45-54', '55-64', '65+']
df_v1['age_group'] = pd.cut(df_v1['age'], bins=bins, labels=labels)

df_v1["region_code"] = df_v1["region_code"].fillna(0).astype(int)

eliminar_columnas(df_v1, ['entry_date_date', 'pk_partition_date'])


# %%
df_v1.to_csv("data/processed/output_data_enriquesida.csv", sep="|", index=False)

