import pandas as pd
import os

def try_null(df):
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    for col in colunas_numericas:
        df[col] = df[col].fillna(df[col].median())

    colunas_categoricas = df.select_dtypes(include=['object']).columns
    for col in colunas_categoricas:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def codificar_categorias(df):
    colunas_categoricas = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)
    return df

df = pd.read_csv("../data/bronze/hmeq_bronze.csv")
df_limpo = try_null(df)
df_final = codificar_categorias(df_limpo)

os.makedirs("../data/silver", exist_ok=True)
df_final.to_csv("../data/silver/hmeq_silver.csv", index=False)

print("Dados salvos na camada Silver")
