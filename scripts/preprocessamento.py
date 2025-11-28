# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Felipe Ottoni Pereira e Letícia Almeida Paulino de Alencar Ferreira
# RA: 804317 (Felipe) e 800408 (Letícia)
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import pandas as pd

def clean_users_info(df):
    """
    Limpa o DataFrame removendo casos inválidos/nulos
    e preenchendo valores nulos relacionados as informações de users_info
    """

    df_clean = df.copy()
    event_columns = ['Stress Inducement', 'Aerobic Exercise', 'Anaerobic Exercise']

    # Remover colunas com valores nulos, "Yes*" ou "Yes**" nas colunas acima
    df_clean = df_clean.dropna(subset=event_columns)
    df_clean = df_clean[
        ~df_clean[event_columns].isin(['Yes*', 'Yes**']).any(axis=1)
    ]

    # substituir campos nulos por média do gênero nas colunas demográficas
    demographic_columns = ['Age', 'Height (cm)', 'Weight (kg)']

    for col in demographic_columns:
        # média agrupada por gênero para cada linha
        mean_by_gender = df_clean.groupby('Gender')[col].transform('mean')
        # substitui valores nulos pela respectiva média
        df_clean[col] = df_clean[col].fillna(mean_by_gender)

    return df_clean

def removeOutliers(df_dataset):
    """
    Remove outliers das coluna numéricas

    -- Entradas:
        df_dataset: DataFrame com outliers

    -- Saída:
        df_dataset: DataFrame sem os outliers
    """
    df_sem_outliers = df_dataset.copy()
    colunas_numericas = df_dataset.select_dtypes(include=np.number).columns
    mascara_final = pd.Series([True] * len(df_dataset), index=df_dataset.index)
    
    for atributo in colunas_numericas:
        # estatisticas
        Q1 = df_sem_outliers[atributo].quantile(0.25)
        Q3 = df_sem_outliers[atributo].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # mascara onde True significa que o valor não é um outlier
        mascara_coluna = (df_sem_outliers[atributo] >= limite_inferior) & (df_sem_outliers[atributo] <= limite_superior)
        
        # combina a mascara da coluna com a msscara final
        mascara_final = mascara_final & mascara_coluna

    df_dataset = df_sem_outliers[mascara_final]    
    
    return df_dataset

def replace_outliers(df, label_col="Label", factor=1.5):
    """
    Identifica outliers usando IQR e substitui os valores extremos 
    pela média dos valores NÃO outliers da mesma classe.

    Parâmetros:
    -----------
    df: DataFrame original
    label_col: Nome da coluna de classes (Label)
    factor: multiplicador do IQR (1.5 = padrão)

    Retorna:
    --------
    df_corrigido: DataFrame com outliers substituídos
    """

    df_corrigido = df.copy()
    colunas_numericas = df.select_dtypes(include="number").columns

    for col in colunas_numericas:

        # Calcula limites globais para detectar outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        li = Q1 - factor * IQR
        ls = Q3 + factor * IQR

        # Identifica outliers
        mask_outlier = (df_corrigido[col] < li) | (df_corrigido[col] > ls)

        if mask_outlier.any():
            # Para cada classe, calcula a média dos valores não-outliers
            medias_por_classe = (
                df_corrigido.loc[~mask_outlier].groupby(label_col)[col].mean()
            )

            # Substitui outliers pela média da classe correspondente
            for idx in df_corrigido[mask_outlier].index:
                classe = df_corrigido.loc[idx, label_col]
                df_corrigido.at[idx, col] = medias_por_classe.get(classe, df_corrigido[col].mean())

    return df_corrigido


def check_remove_outlier(df_original, df_limpo):
    """
    Valida a remoçãode outliers

    -- Entradas:
        df_original: DataFrame original antes da remoção dos outliers
        df_limpo: DataFrame após a remoção

    -- Saída:
        df_relatiorio: DataFrame mostrando dados da verificação: quantidade de outliers removidos, etc.
    """
    # seleciona colunas numéricas
    cols = df_original.select_dtypes(include='number').columns
    
    resultado = {}
    for col in cols:
        # para cada coluna calcula estatisticas
        Q1 = df_original[col].quantile(0.25)
        Q3 = df_original[col].quantile(0.75)
        IQR = Q3 - Q1
        li = Q1 - 1.5*IQR
        ls = Q3 + 1.5*IQR

        # contagem dos dados
        mask_orig = (df_original[col] < li) | (df_original[col] > ls)
        mask_limpo = (df_limpo[col] < li) | (df_limpo[col] > ls)
        total = int(mask_orig.sum())
        restam = int(mask_limpo.sum())
        remov = total - restam
        resultado[col] = {
            'original': total,
            'removidos': remov,
            'restantes': restam
        }

        df_relatorio = pd.DataFrame.from_dict(resultado, orient='index')
    return df_relatorio

def normalize_dataset(df_train, df_test, scaler, drop_cols=["Id", "Label"]):
    """
    Normalização. Aplica um scaler ao dataset de treino e teste.

    -- Entradas:
        df_train: DataFrame de treino
        df_test: DataFrame de teste
        scaler: instancia de scaler do sklearn
        drop_cols (list): colunas para remover antes da normalização

    -- Saída:
        (df_train_normalized, df_test_normalized): DataFrames normalizados
    """

    # Ajustar scaler no treino
    scaler.fit(df_train)

    # Transformar dados
    X_train_normalized = scaler.transform(df_train)
    X_test_normalized = scaler.transform(df_test)

    # Retornar DataFrames
    return (
        pd.DataFrame(X_train_normalized, columns=df_train.columns),
        pd.DataFrame(X_test_normalized,  columns=df_test.columns)
    )

