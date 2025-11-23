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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

def read_users_info(path='dataset/users_info.txt'):
    """
    Lê o arquivo users_info.txt e retorna
    um DataFrame com as informações dos usuários.
    """

    if not os.path.exists(path):
        print(f"Erro: arquivo não encontrado -> {path}")
        return None

    try:
        # Retira linhas de comentários
        lines = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith(',,,,'):  # indica o ínicio do comentário
                    break
                lines.append(line)

        # Cria DataFrame apenas com as linhas úteis
        from io import StringIO
        df = pd.read_csv(
            StringIO(''.join(lines)),
            sep=',',
            na_values='-'
        )

        return df
    
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None
    
def clean_users_info(df):
    """
    Limpa o DataFrame de informações dos usuários removendo casos inválidos/nulos
    e preenchendo valores nulos
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


def read_sensor_csv(user_folder, filename):
    """ 
    Lê um CSV específico do usuário e retorna como DataFrame, extraindo seu timestamp de início e a frequência de sua métrica

    -- Entradas:
        user_folder: nome da pasta = id do usuário
        filename: nome do CSV a ser lido (e.x: ACC.csv, HR.csv, etc.)
    -- Saídas:
        df_metrics: DataFrame com os valores das métricas
        start_time: timestamp de inicio da coleta da métrica pelos sensores
        freq: frequência da métrica
    """

    # Caminho para o csv
    path = os.path.join(user_folder, filename)

    if not os.path.exists(path):
        print(f"Aviso: arquivo {filename} não encontrado para {user_folder}")
        return None

    # Leitura do csv e divisão dele de acordo com as saídas
    try:
        # Caso IBI
        if filename.upper() == "IBI.csv".upper():
            df = pd.read_csv(path, header=None)
            start_time = pd.to_datetime(df.iloc[0, 0], errors='coerce') #linha 0 = timestamp
            freq = None
            df_metrics = df.iloc[1:].reset_index(drop=True)
            df_metrics.columns = ["t_rel", "IBI"]
            return df_metrics, start_time, freq
        else:
            df = pd.read_csv(path)
            start_time = pd.to_datetime(df.iloc[0, 0], errors='coerce')
            freq = float(df.iloc[1, 0])
            third_line = df.iloc[2, 0]

            if filename.upper() == "EDA.CSV" and float(third_line) == 0.0:
                # Pular linha de 1 até 3
                df_metrics = df.iloc[3:].reset_index(drop=True)
            else:
                # Pular linha 1 e 2
                df_metrics = df.iloc[2:].reset_index(drop=True)
            return df_metrics, start_time, freq
    
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None
    

def add_timestamp_column(df, start_time, freq):
    """
    Adiciona uma coluna de timestamps ao dataframe de métricas.

     -- Entradas:
        df: DataFrame com as métricas coletadas pelos sensores (apenas valor das métricas)
        start_time: timestamp de inicio da coleta da métrica pelos sensores
        freq: frequência da métrica
    -- Saídas:
        df_with_ts: df de entrada com a adição de uma coluna timestamp para cada amostra de métrica coletada

    """

    n = len(df) # número de linhas do dataframe

    # Intervalo de tempo entre duas amostras consecutivas (em segundos)
    dt = 1.0 / freq

    # Vetor com os tempos acumulados para cada amostra (instante relativo de cada linha em relação ao início da coleta)
    # [0*dt, 1*dt, 2*dt, 3*dt, ...]
    relative_times = np.arange(n) * dt

    # Cria timestamps reais de cada amostra
    timestamps = start_time + pd.to_timedelta(relative_times, unit='s')

    # Criação do DataFrame com timestamp
    df_with_ts = df.copy()
    df_with_ts["timestamp"] = timestamps

    return df_with_ts

def sumarize_sensor(user_folder, filename, process_function, add_ts=True):
    """
    Sumariza todos os valores coletados do sensor para sumarizar em features específicas para o sensor

    Entradas:
        user_folder: id do usuário = nome da pasta ond estão os CSVs do respectivo usuário
        filename: nome do arquivo CSV
        process_function: função de processar métricas/calcular features
        add_ts: boolean para indicar se deve haver adicao de timestamp (IBI não usa)

    Saída:
        DataFrame com features sumarizadas daquele sensor
    """

    # Leitura do csv e extrai o DataFrame, o start_time e a a frequencia
    df_metrics, start_time, freq = read_sensor_csv(user_folder, filename)

    # Se for um sensor baseado em série temporal, adiciona timestamp
    if add_ts:
        # Adição de uma coluna timestamp para cada amostra
        df_metrics = add_timestamp_column(df_metrics, start_time, freq)

    # Calculos de métricas/features especificas para o sensor
    df_output = process_function(df_metrics)

    return df_output

def process_ACC(df):
    '''
    Processa dados do acelerômetro (ACC), calcula magnitude, estatísticas básicas e o coeficiente angular (slope) da magnitude.

    --Entradas:
        df: DataFrame contendo colunas de aceleração (x, y, z) e timestamp.

    -- Saídas:
        result_df: DataFrame com as features extraídas: STD_X_ACC, STD_Y_ACC, STD_Z_ACC, Mean_Magnitude_ACC, coef_Magnitude_ACC
    '''

    df.columns = ["x", "y", "z", "timestamp"]

    # Converter valores para float
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)

    # Magnitude
    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    # STD
    std_x = df["x"].std()
    std_y = df["y"].std()
    std_z = df["z"].std()

    # mean magnitude
    mean_magnitude = df["magnitude"].mean()

    # Slope (coeficiente angular da magnitude)
    coef_magnitude = compute_slope(df, "magnitude")

    result_df = pd.DataFrame({
        "STD_X_ACC": [std_x],
        "STD_Y_ACC": [std_y],
        "STD_Z_ACC": [std_z],
        "Mean_Magnitude_ACC": [mean_magnitude],
        "coef_Magnitude_ACC": [coef_magnitude]
    })

    return result_df

def process_IBI():
    '''
    Processa dados do IBI (intervalo entre batimentos cardíacos), calculando estatísticas descritivas e variabilidade (RMSSD)
    --Entradas:
        df: DataFrame contendo a coluna "IBI"

    -- Saídas:
        result_df: DataFrame com as features extraídas: Mean_IBI, STD_IBI, Median_IBI, Q1_IBI, Q3_IBI, Range_IBI, RMSSD_IBI
    '''
    df = df.astype(float)

    ibi = df["IBI"].values

    # Mean, STD
    mean_ibi = np.mean(ibi)
    std_ibi = np.std(ibi)

    # Quartis
    q1 = np.percentile(ibi, 25)
    q3 = np.percentile(ibi, 75)

    # Mediana
    median_ibi = np.median(ibi)

    # Range
    range_ibi = np.max(ibi) - np.min(ibi)

    # RMSSD
    diff = np.diff(ibi)
    rmssd = np.sqrt(np.mean(diff**2))

    result_df = pd.DataFrame({
        "Mean_IBI": [mean_ibi],
        "STD_IBI": [std_ibi],
        "Median_IBI": [median_ibi],
        "Q1_IBI": [q1],
        "Q3_IBI": [q3],
        "Range_IBI": [range_ibi],
        "RMSSD_IBI": [rmssd]
    })

    return result_df

def process_BVP(df):
    '''
    Processa dados do BVP gerando estatísticas básicas e o coeficiente angular
    --Entradas:
        df: DataFrame contendo a coluna "BVP" e timestamp

    -- Saídas:
        result_df: DataFrame com as features extraídas: Mean_BVP, STD_BVP, Min_BVP, Max_BVP, Range_BVP, coef_BVP
    '''
    df.columns = ["BVP", "timestamp"]

    # converter métricas para float
    df["BVP"] = df["BVP"].astype(float)


    # Média, STD, min, max, range
    mean = df["BVP"].mean()
    std = df["BVP"].std()
    min_val = df["BVP"].min()
    max_val = df["BVP"].max()
    range_val = max_val - min_val

    # Slope BVP
    coef_bvp = compute_slope(df, "BVP")

    result_df = pd.DataFrame({
        "Mean_BVP": [mean],
        "STD_BVP": [std],
        "Min_BVP": [min_val],
        "Max_BVP": [max_val],
        "Range_BVP": [range_val],
        "coef_BVP": [coef_bvp]
    })

    return result_df

def process_EDA(df):
    '''
    Processa dados de EDA (atividade eletrodérmica) gerando estatísticas básicas e o coeficiente angular
    --Entradas:
        df: DataFrame contendo a coluna "EDA" e timestamp

    -- Saídas:
        result_df: DataFrame com as features extraídas: Mean_EDA, STD_EDA, Min_EDA, Max_EDA, Range_EDA, coef_EDA
    '''

    df.columns = ["EDA", "timestamp"]
    # converão para float
    df["EDA"] = df["EDA"].astype(float)
    

    # Média, STD, min, max, range
    mean = df["EDA"].mean()
    std = df["EDA"].std()
    min_val = df["EDA"].min()
    max_val = df["EDA"].max()
    range_val = max_val - min_val

    # Slope EDA
    coef_eda = compute_slope(df, "EDA")

    # Retornar DataFrame
    result_df = pd.DataFrame({
        "Mean_EDA": [mean],
        "STD_EDA": [std],
        "Min_EDA": [min_val],
        "Max_EDA": [max_val],
        "Range_EDA": [range_val],
        "coef_EDA": [coef_eda]
    })
    
    return result_df

def process_HR(df):
    '''
    Processa dados de HR (frequência cardíaca) gerando estatísticas básicas e o coeficiente angular
    --Entradas:
        df: DataFrame contendo a coluna "HR" e timestamp

    -- Saídas:
        result_df: DataFrame com as features extraídas: Mean_HR, STD_HR, Min_HR, Max_HR, Range_HR, coef_HR
    '''
    
    df.columns = ["HR", "timestamp"]
    # converter métricas para float
    df["HR"] = df["HR"].astype(float)
    

    # Média, STD, min, max, range
    mean = df["HR"].mean()
    std = df["HR"].std()
    min_val = df["HR"].min()
    max_val = df["HR"].max()
    range_val = max_val - min_val

    # Slope HR
    coef_hr = compute_slope(df, "HR")

    # Criar DataFrame de saída
    result_df = pd.DataFrame({
        "Mean_HR": [mean],
        "STD_HR": [std],
        "Min_HR": [min_val],
        "Max_HR": [max_val],
        "Range_HR": [range_val],
        "coef_HR": [coef_hr]
    })

    return result_df

def process_TEMP(df):
    '''
    Processa dados de HR (frequência cardíaca) gerando estatísticas básicas e o coeficiente angular
    --Entradas:
        df: DataFrame contendo a coluna "TEMP" e timestamp

    -- Saídas:
        result_df: DataFrame com as features extraídas: Mean_TEMP, Min_TEMP, Max_TEMP, Range_TEMP, coef_TEMP
    '''
    df.columns = ["TEMP", "timestamp"]
    # converter métricas para float
    df["TEMP"] = df["TEMP"].astype(float)


    # Média, min. max,  Range
    mean = df["TEMP"].mean()
    min_val = df["TEMP"].min()
    max_val = df["TEMP"].max()
    range_val = max_val - min_val

    # Slope TEMP
    coef_temp = compute_slope(df, "TEMP")

    # Criar DataFrame de saída
    result_df = pd.DataFrame({
        "Mean_TEMP": [mean],
        "Range_TEMP": [range],
        "Min_TEMP": [min_val],
        "Max_TEMP": [max_val],
        "Range_TEMP": [range_val],
        "coef_TEMP": [coef_temp]
    })

    return result_df

def compute_slope(df, column_name):
    """
    Calcula o slope (coeficiente angular) de uma coluna por regressão linear.
    
    -- Entradas:
        df : DataFrame contendo a coluna da métrica e a coluna 'timestamp'
        column_name : nome da coluna para a qual o slope será calculado
    
    -- Saídas:
        slope: coeficiente angular
    """

    # garantir que timestamp está presente
    if "timestamp" not in df.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'timestamp'")

    # extrair timestamps e valores
    timestamps = df["timestamp"].astype(float).values.reshape(-1, 1)
    values = df[column_name].astype(float).values.reshape(-1, 1)

    # regressão linear
    model = LinearRegression()
    model.fit(timestamps, values)

    # slope = coef angular
    slope = model.coef_[0][0]

    return slope

# IBI: Cacular média, STD (desvio padrão) e RMSSD(FORMULA ESTRANHA), Q_1/Q_3 (Quartis) 25 e 75 quartil
# ACC: Média de Magnitude (MÉDIA(raiz(X^2 + Y^2 + Z^2))), STD em [X, Y e Z]
# HR: Média, STD (desvio pad), Média de Amplitude (Picos de BVP - Vales de BVP)
# TEMP: Média, Range (MAX - MIN)
# EDA: Média, STD(desvio padrão), SCR Rate (Número de Picos/Duração da Sessão)
# BVP: Média, STD (desvio pad), Média de Amplitude (Picos de BVP - Vales de BVP)


# * IBI vazio, sem nada (U_89740), ou IBI corrompido (U_87186) ??
# * Tratar users_info "pratica atividade fisica?": moda ?

# 1. limpeza users_info (exclusão e substituição de nulos, etc)
# 2. sumarização de cada métrica (oq fazer com yes***, yes****) (outros tratamentos?)
# 3. junção de users_info com a sumarização
# 4. exclusão de outliers em cada coluna
# 5. analise exploratoria de ditribuição, covariancia etc. coluna por coluna para saber se elas interferem no rotulo (exclusão de coluna que não interferem)
# 6. normalizar colunas (tranformações simbolica-numerica, numerica-numerica, etc)
# 7... Aplicação nos modelos e testes/justificativas

