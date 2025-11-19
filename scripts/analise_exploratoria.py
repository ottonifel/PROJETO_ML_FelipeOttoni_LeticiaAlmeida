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

def read_users_info(path='dataset/users_info.txt'):
    """
    Lê o arquivo users_info.txt e retorna
    um DataFrame com as informações dos usuários.
    """

    if not os.path.exists(path):
        print(f"Erro: arquivo não encontrado -> {path}")
        return None

    try:
        lines = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith(',,,,'):  # linha que marca início dos comentários
                    break
                lines.append(line)

        # Criar DataFrame apenas com as linhas úteis
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
    event_cols = ['Stress Inducement', 'Aerobic Exercise', 'Anaerobic Exercise']
    # Remover valores nulos nessas colunas
    df_clean = df_clean.dropna(subset=event_cols)
    # Remover valores "Yes*" ou "Yes**"
    df_clean = df_clean[
        ~df_clean[event_cols].isin(['Yes*', 'Yes**']).any(axis=1)
    ]

    # substituir campos nulos por média do gênero
    numeric_cols = ['Age', 'Height (cm)', 'Weight (kg)']

    for col in numeric_cols:
        # média agrupada por gênero para cada linha
        mean_by_gender = df_clean.groupby('Gender')[col].transform('mean')

        # substitui valores nulos pela respectiva média
        df_clean[col] = df_clean[col].fillna(mean_by_gender)

    return df_clean


def read_sensor_csv(user_folder, filename):
    """ 
    Lê um CSV específico do usuário e retorna como DataFrame, extraindo seu timestamp de início e a frequência de sua métrica
    """

    path = os.path.join(user_folder, filename)

    if not os.path.exists(path):
        print(f"Aviso: arquivo {filename} não encontrado para {user_folder}")
        return None

    try:
        if filename.upper() == "IBI.csv".upper():
            # colocar aqui o processo diferente para o IBI
            # Por enquanto, apenas retornamos um placeholder
            print("Processando arquivo IBI.csv com lógica especial...")
            return
        else:
            df = pd.read_csv(path)
            start_time = pd.to_datetime(df.iloc[0, 0], errors='coerce')
            freq = float(df.iloc[1, 0])
            df_metrics = df.iloc[2:].reset_index(drop=True)  # astype(float)
            return df_metrics, start_time, freq
    
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None
    

def add_timestamp_column(df, start_time, freq):
    """
    Adiciona uma coluna de timestamps ao dataframe de métricas.
    """
    # número de linhas do dataframe
    n = len(df)

    # intervalo entre amostras em segundos
    dt = 1.0 / freq

    # cria vetor de deltas (0, dt, 2dt, 3dt, ...)
    time_offsets = np.arange(n) * dt

    # gera timestamps reais
    timestamps = start_time + pd.to_timedelta(time_offsets, unit='s')

    # adiciona ao DF original
    df_with_ts = df.copy()
    df_with_ts["timestamp"] = timestamps

    return df_with_ts
    
def sumarizar_ACC(user_folder):
    # chama função para le o csv, frequencia e o start_time
    df_metrics, start_time, freq = read_sensor_csv(user_folder, "ACC.csv")

    # -- PULADO -- #
        # chama função que adiciona coluna timestamp de cada amostra (parametros: df, frequencia, start_time)
            ####   df_metrics = add_timestamp_column(df_metrics, start_time, freq)
        # chama função que adiciona coluna "intervalo" categorica: (1, 2 ,3...) de acordo com a coluna timestamp de cada amostra estando nos intervalos de tags, depois exclui a coluna timestamp (juntar isso com a função anterior?) colunas finais até essa etapa: metricas + Intervalo
        # (lista de intervalos timestamps de tags já feita separadamente e passada para a função sumariza_XXX() como parametro)
     # -- PULADO -- #

    # calculos de métricas especificas para o arquivo XX.csv
    df_ACC = process_ACC(df_metrics)
    return df_ACC

def process_ACC(df):

    # Converter valores para float
    df = df.astype(float)

    # Separar colunas X, Y, Z
    x = df[:, 0]
    y = df[:, 1]
    z = df[:, 2]

    # STD de cada coluna
    std_x = np.std(x)
    std_y = np.std(y)
    std_z = np.std(z)

    # Magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    mean_magnitude = np.mean(magnitude)

    '''
    # Análise de tendência (início vs fim)
    first_half = np.mean(magnitude[:len(magnitude)//2])
    second_half = np.mean(magnitude[len(magnitude)//2:])

    tendency = "aumentou" if second_half > first_half else "não aumentou"
    '''

    result_df = pd.DataFrame({
        "STD_X_ACC": [std_x],
        "STD_Y_ACC": [std_y],
        "STD_Z_ACC": [std_z],
        "Mean_Magnitude_ACC": [mean_magnitude]
        # "Magnitude_First_Half_ACC": [first_half],
        # "Magnitude_Second_Half_ACC": [second_half],
        # "Tendency_ACC": [tendency]
    })

    return result_df

def sumarizar_IBI(user_folder):
    # chama função para le o csv, frequencia e o start_time
    df_metrics, start_time, freq = read_sensor_csv(user_folder, "IBI.csv")

    df_IBI = process_IBI(df_metrics)
    return df_IBI

def process_IBI():
    result_df = pd.DataFrame({
        
    })

    return result_df

def sumarizar_BVP(user_folder):
    # chama função para le o csv, frequencia e o start_time
    df_metrics, start_time, freq = read_sensor_csv(user_folder, "BVP.csv")
    # calculos de métricas especificas para o arquivo XX.csv
    df_BVP = process_BVP(df_metrics, freq)

    return df_BVP

def process_BVP(df, freq):
    # converter métricas para float
    data = df.astype(float)

    # Média e STD
    mean = data.mean()
    std = data.std()

    '''
    # detectar picos
    peaks, _ = find_peaks(data, distance=freq*0.5) # evita picos falsos 

    # detectar vales (invertendo o sinal)
    valleys, _ = find_peaks(-data, distance=freq*0.5)

    # amplitude média (pico - vale anterior)
    amplitudes = []
    v_idx = 0
    for p in peaks:
        # pegar vale mais próximo antes do pico
        while v_idx < len(valleys)-1 and valleys[v_idx+1] < p:
            v_idx += 1
        if valleys[v_idx] < p:
            amplitude = data.iloc[p] - data.iloc[valleys[v_idx]]
            amplitudes.append(amplitude)
    '''

    result_df = pd.DataFrame({
        "Mean_BVP": [mean],
        "STD_BVP": [std],
        # "Num_Peaks_BVP": [len(peaks)],
        # "Num_Valleys_BVP": [len(valleys)],
        # "Mean_Amplitude_BVP": [np.mean(amplitudes) if amplitudes else np.nan]
    })

    return result_df

def sumarizar_EDA(user_folder):
    # chama função para le o csv, frequencia e o start_time
    df_metrics, start_time, freq = read_sensor_csv(user_folder, "EDA.csv")

    df_EDA = process_EDA(df_metrics)
    return df_EDA

def process_EDA(df):

    # converão para float
    data = df.astype(float)

    # Métricas principais
    mean = data.mean()
    std = data.std()
    range = data.max() - data.min()

    '''
    # Detecção de picos SCR (respostas rápidas)
    peaks, _ = find_peaks(data, prominence=0.05)

    # Amplitude dos picos (diferença entre pico e vizinhança)
    if len(peaks) > 0:
        amplitudes = data.iloc[peaks].values
        mean_peak_amplitude = amplitudes.mean()
    else:
        mean_peak_amplitude = 0.0
    '''

    # Retornar DataFrame
    result_df = pd.DataFrame({
        "Mean_EDA": [mean],
        "STD_EDA": [std],
        "Range_EDA": [range],
        # "SCR_Count_EDA": [len(peaks)],
        # "SCR_Mean_Amplitude_EDA": [mean_peak_amplitude]
    })
    
    return result_df

def sumarizar_HR(user_folder):
    # chama função para le o csv, frequencia e o start_time
    df_metrics, start_time, freq = read_sensor_csv(user_folder, "HR.csv")

    # processamento
    df_HR = process_HR(df_metrics)

    return df_HR

def process_HR(df):
    # converter métricas para float
    data = df.astype(float)

    # Média e STD
    mean = data.mean()
    std = data.std()

    '''
    # Encontrar picos
    peaks, _ = find_peaks(data)
    valleys, _ = find_peaks(-data)

    # Garantir mesmo tamanho para pareamento (caso necessário)
    n = min(len(peaks), len(valleys))
    if n > 0:
        amplitudes = data.iloc[peaks[:n]].values - data.iloc[valleys[:n]].values
        mean_amplitude = amplitudes.mean()
    else:
        mean_amplitude = np.nan  # caso não haja picos/vales suficientes
    '''

    # Criar DataFrame de saída
    result_df = pd.DataFrame({
        "Mean_HR": [mean],
        "STD_HR": [std],
        # "Mean_Amplitude_HR": [mean_amplitude],
        # "Num_Peaks_HR": [len(peaks)],
        # "Num_Valleys_HR": [len(valleys)]
    })

    return result_df

def sumarizar_TEMP(user_folder):
    # chama função para le o csv, frequencia e o start_time
    df_metrics, start_time, freq = read_sensor_csv(user_folder, "TEMP.csv")
    # calculos de métricas especificas para o arquivo XX.csv
    df_TEMP = process_TEMP(df_metrics)

    return df_TEMP

def process_TEMP(df):
    # converter métricas para float
    data = df.astype(float)

    # Média e Range
    mean = data.mean()
    range = data.max() - data.min()

    # Tendência (início x fim)
    half = len(data) // 2
    mean_first = data.iloc[:half].mean()
    mean_second = data.iloc[half:].mean()
    tendency = "aumentou" if mean_second > mean_first else "não aumentou"

    # Criar DataFrame de saída
    result_df = pd.DataFrame({
        "Mean_TEMP": [mean],
        "Range_TEMP": [range],
        "Tendency_TEMP": [tendency]
    })

    return result_df






# IBI: Cacular média, STD (desvio padrão) e RMSSD(FORMULA ESTRANHA), Q_1/Q_3 (Quartis) 25 e 75 quartil
# ACC: Média de Magnitude (MÉDIA(raiz(X^2 + Y^2 + Z^2))), STD em [X, Y e Z]
# HR: Média, STD (desvio pad), Média de Amplitude (Picos de BVP - Vales de BVP)
# TEMP: Média, Range (MAX - MIN)
# EDA: Média, STD(desvio padrão), SCR Rate (Número de Picos/Duração da Sessão)
# BVP: Média, STD (desvio pad), Média de Amplitude (Picos de BVP - Vales de BVP)


# Tags: para cada medida descobrir a faixa dos valores dos sensores que foram utilizados no experimento controlado de acordo com o tempo em tags, nas tabelas dos sensores e nas frequência de cada sensor


# 1. limpeza users_info (exclusão e substituição de nulos, etc)
# 2. sumarização de cada métrica (oq fazer com yes***, yes****) (outros tratamentos?)
# 3. junção de users_info com a sumarização
# 4. exclusão de outliers em cada coluna
# 5. analise exploratoria de ditribuição, covariancia etc. coluna por coluna para saber se elas interferem no rotulo (exclusão de coluna que não interferem)
# 6. normalizar colunas (tranformações simbolica-numerica, numerica-numerica, etc)
# 7... Aplicação nos modelos e testes/justificativas

