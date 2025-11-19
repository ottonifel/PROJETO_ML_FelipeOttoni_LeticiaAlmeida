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


# IBI: Cacular média, STD (desvio padrão) e RMSSD(FORMULA ESTRANHA), Q_1/Q_3 (Quartis) 25 e 75 quartil
# ACC: Média de Magnitude (MÉDIA(raiz(X^2 + Y^2 + Z^2))), STD em [X, Y e Z], media de cada um (?)
# HR: Média, STD (desvio pad), Média de Amplitude (Picos de BVP - Vales de BVP)
# TEMP: Média, Range (MAX - MIN)
# EDA: Média, STD(desvio padrão), SCR Rate (Número de Picos/Duração da Sessão)
# BVP: Média, STD (desvio pad), Média de Amplitude (Picos de BVP - Vales de BVP)

def analise_ACC():
    pass

def analise_BVP():
    pass

def analise_EDA():
    pass

def analise_HR():
    pass

def analise_IBI():
    pass

def analise_TEMP():
    pass

'''
# Para cada sensor
    for sensor in sensors:
        file_path = os.path.join(user_folder, f"{sensor}.csv")
        
        # Ler o CSV
        df_sensor = pd.read_csv(file_path)

        # Analise exploratória do csv do sensor e modificações e limpeza
        
        
        #criação de uma nova coluna sumarizada para cada coluna do sensor
        # print usuario:
        print(user_id)

        # Para cada coluna do sensor, calcular média
        for col in df_sensor.columns:
            metric_name = f"{sensor}_{col}_mean"
            user_data[metric_name] = df_sensor[col].mean()
        
        
        
    
    # Adiciona as métricas desse usuário
    summary_rows.append(user_data)

# DataFrame final com métricas
df_metrics = pd.DataFrame(summary_rows)

# Fazer merge com df_users_info
#df_final = df_users_info.merge(df_metrics, on="Id", how="left")'''



# Tags: para cada medida descobrir a faixa dos valores dos sensores que foram utilizados no experimento controlado de acordo com o tempo em tags, nas tabelas dos sensores e nas frequência de cada sensor


# 1. limpeza users_info (exclusão e substituição de nulos, etc)
# 2. sumarização de cada métrica (oq fazer com yes***, yes****) (outros tratamentos?)
# 3. junção de users_info com a sumarização
# 4. exclusão de outliers em cada coluna
# 5. analise exploratoria de ditribuição, covariancia etc. coluna por coluna para saber se elas interferem no rotulo (exclusão de coluna que não interferem)
# 6. normalizar colunas (tranformações simbolica-numerica, numerica-numerica, etc)
# 7... Aplicação nos modelos e testes/justificativas

