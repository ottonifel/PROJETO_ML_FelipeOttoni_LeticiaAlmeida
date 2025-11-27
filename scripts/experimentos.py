'''# ################################################################
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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any, List

def avaliar_modelo_grid_search( modelo, nome_modelo, conjuntos_treino, y_treinamento, k_values):
    """
    Realiza o Grid Search (Normalização x K-Fold) para um modelo específico.

    -- Entradas:
        modelo: A instância do modelo a ser avaliado
        nome_modelo: O nome do modelo para print
        conjuntos_treino: Dicionário {nome_tecnica: X_treino}
        y_treinamento: O vetor de rótulos
        k_values: Lista de valores de K a serem testados

    -- Saída:
        Um dicionário de resultados {combinacao: {'Média de Acurácia', 'Desvio Padrão'}}.
    """

    resultados_modelo = {}
    print(f"\n{'='*50}\nBusca em Grade para: {nome_modelo}\n{'='*50}")

    # Itera pelas técnicas de Normalização
    for nome_tecnica, X_treino in conjuntos_treino.items():
        
        # Itera pelos valores de K (Validação Cruzada)
        for k in k_values:
            # Define a Validação Cruzada Estratificada (essencial para balanceamento)
            cv_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            
            # Executa a Validação Cruzada K vezes
            scores = cross_val_score(modelo, X_treino, y_treinamento, cv=cv_k, scoring='accuracy')
            
            # Calcula e armazena os resultados
            chave = f"{nome_tecnica}_K{k}"
            media = np.mean(scores)
            std = np.std(scores)
            
            resultados_modelo[chave] = {
                'Média de Acurácia': media,
                'Desvio Padrão': std
            }
            
            print(f"-> Combinação {nome_modelo} | {chave}: Média de Acurácia = {media:.4f} (+/- {std:.4f})")
    
    # 3. Seleção da Melhor Combinação (Melhor Normalização e Melhor K)
    melhor_combinacao = max(resultados_modelo, key=lambda k: resultados_modelo[k]['Média de Acurácia'])
    melhor_resultado = resultados_modelo[melhor_combinacao]

    print("\n-------------------------------------------------")
    print(f"Melhor combinação selecionada para {nome_modelo}: {melhor_combinacao}")
    print(f"Média de Acurácia: {melhor_resultado['Média de Acurácia']:.4f}")
    print(f"Desvio Padrão: {melhor_resultado['Desvio Padrão']:.4f}")
    print("-------------------------------------------------")

    return resultados_modelo

def find_dataset_model_config(resultados_globais):
    """
    Encontra e imprime a melhor combinação entre todos os modelos e configurações.
    
    -- Entradas:

    -- Saídas:

    """
    # Encontra a melhor combinação em todos os resultados
    melhor_combinacao = max(
        resultados_globais, 
        key=lambda k: resultados_globais[k]['Média de Acurácia']
    )
    melhor_resultado = resultados_globais[melhor_combinacao]

    # Imprime o resultado final
    print("\n" + "="*70)
    print("RESUMO GLOBAL: MELHOR COMBINAÇÃO ENCONTRADA GERAL")
    print("="*70)
    print(f"Melhor Combinação: {melhor_combinacao}")
    print(f"Média de Acurácia: {melhor_resultado['Média de Acurácia']:.4f}")
    print(f"Desvio Padrão: {melhor_resultado['Desvio Padrão']:.4f}")
    print("="*70)

def comparacao_em_grade_modelos(modelos, conjuntos_treino, y_treinamento):
    """
    Define os modelos e executa a comparação em grade para todos eles.
    -- Entradas:

    -- Saídas:


    """

    # 1. DEFINIÇÃO DOS VALORES DE K
    # K=[3, 4, 5] é recomendado para o seu N=46.
    k_values = [3, 4, 5] 
    
    # Dicionário para armazenar todos os resultados (de todos os modelos)
    todos_resultados = {}

    # 3. EXECUÇÃO DO GRID SEARCH PARA CADA MODELO
    for nome_modelo, modelo in modelos.items():
        resultados_modelo = avaliar_modelo_grid_search(modelo, nome_modelo, conjuntos_treino, y_treinamento, k_values)
        # Combina os resultados do modelo atual com os resultados globais
        for chave_combinacao, resultado in resultados_modelo.items():
            # A chave final inclui o nome do modelo para desambiguação
            chave_final = f"{nome_modelo}_{chave_combinacao}"
            todos_resultados[chave_final] = resultado
            
    # 4. ANÁLISE E IMPRESSÃO DO MELHOR RESULTADO GLOBAL
    find_dataset_model_config(todos_resultados)'''

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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder # Mantido para referência, mas não usado nas funções de avaliação
from sklearn.base import clone # Importante para clonar o modelo e setar novos hiperparâmetros
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any, List

# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------

def find_dataset_model_config(todos_resultados: Dict[str, Dict[str, Any]]):
    """
    Encontra e imprime a melhor combinação entre todos os modelos e configurações.
    """
    # Encontra a melhor combinação em todos os resultados
    # Nota: A chave de busca foi alterada para 'Média de Acurácia' conforme suas funções originais
    melhor_combinacao = max(
        todos_resultados, 
        key=lambda k: todos_resultados[k]['Média de Acurácia']
    )
    melhor_resultado = todos_resultados[melhor_combinacao]

    # Imprime o resultado final
    print("\n" + "="*70)
    print("RESUMO GLOBAL: MELHOR COMBINAÇÃO ENCONTRADA GERAL")
    print("="*70)
    print(f"Melhor Combinação: {melhor_combinacao}")
    print(f"Média de Acurácia: {melhor_resultado['Média de Acurácia']:.4f}")
    
    # Adiciona a impressão de hiperparâmetros se existirem
    if 'Hiperparâmetros' in melhor_resultado and melhor_resultado['Hiperparâmetros']:
        print(f"Hiperparâmetros: {melhor_resultado['Hiperparâmetros']}")
        
    print(f"Desvio Padrão: {melhor_resultado['Desvio Padrão']:.4f}")
    print("="*70)


def avaliar_modelo_grid_search(
    modelo: Any, 
    nome_modelo: str, 
    X_treino: np.ndarray, 
    y_treinamento: np.ndarray, 
    k_values: List[int], 
    knn_params: List[int] = None, 
    nome_tecnica: str = "Default"
) -> Dict[str, Dict[str, Any]]:
    """
    Avalia um modelo em todos os valores de K-Fold e, se for KNN, 
    em todos os hiperparâmetros 'n_neighbors' fornecidos. (Lógica de aninhamento)
    """
    todos_resultados = {}
    
    # 1. Checa se é o modelo KNN para ativar a busca por n_neighbors
    is_knn = isinstance(modelo, KNeighborsClassifier)
    
    # Para modelos não-KNN, itera apenas uma vez com o parâmetro original
    n_neighbors_values = knn_params if is_knn else [None]

    print(f"\n{'='*50}\nBusca em Grade para: {nome_modelo} (Técnica: {nome_tecnica})\n{'='*50}")

    # 2. Loop sobre os valores de K-Fold
    for k_fold in k_values:
        cv_k = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        # 3. Loop sobre os hiperparâmetros (n_neighbors) - Executa 1x para modelos não-KNN
        for n_neighbors in n_neighbors_values:
            
            # Clonar o modelo é crucial para garantir que cada teste é independente
            current_model = clone(modelo)
            current_hyperparams = {}
            
            # Configura o modelo e salva o hiperparâmetro se for KNN
            if is_knn and n_neighbors is not None:
                current_model.set_params(n_neighbors=n_neighbors)
                current_hyperparams = {'n_neighbors': n_neighbors}
            
            # Executa a Validação Cruzada K vezes
            try:
                scores = cross_val_score(current_model, X_treino, y_treinamento, cv=cv_k, scoring='accuracy', n_jobs=-1)
                media = np.mean(scores)
                std = np.std(scores)
            except Exception as e:
                print(f"Falha na validação cruzada para {nome_modelo} (K-Fold={k_fold}, Hiperparâmetros={current_hyperparams}): {e}")
                media = 0.0
                std = 0.0
                
            # Armazena os resultados, ajustando a chave para incluir o hiperparâmetro do KNN
            chave = f"{nome_tecnica}_KFold{k_fold}"
            if is_knn and n_neighbors is not None:
                chave += f"_KNN_k{n_neighbors}"
                
            # A chave final é apenas a combinação (Técnica_KFold_Hiperparam)
            todos_resultados[chave] = {
                'Média de Acurácia': media,
                'Desvio Padrão': std,
                'Hiperparâmetros': current_hyperparams
            }
            
            # Impressão do resultado
            param_str = f" (k={n_neighbors})" if is_knn and n_neighbors is not None else ""
            print(f"-> Combinação {nome_modelo} | {nome_tecnica} | K-Fold={k_fold}{param_str}: Média de Acurácia = {media:.4f} (+/- {std:.4f})")
            
    # Seleção da Melhor Combinação (Melhor Normalização e Melhor K)
    if todos_resultados:
        melhor_combinacao = max(todos_resultados, key=lambda k: todos_resultados[k]['Média de Acurácia'])
        melhor_resultado = todos_resultados[melhor_combinacao]

        print("\n-------------------------------------------------")
        print(f"Melhor combinação selecionada para {nome_modelo} ({nome_tecnica}): {melhor_combinacao}")
        print(f"Média de Acurácia: {melhor_resultado['Média de Acurácia']:.4f}")
        print(f"Desvio Padrão: {melhor_resultado['Desvio Padrão']:.4f}")
        print("-------------------------------------------------")


    return todos_resultados

# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ----------------------------------------------------------------------

def comparacao_em_grade_modelos(
    modelos: Dict[str, Any], 
    conjuntos_treino: Dict[str, np.ndarray], 
    y_treinamento: np.ndarray,
    k_values: List[int],
    knn_params: List[int]
):
    """
    Executa a comparação em grade para o dicionário de modelos, 
    tunando o hiperparâmetro 'n_neighbors' (K do KNN) junto com 
    as técnicas de pré-processamento e os K-Folds.
    """

    # Dicionário para armazenar todos os resultados
    todos_resultados = {}

    # Loop 1: Itera sobre as técnicas de pré-processamento (e.g., Normalização, Padronização)
    for nome_tecnica, X_treino in conjuntos_treino.items():
        
        # Loop 2: Itera sobre os modelos
        for nome_modelo, modelo in modelos.items():
            
            # O avaliar_modelo_grid_search agora recebe e utiliza os knn_params se for um KNN
            resultados_modelo = avaliar_modelo_grid_search(
                modelo=modelo, 
                nome_modelo=nome_modelo, 
                X_treino=X_treino, 
                y_treinamento=y_treinamento, 
                k_values=k_values, 
                knn_params=knn_params, # Passa os valores de n_neighbors
                nome_tecnica=nome_tecnica
            )
            
            # Combina os resultados do modelo atual com os resultados globais
            for chave_combinacao, resultado in resultados_modelo.items():
                # A chave final inclui o nome do modelo para desambiguação
                chave_final = f"{nome_modelo}_{chave_combinacao}"
                todos_resultados[chave_final] = resultado
            
    # 4. ANÁLISE E IMPRESSÃO DO MELHOR RESULTADO GLOBAL
    find_dataset_model_config(todos_resultados)