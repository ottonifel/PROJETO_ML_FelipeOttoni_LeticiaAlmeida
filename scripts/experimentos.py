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
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any, List

def find_dataset_model_config(resultados):
    """
    Encontra e imprime a melhor combinação entre todos os modelos e configurações.
    """
    # Encontra a melhor combinação em todos os resultados
    melhor_combinacao = max(resultados, key=lambda k: resultados[k]['Média de Acurácia'])
    melhor_resultado = resultados[melhor_combinacao]

    # Imprime o resultado
    print("\n" + "="*70)
    print("RESUMO GLOBAL: MELHOR COMBINAÇÃO ENCONTRADA GERAL")
    print("="*70)
    print(f"Melhor Combinação: {melhor_combinacao}")
    print(f"Média de Acurácia: {melhor_resultado['Média de Acurácia']:.4f}")
    
    # Imprime os hiperparâmetros, se existirem
    if 'Hiperparâmetros' in melhor_resultado and melhor_resultado['Hiperparâmetros']:
        print(f"Hiperparâmetros: {melhor_resultado['Hiperparâmetros']}")
        
    print(f"Desvio Padrão: {melhor_resultado['Desvio Padrão']:.4f}")
    print("="*70)


def avaliar_modelo_grid_search(modelo, nome_modelo, X_treino, y_treinamento, k_values, knn_params, nome_tecnica):
    """
    Avalia um modelo em todos os valores de K-Fold e, se for KNN, 
    em todos os hiperparâmetros 'n_neighbors' fornecidos
    """
    resultados = {}
    
    # verifica se o modelo é KNN
    is_knn = isinstance(modelo, KNeighborsClassifier)
    
    # Para modelos não-KNN executa apenas uma vez o Loop
    n_neighbors_values = knn_params if is_knn else [None]

    print(f"\n{'='*50}\nBusca em Grade para: {nome_modelo} (Técnica: {nome_tecnica})\n{'='*50}")

    # Loop sobre os valores de K-Fold
    for k_fold in k_values:
        cv_k = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        # 3. Loop sobre os hiperparâmetros (n_neighbors) - Executa 1x para modelos não-KNN
        for n_neighbors in n_neighbors_values:
            
            # Clona o modelo
            current_model = clone(modelo)
            current_hyperparams = {}
            
            # se for KNN configura o modelo
            if is_knn and n_neighbors is not None:
                current_model.set_params(n_neighbors=n_neighbors)
                current_hyperparams = {'n_neighbors': n_neighbors}
            
            # Executa a Validação Cruzada K vezes
            scores = cross_val_score(current_model, X_treino, y_treinamento, cv=cv_k, scoring='accuracy', n_jobs=-1)
            media = np.mean(scores)
            std = np.std(scores)
                
            # Armazena os resultados
            chave = f"{nome_tecnica}_KFold{k_fold}"
            # se for KNN ajusta
            if is_knn and n_neighbors is not None:
                chave += f"_KNN_k{n_neighbors}"
                
            # formato: (Técnica_KFold_Hiperparam)
            resultados[chave] = {
                'Média de Acurácia': media,
                'Desvio Padrão': std,
                'Hiperparâmetros': current_hyperparams
            }
            
            # Print resultado
            param_str = f" (k={n_neighbors})" if is_knn and n_neighbors is not None else ""
            print(f"-> Combinação {nome_modelo} | {nome_tecnica} | K-Fold={k_fold}{param_str}: Média de Acurácia = {media:.4f} (+/- {std:.4f})")
            
    # Imprime a melhor combinação (Melhor Normalização e Melhor K folds)
    if resultados:
        melhor_combinacao = max(resultados, key=lambda k: resultados[k]['Média de Acurácia'])
        melhor_resultado = resultados[melhor_combinacao]

        print("\n-------------------------------------------------")
        print(f"Melhor combinação para {nome_modelo} ({nome_tecnica}): {melhor_combinacao}")
        print(f"Média de Acurácia: {melhor_resultado['Média de Acurácia']:.4f}")
        print(f"Desvio Padrão: {melhor_resultado['Desvio Padrão']:.4f}")
        print("-------------------------------------------------")


    return resultados

def comparacao_em_grade_modelos(modelos, conjuntos_treino, y_treinamento ,k_values, knn_params):
    """
    Executa a comparação em grade para o dicionário de modelos, 
    ajustando o hiperparâmetro 'n_neighbors' (K do KNN) junto com as técnicas de pré-processamento e os K-Folds.
    """

    # Dicionário para armazenar todos os resultados
    resultados = {}

    # Loop 1: Itera sobre as técnicas de pré-processamento ( tipos de Normalização)
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
                knn_params=knn_params,
                nome_tecnica=nome_tecnica
            )
            
            # Combina os resultados do modelo atual com os dicionario de resultados globais
            for chave_combinacao, resultado in resultados_modelo.items():
                chave_final = f"{nome_modelo}_{chave_combinacao}"
                resultados[chave_final] = resultado
            
    # Resultado final
    find_dataset_model_config(resultados)