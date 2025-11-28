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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def find_dataset_model_config(resultados):
    """
    Escolhe o melhor modelo usando:
    1) F1-score macro
    2) Acurácia
    """
    
    melhor = None
    melhor_key = None

    for k, r in resultados.items():
        if melhor is None:
            melhor = r
            melhor_key = k
            continue

        cond1 = r["F1-score"] > melhor["F1-score"]
        cond2 = r["F1-score"] == melhor["F1-score"] and r["Acurácia"] > melhor["Acurácia"]

        if cond1 or cond2:
            melhor = r
            melhor_key = k

    print("\n" + "="*70)
    print("MELHOR MODELO (Critério: F1 > Acurácia)")
    print("="*70)
    
    print(f"Melhor Combinação: {melhor_key}")
    print(f"Acurácia: {melhor['Acurácia']:.4f}")
    print(f"F1-score: {melhor['F1-score']:.4f}")
    print(f"Desvio Acc: {melhor['Desvio Acc']:.4f}")
    print(f"Desvio F1: {melhor['Desvio F1']:.4f}")
    print("Hiperparâmetros:", melhor["Hiperparâmetros"])
    print("="*70)


def avaliar_modelo_grid_search(modelo, nome_modelo, X_treino, y_treinamento, k_values, knn_params, nome_tecnica):

    resultados = {}

    # tipo de modelo
    is_knn = isinstance(modelo, KNeighborsClassifier)
    is_logreg = isinstance(modelo, LogisticRegression)
    is_svm = isinstance(modelo, SVC)
    is_mlp = isinstance(modelo, MLPClassifier)
    is_rf = isinstance(modelo, RandomForestClassifier)

    # Grid Search
    if is_knn:
        hyperparams_list = [{"n_neighbors": k} for k in knn_params]

    elif is_logreg:
        hyperparams_list = [{"C": c} for c in [0.1, 1, 10]]

    elif is_svm:
        if modelo.kernel == "linear":
            hyperparams_list = [{"C": c} for c in [0.1, 1, 10]]
        elif modelo.kernel == "rbf":
            hyperparams_list = [
                {"C": 1, "gamma": 0.01},
                {"C": 1, "gamma": 0.1},
                {"C": 10, "gamma": 0.01},
            ]
        else:
            hyperparams_list = [{}]

    elif is_mlp:
        hyperparams_list = [
            {"hidden_layer_sizes": (10,)},
            {"hidden_layer_sizes": (20, 10)},
            {"hidden_layer_sizes": (50,)},
        ]

    elif is_rf:
        hyperparams_list = [
            {"n_estimators": 50, "max_depth": 4},
            {"n_estimators": 150, "max_depth": 10},
        ]

    else:
        hyperparams_list = [{}]

    # Métricas
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro")
    }

    print(f"\n{'='*50}\nBusca para: {nome_modelo} | Técnica: {nome_tecnica}\n{'='*50}")

    # Loop kfold
    for k_fold in k_values:
        cv_k = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)

        # loop hiperparametro
        for hyperparams in hyperparams_list:

            model_clone = clone(modelo)
            if hyperparams:
                model_clone.set_params(**hyperparams)

            scores = cross_validate(
                estimator=model_clone,
                X=X_treino,
                y=y_treinamento,
                cv=cv_k,
                scoring=scorers,
                n_jobs=-1
            )

            # métricas
            media_acc = scores["test_accuracy"].mean()
            media_f1 = scores["test_f1_macro"].mean()

            # formatação chave
            chave = f"{nome_tecnica}_K{k_fold}_{hyperparams}"

            resultados[chave] = {
                "Acurácia": media_acc,
                "F1-score": media_f1,
                "Desvio Acc": scores["test_accuracy"].std(),
                "Desvio F1": scores["test_f1_macro"].std(),
                "Hiperparâmetros": hyperparams
            }

            print(f"{nome_modelo} | {nome_tecnica} | K={k_fold} | Params={hyperparams} "
                  f"=> Acc={media_acc:.4f}, F1={media_f1:.4f}")

    return resultados


def comparacao_em_grade_modelos(modelos, conjuntos_treino, y_treinamento ,k_values, knn_params):
    resultados = {}

    # itera sobre normalização
    for nome_tecnica, X_treino in conjuntos_treino.items():

        # itera sobre modelo
        for nome_modelo, modelo in modelos.items():

            resultados_modelo = avaliar_modelo_grid_search(
                modelo=modelo, 
                nome_modelo=nome_modelo, 
                X_treino=X_treino, 
                y_treinamento=y_treinamento,
                k_values=k_values,
                knn_params=knn_params,
                nome_tecnica=nome_tecnica
            )

            for chave_combinacao, resultado in resultados_modelo.items():
                chave_final = f"{nome_modelo}_{chave_combinacao}"
                resultados[chave_final] = resultado
    # mostra o melhor
    find_dataset_model_config(resultados)
    return (resultados)

def avaliar_dummy_baseline(X, y, k_fold=5):
    """
    Avalia DummyClassifier.
    
    -- Entradas:
        X: Features de treinamento.
        y: Labels.
        k_fold: Quantidade de folds para validação cruzada.
    
    -- Saída:
    resultados: Dicionário contendo média e desvio padrão da acurácia por estratégia.
    """
    strategies = ["most_frequent", "stratified"]

    resultados = {}

    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)

    for strategy in strategies:
        # Modelo dummy
        dummy = DummyClassifier(strategy=strategy, random_state=42)

        # Validação cruzada
        scores = cross_val_score(dummy, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        resultados[strategy] = {
            "media": np.mean(scores),
            "std": np.std(scores),
            "scores": scores
        }

        print(f"[Dummy: {strategy.replace('_', ' ')}]  Acurácia Média = {np.mean(scores):.4f}  (+/- {np.std(scores):.4f})")

    return resultados
