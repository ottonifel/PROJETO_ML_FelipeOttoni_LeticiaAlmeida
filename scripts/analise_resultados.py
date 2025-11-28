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

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve_error(
    estimator, 
    X: np.ndarray, 
    y: np.ndarray, 
    cv: int = 5, 
    n_jobs: int = -1, 
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5)
):
    """
    Gera uma Curva de Aprendizado onde o Eixo Y é a métrica de ERRO/PERDA (Log Loss).

    O Log Loss é usado como métrica de pontuação e o sinal é invertido para 
    representar o Erro (Loss) Positivo.

    Args:
        estimator (Any): O modelo de ML a ser avaliado (ex: KNeighborsClassifier()).
        X (np.ndarray): O conjunto de features (treino).
        y (np.ndarray): Os rótulos (classes) de treino.
        cv (int): Número de K-Folds para a validação cruzada.
        n_jobs (int): Número de CPUs a serem usadas.
        train_sizes (np.ndarray): Frações do dataset para calcular os pontos da curva.
    """
    
    # 1. Geração da curva usando 'neg_log_loss'
    # Esta é a métrica padrão para problemas de classificação multi-classe.
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring='neg_log_loss' # Usando métrica de PERDA
    )

    # 2. Conversão da métrica: Inverter o sinal para obter o ERRO POSITIVO
    # scoring='neg_log_loss' retorna valores negativos. Invertemos o sinal para
    # que o Log Loss seja plotado como um erro positivo (Loss = -neg_log_loss)
    train_errors = -train_scores
    test_errors = -test_scores

    # 3. Cálculo da Média e Desvio Padrão do Erro
    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)

    # 4. PLOTAGEM
    plt.figure(figsize=(10, 6))
    plt.title(f"Curva de Aprendizado (ERRO/PERDA) - {estimator.__class__.__name__}")
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Erro (Log Loss)")
    plt.grid(True)
    
    # Linha do Erro de Treino
    plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Erro no Treino")
    # Área de Desvio Padrão (Variação)
    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1, color="r")

    # Linha do Erro de Validação
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Erro na Validação Cruzada")
    # Área de Desvio Padrão (Variação)
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")

    plt.legend(loc="best")
    plt.show()

def plot_learning_curve(model, X, y, titulo="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, marker='o', label="Treinamento")
    plt.plot(train_sizes, val_mean, marker='s', label="Validação")
    plt.title(titulo)
    plt.xlabel("Tamanho do conjunto de treino")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.legend()
    plt.show()

    return train_sizes, train_mean, val_mean
