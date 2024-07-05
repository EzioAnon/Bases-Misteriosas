import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


algoritimos = {
    'Perceptron': Pipeline([('normalizador', StandardScaler()), ('clf', Perceptron(max_iter=1000, tol=1e-3))]),
    'Decision Tree': Pipeline([('normalizador', StandardScaler()), ('clf', DecisionTreeClassifier())]),
    'KNN': Pipeline([('normalizador', StandardScaler()), ('clf', KNeighborsClassifier())])
}


def carregar_pasta(pasta):
    df = pd.read_csv(pasta, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def validacao_cruzada(X, y, algoritimos):
    resultados = {}
    for nome, pipline in algoritimos.items():
        acuracias = []
        for treino_idx, teste_idx, in KFold(n_splits= 10).split(X):
            X_treino, X_teste = X[treino_idx], X[teste_idx]
            y_treino, y_teste = y[treino_idx], y[teste_idx]

            normalizador = StandardScaler()
            X_treino_normalizado = normalizador.fit_transform(X_treino)
            
            pipline.fit(X_treino_normalizado,y_treino)

            X_teste_normalizado = normalizador.transform(X_teste)

            acuracia = pipline.score(X_teste_normalizado,y_teste)
            acuracias.append(acuracia)
        
        resultados[nome] = acuracias
        
        print(f"{nome}- Média da Acurácia: {np.mean(acuracias):.2f}")
    return resultados
    
def plot_resultados(resultados):
    nomes = list(resultados.keys())
    medias = [np.mean(acuracia) for acuracia in resultados.values()]
    desvios =[np.std(acuracia) * 2 for acuracia in resultados.values()]

    plt.figure(figsize=(10,5))
    plt.bar(nomes,medias,yerr=desvios, capsize=5)
    plt.xlabel('Algoritimos')
    plt.show()