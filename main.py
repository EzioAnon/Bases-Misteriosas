import funcoes



X_base_1, y_base_1 = funcoes.carregar_pasta('bases/0-Todos-CSV/13.csv')
X_base_2, y_base_2 = funcoes.carregar_pasta('bases/0-Todos-CSV/27.csv')


base_1 = funcoes.validacao_cruzada(X_base_1,y_base_1,funcoes.algoritimos)
base_2 = funcoes.validacao_cruzada(X_base_2,y_base_2,funcoes.algoritimos)


funcoes.plot_resultados(base_1)
funcoes.plot_resultados(base_2)