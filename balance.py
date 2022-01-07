import mlp as mlp

# dataset: https://archive.ics.uci.edu/ml/datasets/Balance+Scale
# 1. Class Name: 3 (L, B, R) -> L = [1, 0, 0], B = [0, 1, 0], R = [0, 0, 1]
# 2. Left-Weight: 5 (1, 2, 3, 4, 5)
# 3. Left-Distance: 5 (1, 2, 3, 4, 5)
# 4. Right-Weight: 5 (1, 2, 3, 4, 5)
# 5. Right-Distance: 5 (1, 2, 3, 4, 5)

# TREINANDO REDE SOBRE DATASET DE CANCER DE MAMA

# DESCRIÇÃO DO DATASET
dataset = 'datasets/balance.csv'
entradas = 4
saidas = 3

# DESCRICAO DA REDE
camadas_ocultas = [6, 6, 6]
camadas = [entradas] + camadas_ocultas + [saidas]
velocidade = 0.001
n_epocas = 250
p_dados_teste = 0.1


ativ_ocultas = 'mish'
ativ_saida = 'sigmoid'
custo = 'mse'

# INICIALIZACAO DA REDE
iris = mlp.Mlp(eta=velocidade, epochs=n_epocas, layers=camadas)
iris.set_functions(hidden_layers=ativ_ocultas, out_layer=ativ_saida, cost_function=custo)
iris.load_dataset(dataset, test_coef=p_dados_teste)

# TREINANDO
iris.train()

# CLASSIFICANDO
iris.classify()
