import mlp as mlp

# dataset: https://archive.ics.uci.edu/ml/datasets/seeds

# 1. area A,
# 2. perimeter P,
# 3. compactness C = 4*pi*A/P^2,
# 4. length of kernel,
# 5. width of kernel,
# 6. asymmetry coefficient
# 7. length of kernel groove.
# All of these parameters were real-valued continuous.
# Output: Karma = [1, 0, 0], Rosa = [0, 1, 0], Canadian = [0, 0, 1]

# TREINANDO REDE SOBRE DATASET DE CANCER DE MAMA

# DESCRIÇÃO DO DATASET
dataset = 'datasets/seeds.csv'
entradas = 7
saidas = 3

# DESCRICAO DA REDE
camadas_ocultas = [5]
camadas = [entradas] + camadas_ocultas + [saidas]
velocidade = 0.001
n_epocas = 3000
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
