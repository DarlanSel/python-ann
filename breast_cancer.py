import mlp as mlp

# dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

# Clump Thickness: 1 - 10
# Uniformity of Cell Size: 1 - 10
# Uniformity of Cell Shape: 1 - 10
# Marginal Adhesion: 1 - 10
# Single Epithelial Cell Size: 1 - 10
# Bare Nuclei: 1 - 10
# Bland Chromatin: 1 - 10
# Normal Nucleoli: 1 - 10
# Mitoses: 1 - 10
# Benign: [1, 0]
# Malign: [0, 1]

# TREINANDO REDE SOBRE DATASET DE CANCER DE MAMA

# DESCRIÇÃO DO DATASET
dataset = 'datasets/breast-cancer.csv'
entradas = 9
saidas = 2

# DESCRICAO DA REDE
camadas_ocultas = [5, 5]
camadas = [entradas] + camadas_ocultas + [saidas]
velocidade = 0.01
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
