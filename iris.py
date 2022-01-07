from mlp import Mlp

# TREINANDO REDE SOBRE DATASET DA IRIS

# DESCRIÇÃO DO DATASET
dataset = 'datasets/iris.csv'
entradas = 4  # sepallength, sepalwidth, petallength, petalwidth
saidas = 3    # setosa, versicolor, virginica

# CONFIGURACAO DA REDE
camadas_ocultas = [5]  # uma camada com 5 neuronios
camadas = [entradas] + camadas_ocultas + [saidas]
velocidade = 0.001
n_epocas = 500
p_dados_teste = 0.1

ativ_ocultas = 'sigmoid'
ativ_saida = 'softmax'
custo = 'mse'

# INICIALIZACAO DA REDE
iris = Mlp(eta=velocidade, epochs=n_epocas, layers=camadas)
iris.set_functions(hidden_layers=ativ_ocultas, out_layer=ativ_saida, cost_function=custo)
iris.load_dataset(dataset, has_header=True, test_coef=p_dados_teste)

# TREINANDO SOBRE A IRIS
iris.train()

# CLASSIFICANDO A IRIS
iris.classify()
