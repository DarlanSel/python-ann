import torch
import random


class Mlp:
    # Mish Helpers
    serialize = lambda v: ((v <= 20) * v) + (v > 20) * 20.0
    softplus = lambda v: torch.log(1 + torch.exp(Mlp.serialize(v)))
    sech = lambda x: (1 / torch.cosh(x))

    # Funções de ativação
    sigmoid = lambda v: 1/(1+torch.exp(-v))
    sigmoid_ = lambda y: y * (1 - y)

    softmax = lambda v: torch.exp(v - v[v.argmax()]) / (torch.exp(v - v[v.argmax()]).sum())
    softmax_ = lambda y: y - y**2

    relu = lambda v: (v > 0) * v
    relu_ = lambda y: (y >= 0).float()

    mish = lambda v: v * (torch.tanh(Mlp.softplus(v)))
    mish_ = lambda v: (Mlp.sech(Mlp.softplus(v))**2) * v * Mlp.sigmoid(v) + (Mlp.mish(v) / v)

    # Funções de custo
    mse = lambda y, d: ((y - d)**2).sum()
    mse_ = lambda y, d: (y - d)

    ecc = lambda y, d: (-1) * torch.matmul(d, torch.log(y).T)
    ecc_ = lambda y, d: (-1) * (d / y)

    activation_functions = {
        'sigmoid': [sigmoid, sigmoid_],
        'softmax': [softmax, softmax_],
        'relu': [relu, relu_],
        'mish': [mish, mish_],
    }

    cost_functions = {
        'mse': [mse, mse_],
        'ecc': [ecc, ecc_],
    }

    # ----------------------------------------INITIALIZE--------------------------------------------
    def __init__(self, *, eta=0.01, epochs=150, layers=[4, 5, 3]):
        self.ds = None
        self.ls = None
        self.testDs = None
        self.testLs = None

        self.f = None
        self.f_ = None
        self.j = None
        self.j_ = None
        self.act_strings = None
        self.cost_string = None

        # hiperparâmetros
        self.eta = eta
        self.epochs = epochs
        self.layers = layers

        # CONFIGURAÇÕES
        self.create_layers()

        self.set_functions()

    # ----------------------------------------CREATE LAYERS--------------------------------------------
    def create_layers(self):
        self.w = [None]
        self.b = [None]

        for i in range(1, len(self.layers)):
            self.w.append(torch.randn([self.layers[i-1], self.layers[i]]))
            self.b.append(torch.zeros([self.layers[i]]))

    # ----------------------------------------SET FUNCTIONS--------------------------------------------
    def set_functions(self, hidden_layers='sigmoid', out_layer='sigmoid', cost_function='mse'):
        hidden_layer_funcs = []
        hidden_layer_funcs_ = []

        self.cost_string = cost_function

        if(isinstance(hidden_layers, str)):
            self.act_strings = [hidden_layers] * (len(self.layers) - 2) + [out_layer]

            hidden_layer_funcs = [Mlp.activation_functions[hidden_layers][0]] * (len(self.layers) - 2)
            hidden_layer_funcs_ = [Mlp.activation_functions[hidden_layers][1]] * (len(self.layers) - 2)
        else:
            self.act_strings = [hidden_layers] + [out_layer]

            for f in hidden_layers:
                hidden_layer_funcs.append(Mlp.activation_functions[f][0])
                hidden_layer_funcs_.append(Mlp.activation_functions[f][1])

        # [Camada de Entrada]  +  [Camadas Ocultas]  +  [Camada de Saída]
        self.f = [None] + hidden_layer_funcs + [Mlp.activation_functions[out_layer][0]]
        self.f_ = [None] + hidden_layer_funcs_ + [Mlp.activation_functions[out_layer][1]]

        self.j = Mlp.cost_functions[cost_function][0]
        self.j_ = Mlp.cost_functions[cost_function][1]

    # ----------------------------------------LOAD DATASET--------------------------------------------
    def load_dataset(self, file, *, test_coef=0.1, has_header=False):
        # Pega o total de atributos baseado na camada de entrada
        attr_number = self.layers[0]

        ds = []
        ls = []
        tds = []  # Test dataset
        tls = []  # Test labelset

        arquivo = open(file, "r")
        linhas = arquivo.read().split("\n")
        arquivo.close()

        if(has_header):
            linhas.remove(linhas[0])

        linhas = list(filter(lambda a: a != "", linhas))

        for i in range(len(linhas)):
            linhas[i] = linhas[i].split(',')
            for j in range(len(linhas[i])):
                linhas[i][j] = float(linhas[i][j])

        # Separa a porcentagem de atributos informada em test_coef para os dados de teste
        for i in range(int(len(linhas) * test_coef)):
            pos = random.randint(0, len(linhas))

            tds.append(linhas[pos][0:attr_number])
            tls.append(linhas[pos][attr_number:])

            linhas.pop(pos)

        for i in range(len(linhas)):
            ds.append(linhas[i][0:attr_number])
            ls.append(linhas[i][attr_number:])

        self.ds = torch.tensor(ds).float()
        self.ls = torch.tensor(ls).float()
        self.testDs = torch.tensor(tds).float()
        self.testLs = torch.tensor(tls).float()

    # ----------------------------------------TRAIN--------------------------------------------
    def train(self):
        for e in range(self.epochs):
            loss = 0
            acc = 0

            y = [None] * len(self.w)
            delta = [None] * len(self.w)

            for i in range(len(self.ds)):
                # fase forward

                y[0] = self.ds[i]  # y[0] recebe as entradas do primeiro dado
                d = self.ls[i]     # d recebe a saida desejada da primeira entrada

                v = [None] * len(self.b)

                for k in range(1, len(self.w)):
                    v[k] = torch.matmul(y[k-1], self.w[k]) + self.b[k]
                    y[k] = self.f[k](v[k])

                loss += self.j(y[-1], d)
                acc += d[y[-1].argmax()]

                # fase backward: camada de saída
                delta[-1] = self.f_[-1](y[-1]) * self.j_(y[-1], d)
                self.w[-1] = self.w[-1] - self.eta * torch.matmul(y[-2].view(-1, 1), delta[-1].view(1, -1))
                self.b[-1] = self.b[-1] - self.eta * delta[-1]

                # fase backward: camada oculta
                for j in range(len(self.w) - 2, 0, -1):
                    if(self.act_strings[j-1] == 'mish'):
                        delta[j] = self.f_[j](v[j]) * torch.matmul(delta[j+1], self.w[j+1].T)
                    else:
                        delta[j] = self.f_[j](y[j]) * torch.matmul(delta[j+1], self.w[j+1].T)

                    self.w[j] = self.w[j] - self.eta * torch.matmul(y[j-1].view(-1, 1), delta[j].view(1, -1))
                    self.b[j] = self.b[j] - self.eta * delta[j]

            print("e", e + 1, "| acc", acc/len(self.ds), "| loss", loss/len(self.ds))

    # ----------------------------------------CLASSIFY--------------------------------------------
    def classify(self):
        acc = 0
        y = [None] * len(self.w)

        for i in range(len(self.testDs)):
            y[0] = self.testDs[i]

            for j in range(1, len(self.w)):
                v = torch.matmul(y[j-1], self.w[j]) + self.b[j]
                y[j] = self.f[j](v)

            acc += self.testLs[i][y[-1].argmax()]

            print("Esperado: ", self.testLs[i], " -  Obtido: ", y[-1], " -  Acurácia: ", acc/(i + 1))
