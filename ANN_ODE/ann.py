import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): s = sigmoid(x); return s * (1 - s)
def d2_sigmoid(x): s = sigmoid(x); return s * (1 - s) * (1 - 2 * s)

def tanh(x): return np.tanh(x)
def d_tanh(x): t = tanh(x); return 1 - t**2
def d2_tanh(x): t = tanh(x); return -2 * t * (1 - t**2)

def relu(x): return np.maximum(0, x)
def d_relu(x): return np.where(x > 0, 1.0, 0.0)
def d2_relu(x): return np.zeros_like(x)

activation_funcs = {
    'sigmoid': (sigmoid, d_sigmoid, d2_sigmoid),
    'tanh': (tanh, d_tanh, d2_tanh),
    'relu': (relu, d_relu, d2_relu)
}

class NeuralNet:
    def __init__(self, n_hidden=10, n_output=1, activation='sigmoid'):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.w1 = np.random.randn(n_hidden)
        self.b1 = np.random.randn(n_hidden)
        self.w2 = np.random.randn(n_output, n_hidden)
        self.b2 = np.random.randn(n_output)
        self.activation_name = activation
        self.activation, self.d_activation, self.d2_activation = activation_funcs[activation]

    def forward(self, x):
        x_reshaped = np.atleast_2d(x)
        z = x_reshaped @ self.w1.reshape(1, -1) + self.b1
        a = self.activation(z)
        out = a @ self.w2.T + self.b2
        return out

    def derivative(self, x):
        n = len(x)
        if n < 2:
            return np.zeros_like(x)
            
        n_x = self.forward(x)
        derivatives = np.zeros_like(n_x)

        if n > 2:
            dx_central = x[2:] - x[:-2]
            dn_central = n_x[2:] - n_x[:-2]
            derivatives[1:-1] = dn_central / dx_central

        derivatives[0] = (n_x[1] - n_x[0]) / (x[1] - x[0])
        derivatives[-1] = (n_x[-1] - n_x[-2]) / (x[-1] - x[-2])

        return derivatives

    def second_derivative(self, x):
        z = x @ self.w1.reshape(1, -1) + self.b1
        d2z_dx2 = (self.w1**2).reshape(1, -1)
        d2a = self.d2_activation(z) * d2z_dx2
        return d2a @ self.w2.T

def pack_params(net):
    return np.concatenate([
        net.w1.flatten(),
        net.b1.flatten(),
        net.w2.flatten(),
        net.b2.flatten()
    ])

def unpack_params(net, params):
    n, m = net.n_hidden, net.n_output
    net.w1 = params[:n]
    net.b1 = params[n:2*n]
    net.w2 = params[2*n:2*n + n*m].reshape(m, n)
    net.b2 = params[2*n + n*m:2*n + n*m + m]