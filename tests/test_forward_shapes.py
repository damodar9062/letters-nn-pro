import numpy as np
from letters_nn import NeuralNet, one_hot

def test_forward_shapes():
    N,M,D,K = 5, 128, 10, 10
    X = np.random.rand(N,M)
    y = np.random.randint(0,K,size=N)
    Y = one_hot(y,K)
    nn = NeuralNet(M=M, D=D, K=K, init="random", lr=0.1)
    yhat, cache = nn.forward(X)
    assert yhat.shape == (N,K)
    gA, gB = nn.backward(cache, Y)
    assert gA.shape == (D, M+1)
    assert gB.shape == (K, D+1)
