import numpy as np

if __name__ == '__main__':
    sensitivity=1
    privacy_budget = 0.9
    p = np.exp(privacy_budget/sensitivity) / (np.exp(privacy_budget/sensitivity) + 1)
    noise=np.random.laplace(size=(3,3))
    coin=np.random.random(size=(3,3))
    flip=np.where(coin<p,1,0)
    noise=noise*flip
    print(noise)