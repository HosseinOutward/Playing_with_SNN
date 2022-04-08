import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


def fft_u(y,t):
    from scipy.fftpack import fft
    # Number of sample points
    N = t
    # sample spacing
    T = 1.0
    x = np.arange(N)
    yf = fft(y,)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (30, 10)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.ylim(0,0.25)
    plt.show()
# fft_u(np.array(snn_model.layers[3].history_U[1:])[:,3],t)


def plot_delta_comparison(snn_model,xx, t, l=None, ylim=None, xmlim=-0.1, xlim=None, nnn=False):
    for lay in range(len(snn_model.layers) - 1):
        print(lay)
        if l is not None and l==lay: continue
        for n in range(len(snn_model.layers[lay].U)):
            plt.rcParams['figure.figsize'] = (30, 1.5)
            # real_u=np.array(snn_model.layers[lay].history_U).T[n]
            # abs_u=np.array([snn_model.abstract_v_layer(xx,tt,lay) for tt in range(t+1)]).T[n]

            if nnn:
                real_u = np.array(snn_model.layers[lay].history_N).T[n]
                abs_u = np.array([snn_model.abstract_n_layer(xx, tt, lay) for tt in range(t + 1)]).T[n]

            plt.step(np.arange(len(abs_u)), abs_u - real_u, where='post')

            if ylim is not None: plt.ylim(-ylim, ylim)
            if xlim is not None: plt.xlim(xmlim, xlim)

            plt.grid()
            plt.show()


def plot_v_comparison(snn_model,xx, t, lay, ylim=None, xlim=None, xmlim=-0.1, ymlim=-0.1):
    for n in range(len(snn_model.layers[lay].U)):
        plt.rcParams['figure.figsize'] = (40, 5)
        real_u=np.array(snn_model.layers[lay].history_U).T[n]
        plt.step(np.arange(len(real_u)), real_u, where='post')

        abs_u=np.array([snn_model.abstract_v_layer(xx,tt,lay) for tt in range(t+1)]).T[n]
        plt.step(np.arange(len(abs_u)), abs_u, where='post')

        if ylim is not None: plt.ylim(ymlim,ylim)
        if xlim is not None: plt.xlim(xmlim,xlim)

        plt.grid()
        plt.show()


def plot_n_comparison(snn_model,xx, t, lay, ylim=None, xlim=None, xmlim=-0.1, ymlim=-0.1):
    for n in range(len(snn_model.layers[lay].U)):
        plt.rcParams['figure.figsize'] = (40, 5)
        real_n=np.array(snn_model.layers[lay].history_N).T[n]
        plt.step(np.arange(len(real_n)), real_n, where='post')

        abs_n=np.array([snn_model.abstract_n_layer(xx,tt,lay) for tt in range(t+1)]).T[n]
        plt.step(np.arange(len(abs_n)), abs_n, where='post')

        if ylim is not None: plt.ylim(ymlim,ylim)
        if xlim is not None: plt.xlim(xmlim,xlim)

        plt.grid()
        plt.show()


def make_dataset():
    x1 = np.random.normal(0, 1, (4, 2))
    # concatenate two 2d vectors
    x1 = np.concatenate(([x1 + np.random.normal(0, 0.01, x1.shape) for _ in range(100)]), axis=0)
    x1 = x1 + np.random.normal(0, 0.2, x1.shape)
    x1 = x1[np.random.permutation(len(x1))]
    y1 = np.ones(x1.shape[0])

    x2 = np.random.normal(0, 1, (4, 2))
    # concatenate two 2d vectors
    x2 = np.concatenate(([x2 + np.random.normal(0, 0.01, x2.shape) for _ in range(100)]), axis=0)
    x2 = x2 + np.random.normal(0, 0.2, x2.shape)
    x2 = x2[np.random.permutation(len(x2))]
    y2 = np.ones(x2.shape[0])

    y1 = np.array([np.zeros(y2.shape[0]), y1]).T
    y2 = np.array([y2, np.zeros(y1.shape[0])]).T
    y = np.concatenate((y1, y2), axis=0)
    x = np.concatenate((x1, x2), axis=0)

    a = np.random.permutation(int(x.shape[0] / 1.1))
    x = x[a]
    y = y[a]
    y = y[~np.isnan(x).any(axis=1)]
    x = x[~np.isnan(x).any(axis=1)]
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
    plt.show()

    return x, y


def make_model(x, y):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 5)
            self.fc3 = nn.Linear(5, 5)
            self.fc4 = nn.Linear(5, 4)
            self.fc5 = nn.Linear(4, 3)
            self.fc6 = nn.Linear(3, 2)
            self.fc7 = nn.Linear(2, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            x = torch.relu(x)
            x = self.fc4(x)
            x = torch.relu(x)
            x = self.fc5(x)
            x = torch.relu(x)
            x = self.fc6(x)
            x = torch.relu(x)
            x = self.fc7(x)
            x = torch.sigmoid(x)
            return x

    model = Net()
    # train model on x and y
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_func = nn.MSELoss()
    for epoch in range(2000):
        prediction = model(torch.from_numpy(x).float())
        loss = loss_func(prediction, torch.from_numpy(y).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

    plt.scatter(x[:, 0], x[:, 1], c=model(torch.from_numpy(x).float()).data.numpy()[:, 0])
    plt.show()

    return model


def get_changed_weight(x, wb, model):
    a = [torch.from_numpy(x.copy()).float()]
    b = [torch.from_numpy(x.copy()).float()]
    for i in range(len(wb[0])):
        a.append(eval('model.fc%s(b[-1])' % (i + 1)))
        b.append(torch.relu(a[-1]))
    b.pop(-1)
    b.append(torch.sigmoid(a[-1]))

    for i in range(len(a)):
        a[i] = a[i].data.numpy().copy()
        b[i] = b[i].data.numpy().copy()
    a.pop(0)
    b.pop(0)

    f = lambda x: np.percentile(x.max(),99.9) # np.percentile(abs(x).max(),99.9)
    wb = wb.copy()
    for i in range(len(wb[0])):
        for j in range(len(wb[0][i])):
            if i == len(wb[0]) - 1: continue
            m = a[i][:, j]
            if max(m) <= 0: continue
            wb[0][i][j] /= f(m)
            wb[1][i][j] /= f(m)

        for j in range(len(wb[0][i].T)):
            if i == 0: continue
            m = a[i - 1][:, j]
            if max(m) <= 0: continue
            wb[0][i][:, j] *= f(m)

    return wb, (a,b)


if __name__ == '__main__':
    x, y = make_dataset()
    model = make_model(x, y)

    w = list(model.parameters())
    wb, (a,b) = [[w[i].data.numpy().copy() for i in range(0, len(w), 2)],
          [w[i].data.numpy().copy() for i in range(1, len(w), 2)]]

    wb=get_changed_weight(x,wb,model)
    snn_model=SNN(wb[0],wb[1],fin_act=lambda x:1/(1+np.exp(-x)),layer_type=Layer)

    # snn_model=SNN(wb[0],wb[1],fin_act=lambda x:1/(1+np.exp(-x)),layer_type=AlterLayer)

    xx = x.astype('float64')[2]
    t = 1000
    print(model(torch.from_numpy(xx.copy()).float()).data.numpy() * 1000 // 1 / 1000)
    print(snn_model(xx, t) * 1000 // 1 / 1000)
    print(snn_model.abstract_n(xx, t) * 1000 // 1 / 1000)
