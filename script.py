import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


#####################
def relu(x):
    x *= x > 0
    return x


class Layer:
    def __init__(self, w, b):
        self.W = w
        self.U = np.array([0 for i in range(self.W.shape[0])]).astype('float64')
        self.b = b
        self.N = self.U.copy()*0

        self.history_out=[self.U.copy()]
        self.history_U = [self.U.copy()]
        self.history_S = [self.N.copy()]
        self.history_N = [self.N.copy()]

    def reset(self):
        self.U = self.U*0
        self.N = self.U.copy()

        self.history_U = [self.U.copy()]
        self.history_S = [self.N.copy()]
        self.history_N = [self.N.copy()]

    def forward(self, inp):
        x, spiked = self.simulate(inp)
        self.history_out.append(x.copy())
        self.history_U.append(self.U.copy())
        self.history_S.append(spiked.copy())
        self.history_N.append(self.N.copy())
        # if (self.U>=1).any() and type(self) is Layer: print(self.U)
        return x

    def simulate(self, inp):
        self.U += np.dot(self.W, inp) + self.b
        self.U=self.U.round(5)
        spiked = self.U >= 1
        return self.neuron_func(spiked), spiked

    def neuron_func(self, spiked):
        if not spiked.any(): return spiked
        self.N += spiked
        self.U[spiked] -= 1
        return spiked

    def abstract_n(self, inp, t, rou=4):
        if len(np.array(inp).shape) == 2:
            return np.array([self.abstract_n(xx, t) for xx in inp])
        aw=np.dot(self.W, inp) + self.b * t
        return np.floor(relu((aw + relu(-self.abstract_v(inp, t))).round(rou)))
        # return (aw-self.abstract_v(inp, t)).round(rou)

        # v=self.history_U[t]
        # def f(A,V):
        #     if -V>0:
        #         if A-V>0:return np.floor((A-V).round(rou))
        #         else: return 0
        #     else:
        #         if A>0: return np.floor(A.round(rou))
        #         else: return 0
        # def f(A,V):
        #     if V<0 and A>V:return np.floor((A-V).round(rou))
        #     if V<0 and A<=V: return 0
        #     if V>=0 and A>0: return np.floor(A.round(rou))
        #     if V>=0 and A<=0: return 0
        # return np.array([f(aa,vv) for aa,vv in zip(aw,v)])

    def abstract_v(self, inp, t):
        if len(np.array(inp).shape) == 2:
            return np.array([self.abstract_v(xx, t) for xx in inp])
        return self.history_U[t]
        # return aw-self.abstract_n(inp, t)
        # return aw-(aw>0)*np.floor(aw)

        # aw=np.dot(self.W, inp) + self.b * t
        # v=self.history_U[t]
        # p=np.dot(self.W*(self.W>0), inp) + self.b*(self.b>0) * t
        # self.VA=[]
        # def f(A,V,P):
        #     VA = list(range(int(np.ceil(A.round(4))),int(np.ceil(P.round(4)))))
        #     if A.round(4)//1==A.round(4): VA.append(A.round(4))
        #     self.VA.append(A-np.array(VA))
        #
        #     if V<0 and A>V: return A-np.floor((A-V).round(4))
        #     else: return A-np.floor(relu(A).round(4))
        #
        # return np.array([f(aa,vv,pp) for aa,vv,pp in zip(aw,v,p)])

    def plot_history(self, lim, ylim=None):
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.step(np.arange(len(self.history_U)), self.history_U, label='U', where='post')
        plt.scatter(np.arange(len(self.history_S)), np.array(self.history_S).sum(axis=1) * 3 - 2, label='S')
        plt.ylim(-0.1, max(np.max(self.history_U) + 0.1, 1.1))
        plt.xlim(0, lim)
        plt.grid()
        plt.show()

        history_N = [np.array(self.history_S[:tt+1]).sum(axis=0) for tt in range(len(self.history_S))]
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.step(np.arange(len(self.history_S)), np.array(history_N), label='N', where='post')
        if ylim is not None: plt.ylim(-0.1, ylim)
        plt.xlim(0, lim)
        plt.grid()
        plt.show()


class AlterLayer(Layer):
    def neuron_func(self, spiked):
        if not spiked.any(): return spiked
        output = spiked * (self.U // 1)
        self.N += output
        self.U[spiked] %= 1
        return output


class OutLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(OutLayer, self).__init__(*args, **kwargs)

    def neuron_func(self, spiked):
        self.N += self.U.copy()
        self.U = self.U*0
        return self.U.copy()

    def abstract_n(self, inp, t):
        if len(np.array(inp).shape) == 2:
            return np.array([self.abstract_n(xx, t) for xx in inp])
        return np.dot(self.W, inp)+self.b*t


class SNN:
    def __init__(self, w, b, fin_act=lambda x: x, layer_type=Layer):
        self.layers = [layer_type(w[m], b[m]) for m in range(len(w) - 1)] + [OutLayer(w[-1], b[-1])]
        self.final_activation = fin_act

    def reset(self):
        for l in self.layers: l.reset()

    def __call__(self, x, t=1):
        if len(x.shape) == 1:
            out = self.predict(x, t).copy()
            self.reset()
            return out
        if len(x.shape) != 2: raise ValueError('x must be a vector or 2D matrix')

        out = []
        for i in range(x.shape[0]):
            out.append(self.predict(x[i], t).copy())
            self.reset()
        return np.array(out)

    def forward(self, x):
        layer_out = x.copy()
        for i, l in enumerate(self.layers): layer_out = l.forward(layer_out)
        return layer_out

    def predict(self, x, t):
        for _ in range(t): self.forward(x.copy())
        return self.final_activation(self.layers[-1].N / t)

    def abstract_n_layer(self, x, t, lim):
        x = x * t
        for l in self.layers[:lim+1]:
            x = l.abstract_n(x.copy(), t)
        return x

    def abstract_n(self, x, t):
        return self.final_activation(self.abstract_n_layer(x, t, len(self.layers)) / t)

    def abstract_v_layer(self, x, t, lim):
        x = x * t
        for l in self.layers[:lim]: x = l.abstract_n(x.copy(), t)
        return self.layers[lim].abstract_v(x.copy(), t)

    def abstract_v(self, x, t):
        return self.final_activation(self.abstract_v_layer(x, t, len(self.layers)) / t)

#####################


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
        if l is not None and l!=lay: continue
        print("***",lay)
        his_u=np.array(snn_model.layers[lay].history_U).T
        abs_his_u=np.array([snn_model.abstract_v_layer(xx,tt,lay) for tt in range(t+1)]).T
        for n in range(len(snn_model.layers[lay].U)):
            plt.rcParams['figure.figsize'] = (40, 1.5)
            real_u=his_u[n]
            abs_u=abs_his_u[n]

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

        real_s=np.array(snn_model.layers[lay].history_S).T[n]
        for i,xc in enumerate(real_s):
            if xc: plt.axvline(x=i)
        plt.axhline(y=0)

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


def plot_r_comparison(snn_model,xx, t, lay, ylim=None, xlim=None, xmlim=-0.1, ymlim=-0.1):
    for n in range(len(snn_model.layers[lay].U)):
        plt.rcParams['figure.figsize'] = (40, 5)
        real_n=np.array(snn_model.layers[lay].history_N).T[n]
        plt.step(np.arange(len(real_n)), [r/(tt if tt!=0 else 1) for tt,r in enumerate(real_n)], where='post')

        if ylim is not None: plt.ylim(ymlim,ylim)
        if xlim is not None: plt.xlim(xmlim,xlim)

        plt.grid()
        plt.show()


def plot_r_delta_comparison(snn_model, ann_model, xx, t, l=None, ylim=None, xlim=None, xmlim=0.9, ymlim=-0.1):
    for lay in range(len(snn_model.layers) - 1):
        if l is not None and l!=lay: continue
        ann_a=torch.from_numpy(xx.copy()).float()
        for i in range(0,lay+1): ann_a=torch.relu(eval('ann_model.fc%s(ann_a)' % (i+1)))
        ann_a=ann_a.data.numpy().astype('float')

        snn_a=np.array(snn_model.layers[lay].history_N).T[:,1:]

        print("***",lay)
        for n in range(len(snn_model.layers[lay].U)):
            plt.rcParams['figure.figsize'] = (40, 3)

            print(ann_a[n].round(4))
            plt.step(np.arange(len(snn_a[n]))+1, snn_a[n]/np.arange(1,len(snn_a[n])+1)-ann_a[n], where='post')

            if ylim is not None: plt.ylim(-ylim, ylim)
            if xlim is not None: plt.xlim(xmlim, xlim)

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


def get_max_act_x(x,snn_model,wb,t):
    snn=snn_model(wb[0],wb[1],fin_act=lambda x:x,layer_type=AlterLayer)
    snn.predict(x, t)
    return np.array([np.array(l.history_out).max(axis=0) for l in snn.layers])


def get_max_act(x,snn_model,wb,t):
    from multiprocessing import Pool as ThreadPool

    p = ThreadPool(36)
    layer_x = np.array(p.map(get_max_act_x, [(xx,snn_model,wb,t) for xx in x])).T
    p.close()
    p.join()

    return [np.array(list(l)).max(axis=0) for l in layer_x]


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