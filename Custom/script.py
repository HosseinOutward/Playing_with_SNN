import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from spikingjelly.clock_driven import functional


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


def plot_delta_r_comparison(snn,ann,xx, ylim=None, xmlim=-0.1, xlim=None):
    torch.cuda.empty_cache()
    plt.rcParams['figure.figsize'] = (40, 1.5)

    functional.reset_net(snn)
    for _ in range(xlim): snn(xx)

    ann_x = torch.clone(xx)
    for i,l in enumerate(snn.fc):
        ann_x=ann.fc[i](ann_x)
        if str(type(l)) != "<class \'__main__.AIFNode\'>" and \
            str(type(l)) != "<class \'__main__.HisIFNode\'>": continue

        print("***",i)
        for snn_his, ann_his in zip(torch.stack(l.history_n).T, ann_x):
            print(ann_his.data.cpu().numpy())

            x_ax=torch.arange(len(snn_his), device='cuda')
            x_ax[0]=1
            plt.step(x_ax.data.cpu().numpy(), (snn_his/x_ax-ann_his).data.cpu().numpy(), where='post')

            if ylim is not None: plt.ylim(-ylim, ylim)
            if xmlim is not None: plt.xlim(xmlim, xlim)

            plt.grid()
            plt.show()


def ann_max_norm(x_tensor,new_ann, m=1,only_mul=False):

    for i,l in enumerate(new_ann.fc):
        if type(l) is not torch.nn.ReLU: continue

        torch.cuda.empty_cache()
        with torch.no_grad():
            ann_x=torch.clone(x_tensor)
            for j in range(i+1): ann_x=new_ann.fc[j](ann_x)

        spike_max=torch.quantile(ann_x,0.999,0,False,interpolation='lower')
        spike_max[spike_max<=0.]=1.

        if only_mul: spike_max=torch.ones_like(spike_max)

        new_ann.fc[i-1].bias = torch.nn.Parameter(torch.div(new_ann.fc[i-1].bias,spike_max/m))
        new_ann.fc[i-1].weight=torch.nn.Parameter(torch.div(new_ann.fc[i-1].weight.T,spike_max/m).T)
        new_ann.fc[i+1].weight=torch.nn.Parameter(torch.mul(new_ann.fc[i+1].weight,spike_max/m))



def snn_max_norm(x_tensor,AIFsnn,t):
    for i,l in enumerate(AIFsnn.fc):
        if str(type(l)) != "<class \'__main__.AIFNode\'>": continue

        torch.cuda.empty_cache()
        with torch.no_grad():
            functional.reset_net(AIFsnn)
            for _ in range(t): AIFsnn(x_tensor)

            spike_max=torch.quantile(torch.cat(l.history_s),0.999,0,False,interpolation='lower')#
            spike_max[spike_max==0.]=1.

            AIFsnn.fc[i-1].bias = torch.nn.Parameter(torch.div(AIFsnn.fc[i-1].bias,spike_max))
            AIFsnn.fc[i-1].weight=torch.nn.Parameter(torch.div(AIFsnn.fc[i-1].weight.T,spike_max).T)
            AIFsnn.fc[i+1].weight=torch.nn.Parameter(torch.mul(AIFsnn.fc[i+1].weight,spike_max))


def check_neg_error(snn,ann,x_tensor,t,ylim=None):
    torch.cuda.empty_cache()
    with torch.no_grad():
        functional.reset_net(snn)
        for _ in range(t-1): snn(x_tensor)

    ann_x = torch.clone(x_tensor)
    for i,l in enumerate(snn.fc):
        ann_x=ann.fc[i](ann_x)

        if str(type(l)) != "<class \'__main__.AIFNode\'>" and \
            str(type(l)) != "<class \'__main__.HisIFNode\'>": continue

        x_ax=torch.arange(len(l.history_n), device='cuda')
        x_ax[0]=1

        his_n=torch.stack([sss.T for sss in l.history_n]).T

        for sample_i, (his_n_xx, ann_xx) in enumerate(zip(torch.round(his_n),ann_x)):
            for sneu,aneu in zip(his_n_xx/x_ax,ann_xx):
                if (sneu-aneu>0.0001).any():
                    print(sample_i)
                    plt.plot(x_ax.data.cpu().numpy(),(sneu-aneu).data.cpu().numpy())
                    if ylim is not None: plt.ylim(-ylim, ylim)
                    plt.grid()
                    plt.show()


def proc_vs_acc(snn,ann,x_tensor,y_tensor,t,ylim=None):
    loss_func = nn.CrossEntropyLoss()

    torch.cuda.empty_cache()
    with torch.no_grad():
        functional.reset_net(snn)
        acc_snn=[0,*[loss_func(snn(x_tensor), y_tensor) for _ in range(t-1)]]
        acc_snn[0]=acc_snn[1]
        acc_snn=torch.stack(acc_snn)


    with torch.no_grad():
        functional.reset_net(ann)
        acc_ann=loss_func(ann(x_tensor), y_tensor)

    time_ann = 0; time_snn = 0
    for i,l in enumerate(snn.fc):
        if str(type(l)) != "<class \'__main__.TimedLinear\'>" and \
            str(type(l)) != "<class \'__main__.TimedLinearANN\'>": continue

        l.proc_time[0] = l.proc_time[-1] * 0
        time_ann += ann.fc[i].proc_time[1]
        time_snn += torch.stack([aaa.T for aaa in l.proc_time]).T

    acc_snn=np.array([acc_snn.data.cpu().numpy() for _ in time_snn]).T
    time_snn=time_snn.data.cpu().numpy().T

    plt.rcParams['figure.figsize'] = (40, 10)
    plt.scatter(time_snn,acc_snn,alpha=0.01,s=1)
    plt.plot(time_snn,acc_snn,alpha=0.005)

    perc=lambda x: np.percentile(x,70,axis=1)
    plt.scatter(perc(time_snn),perc(acc_snn),s=15)
    plt.plot(perc(time_snn),perc(acc_snn))

    plt.axvline(x=time_ann.data.cpu().numpy().mean())
    plt.axhline(y=acc_ann.data.cpu().numpy())
    if ylim is not None: plt.ylim(np.percentile(acc_ann.data.cpu().numpy(),99.9), ylim)
    plt.xlim(-0.02, np.percentile(time_snn,95))
    plt.show()


def snn_min_norm(x_tensor,IFsnn,t):
    for i,l in enumerate(IFsnn.fc):
        if str(type(l)) != "<class \'__main__.AIFNode\'>": continue

        torch.cuda.empty_cache()
        with torch.no_grad():
            functional.reset_net(IFsnn)
            for _ in range(t): IFsnn(x_tensor)

            mult=[]
            for sxx in l.history_s:
                mult.append([])
                for neu in sxx:
                    while neu[0]==0: neu=neu[1:]
                    while neu[-1]!=0: neu=neu[:-1]
                    num_ones=(neu==0.).sum().data.cpu().numpy()
                    num_zeros=(len(neu)-num_ones)
                    mult[-1].append(num_zeros/num_ones)

            mult=np.array(mult).min(1)

            spike_max=torch.quantile(torch.cat(l.history_s),0.999,0,False,interpolation='lower')#
            spike_max[spike_max==0.]=1.

            AIFsnn.fc[i-1].bias = torch.nn.Parameter(torch.div(AIFsnn.fc[i-1].bias,spike_max))
            AIFsnn.fc[i-1].weight=torch.nn.Parameter(torch.div(AIFsnn.fc[i-1].weight.T,spike_max).T)
            AIFsnn.fc[i+1].weight=torch.nn.Parameter(torch.mul(AIFsnn.fc[i+1].weight,spike_max))


