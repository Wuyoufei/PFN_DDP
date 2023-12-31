#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#            PFN                        
r"""
try_gpu(i=0) 

initialize_pfn(m)

class Accumulator

accuracy(y_hat, y)

evaluate_accuracy(net, loss, data_iter, device,test=False)

train_procedure_in_each_epoch(net,tran_set,loss,optimizer,device)

plot_confusion_matrix(id, n, y_true, y_pred, classes="", normalize=True, cmap=plt.cm.Greens)

remap_pids(events, pid_i=0, error_on_unknown=True)

save_net(net,suffix)
"""                   
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import os
import sys
import torch.distributed as dist
import time
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    matplotlib.use('Agg')
except:
    print('please install matploltib in order to make plots')
    plt = False

def cheems():
    print(f'''
        ⠀⠀⠀⠀⠀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⡀⣯⡭⠀⢟⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⢠⣼⠏⠴⠶⠈⠘⠻⣘⡆⠀⠀⠀⠀⠀⠀
        ⠀⠀⣠⣾⡟⠁⡀⠀⠀⠀⡼⠡⠛⡄⠀⠀⠀⠀⠀
        ⠀⠀⠙⠻⢴⠞⠁⠀⠊⠀⠀⠀⠈⠉⢄⠀⠀⠀⠀
        ⠀⠀⠀⠀⢀⠀⠀⠀⢃⠄⠂⠀⠀⢀⠞⢣⡀⠀⠀           
        ⠀⠀⠀⠀⡌⠁⠀⠀⠀⢀⠀⠐⠈⠀⠀⡺⠙⡄⠀   
        ⠀⠀⠀⠀⡿⡀⠀⠀⠀⠁⠀⠴⠁⠀⠚⠀⡸⣷⠀
        ⠀⠀⠀⠀⢹⠈⠀⠀⠀⠀⠔⠁⠀⢀⠄⠀⠁⢻⣧
        ⠀⠀⠀⠀⣸⠀⢠⣇⠀⢘⣬⠀⠀⣬⣠⣦⣼⣿⠏
        ⡠⠐⢂⡡⠾⢀⡾⠋⠉⠉⡇⠀⢸⣿⣿⣿⡿⠃⠀
        ⠉⢉⡠⢰⠃⠸⠓⠒⠂⠤⡇⠀⡿⠟⠛⠁⠀⠀⠀
        ⠘⢳⡞⣅⡰⠁⠀⠀⠀⢀⠇⠀⡇⠀⠀⠀⠀⠀⠀
        ⠀⠀⠉⠉⠀⠀⠀⠀⢀⣌⢀⢀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠘⠊⠀⠀⠀⠀⠀⠀⠀
    Cheems shows up and tells you that the program has started!!!
        ''')

def try_gpu(i=0):  
    
    if (torch.cuda.device_count()) >= i + 1:     
        print(f'The name of GPU[{i}] is {torch.cuda.get_device_name(i)}')
        return torch.device(f'cuda:{i}')
    else:
        print(f'NO GPU{i}, have to use CPU...')
        return torch.device('cpu')
    
def initialize_pfn(m):
    if isinstance(m,torch.nn.Conv1d) or isinstance(m,torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)

class Accumulator:  
    #accumulate in n variables
    def __init__(self, n):
        self.data = [torch.tensor(0.0)] * n

    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [torch.tensor(0.0)] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def accuracy(y_hat, y):  
    #calculate the number of correct predictions
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()



def evaluate_accuracy(net, loss, data_iter,test=False):  
    #caculate the accuracy on specified dataset
    if isinstance(net, torch.nn.Module):
        net.eval()  #Essential!
    metric = Accumulator(3)  #sum_of_loss_of_all_events, sum_of_the_number_of_correct_predictions, all_events
    y_pred_test=[]
    y_true_test=[]
    with torch.no_grad():
        for X, y in data_iter:
            X=X.cuda()
            y=y.cuda()
            y_hat=net(X)
            l=loss(y_hat,y)
            metric.add(l.sum(), accuracy(y_hat, y), torch.tensor(y.numel()).cuda())
            if test:
                y_pred_test.append(F.softmax(y_hat,dim=1))
                y_true_test.append(y)
        if test:
            y_pred_test=torch.cat(y_pred_test,dim=0)
            y_true_test=torch.cat(y_true_test,dim=0)
    if test:
        #score, true, loss, acc 
        return y_pred_test,y_true_test,metric[0] / metric[2], metric[1] / metric[2]
    else:
        dist.reduce(metric[0],0,op=dist.ReduceOp.SUM)
        dist.reduce(metric[1],0,op=dist.ReduceOp.SUM)
        dist.reduce(metric[2],0,op=dist.ReduceOp.SUM)
        #loss, acc 
        return metric[0] / metric[2], metric[1] / metric[2]

def train_procedure_in_each_epoch(net,tran_set,loss,optimizer,local_rank):
    #sum_of_loss_of_all_events, sum_of_the_number_of_correct_predictions, all_events
    metric=Accumulator(3)
    if local_rank==0:
        pbar=tqdm(tran_set)
    net.train()
    for X, y in tran_set:
        X=X.cuda()
        y=y.cuda()
        optimizer.zero_grad()
        y_hat=net(X) # input_var can be on any device, including CPU
        l=loss(y_hat,y)
        l.mean().backward()
        optimizer.step()
        ################################################
        b_loss=l.sum()
        b_acc=accuracy(y_hat, y)
        b_num=torch.tensor(y.numel()).cuda()
        dist.reduce(b_loss,0,op=dist.ReduceOp.SUM)
        dist.reduce(b_acc,0,op=dist.ReduceOp.SUM)
        dist.reduce(b_num,0,op=dist.ReduceOp.SUM)
        metric.add(b_loss, b_acc, b_num)
        if local_rank==0:
            pbar.set_description(f'batch_acc:{float(b_acc/b_num):.3f}, batch_loss:{float(b_loss/b_num):.3f}')
            pbar.update(1)
    #loss_train, acc_train
    return metric[0] / metric[2], metric[1] / metric[2]


def plot_confusion_matrix(id, n, y_true, y_pred, classes="", normalize=True, cmap=plt.cm.Greens):

    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    cm = confusion_matrix(y_true, y_pred)
    cm=cm.astype('float')
    num_of_each_mode=np.sum(cm,axis=1)
    for i in range(len(cm)):
        cm[i]/=float(num_of_each_mode[i])
    with np.printoptions(precision=4, suppress=True):
        print(cm)
        print("%8.4f" % np.linalg.det(cm))
    if not os.path.exists('./conf_mat_npy/'):
        os.system('mkdir conf_mat_npy')
    np.save('conf_mat_npy/conf_%s.npy'% id, cm)
    print(f'conf_matrix.npy has been saved in {os.getcwd()}/conf_mat_npy/conf_{id}.npy')

    label_font = {'size': '14'}
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title("Confusion Matrix of Higgs decays", fontdict=label_font)
    plt.colorbar()
    if classes == "":
        classes = ["cc", "bb", r"$\mu \mu$", r"$\tau \tau$",
                   "gg", r"$\gamma\gamma$", "ZZ", "WW", r"$\gamma Z$"]
    tick_marks = np.arange(n)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True', fontdict=label_font)
    plt.xlabel('Predicted', fontdict=label_font)
    if not os.path.exists('./figs/'):
        os.system('mkdir figs')
    plt.savefig("figs/pfn_conf_%s.pdf" % id, dpi=800)
    print(f'conf_matrix.pdf has been saved in {os.getcwd()}/figs/pfn_conf_{id}.pdf')






PID2FLOAT_MAP = {22: 0,
                 211: .1, -211: .2, 
                 321: .3, -321: .4, 
                 130: .5, 
                 2112: .6, -2112: .7, 
                 2212: .8, -2212: .9, 
                 11: 1.0, -11: 1.1,
                 13: 1.2, -13: 1.3}

def remap_pids(events, pid_i=0, error_on_unknown=True):
    """Remaps PDG id numbers to small floats for use in a neural network.
    `events` are modified in place and nothing is returned.

    **Arguments**

    - **events** : _numpy.ndarray_
        - The events as an array of arrays of particles.
    - **pid_i** : _int_
        - The column index corresponding to pid information in an event.
    - **error_on_unknown** : _bool_
        - Controls whether a `KeyError` is raised if an unknown PDG ID is
        encountered. If `False`, unknown PDG IDs will map to zero.
    """

    if events.ndim == 3:
        pids = events[:,pid_i,:].astype(int).reshape((events.shape[0]*events.shape[2]))
        if error_on_unknown:
            events[:,pid_i,:] = np.asarray([PID2FLOAT_MAP[pid]
                                            for pid in pids]).reshape(events.shape[0],events.shape[2])
        else:
            events[:,pid_i,:] = np.asarray([PID2FLOAT_MAP.get(pid, 0)
                                            for pid in pids]).reshape(events.shape[0],events.shape[2])
    else:
        print('no remap_pid')

def save_net(net,suffix):
    if not os.path.exists('./net_params/'):
        os.system('mkdir net_params')
    torch.save(net.state_dict(),f'./net_params/net_{suffix}.params')
    print(f'net has been saved in [{os.getcwd()}/net_params/net_{suffix}.params]')










