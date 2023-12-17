import torch
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchsummary import summary
import my_pfn_load
import my_PFN
import my_tools
import numpy as np
import sys
import os
import time
import DDP_Config
import torch.distributed as dist
import torch.multiprocessing as mp
def main(local_rank, suffix, process, train_set, val_set, test_set):
    DDP_Config.init_ddp(local_rank)
    ########################################################################################################    
                                                #Hyperparameters
    epochs=20
    batchsize=512
    lr=0.001
    use_bn=True
    Phi_sizes=(100, 100, 128)
    F_sizes=(100, 100, 100)
    num_classes=len(process)
    #########################################################################################################
    if local_rank == 0:
        print(f'''
    classes:       {num_classes}
    epochs:        {epochs}
    batchsize:     {batchsize}
    lr:            {lr}
    use_bn:        {use_bn}
    Phi_sizes:     {Phi_sizes}
    F_sizes:       {F_sizes}
        ''')


    input_shape_for_one_event=train_set[0][0].shape
    feature_dim=input_shape_for_one_event[0]


    train_sampler=data.DistributedSampler(train_set)
    val_sampler=data.DistributedSampler(val_set)
    
    train_set=data.DataLoader(train_set,batch_size=batchsize,shuffle=False,sampler=train_sampler)
    val_set=data.DataLoader(val_set,batch_size=batchsize,shuffle=False,sampler=val_sampler)
    test_set=data.DataLoader(test_set,batch_size=batchsize,shuffle=False)


    net=my_PFN.ParticleFlowNetwork(num_classes=num_classes, input_dims=feature_dim, Phi_sizes=Phi_sizes,
                                F_sizes=F_sizes,use_bn=use_bn)
    net.apply(my_tools.initialize_pfn)
    loss=nn.CrossEntropyLoss(reduction='none') #none is essential！
    loss.to(local_rank)
    net.to(local_rank)

    if local_rank==0:
        writer=SummaryWriter(f'log_tensorboard/log_{suffix}')
        print(f'TensorBoard log path is {os.getcwd()}/log_tensorboard/log_{suffix}')
        writer.add_graph(net,input_to_model=torch.rand(1,*input_shape_for_one_event).to(local_rank))
        print(summary(net,input_size=input_shape_for_one_event))

    net=nn.parallel.DistributedDataParallel(net,device_ids=[local_rank])

    optimizer=torch.optim.NAdam(net.parameters(),lr=lr)#After the  "net.to(device=device)"


    if local_rank==0:
        torch.cuda.synchronize()
        start=time.time()


    for epoch in range(epochs):
        train_set.sampler.set_epoch(epoch) #Essential!!! Otherwize each GPU only get same ntuples every epoch
        #train_sampler.set_epoch(epoch)    #Same as the above, both are correct
        loss_train,acc_train=my_tools.train_procedure_in_each_epoch(net,train_set,loss,optimizer,local_rank)
        loss_val,acc_val=my_tools.evaluate_accuracy(net,loss,val_set,test=False)

        if local_rank==0:
            writer.add_scalars('loss_and_acc_in_train_and_val',{'loss_train':loss_train,
                                                                'acc_train':acc_train,
                                                                'loss_val':loss_val,
                                                                'acc_val':acc_val},epoch)
            print(f'''epoch: {epoch} | acc_train: {acc_train:.3f} | acc_val: {acc_val:.3f} | loss_train: {loss_train:.3f} | loss_val: {loss_val:.3f} | 
            ''')

    if local_rank==0:
        torch.cuda.synchronize()
        end=time.time()
        print('Total time in training is ',end-start)

    if local_rank==0:
        my_tools.save_net(net,suffix)
        score,true_label,loss_test,acc_test=my_tools.evaluate_accuracy(net,loss,test_set,test=True)
        print('{:-^80}'.format(f'loss in test is {loss_test:.3f}, accuracy in test is {acc_test:.3f},'))
        score_label=torch.argmax(score,dim=1)
        true_label=true_label.cpu().numpy()
        score_label=score_label.cpu().numpy()
        my_tools.plot_confusion_matrix(suffix,num_classes,true_label,score_label,classes=process)

    dist.destroy_process_group()
################################################################################################################################### 

if __name__=='__main__':
    my_tools.cheems()#means start...
    try:
        suffix=sys.argv[1]
    except:
        default_suffix=time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime())
        print(f"Seems like you didn't type a suffix, so the [tag_'localtime({default_suffix})'] has been used as the suffix") 
        suffix='tag_'+default_suffix
    gpu_nums=torch.cuda.device_count()
    print(f"The number of GPU(s) is {gpu_nums}!")
    #########################################################################################################
    process=['Hbb','Hcc', 'Hgg', 'Hww', 'Hzz', 'Pll', 'Pww_l', 'Pzz_l', 'Pzz_sl']
    num_data_each_class=300_00
    train_val_test=[0.8,0.1,0.1]
    #########################################################################################################
    fimename=[i +'.root' for i in process]
    dataset=my_pfn_load.load(filename=fimename,num_data=num_data_each_class)
    train_set,val_set,test_set=data.random_split(dataset=dataset,lengths=train_val_test)
    print(f'''
    suffix:        {suffix}
    gpu_nums:      {gpu_nums}
    process:       {process}
    tra_val_test:  {train_val_test}
    numdata_ec     {num_data_each_class}
    ''')

    mp.spawn(main,args=(suffix ,process, train_set, val_set, test_set), nprocs=torch.cuda.device_count())
    print('Program Is Over !!!')









