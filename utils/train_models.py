import numpy as np

from PCaNet_Submit.utils.dataset import trainloader,testloader,valloader
import os
import torch
import torch.optim as optim
import torch.nn as nn
from PCaNet_Submit.utils.evaluation import evaluate
from torch.optim.lr_scheduler import StepLR

def calc_loss(pred, target):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred,target)
    return loss

def train_one_epoch(model,epoch,optimizer,device,train_data):
    loss_sum = 0
    acc = 0
    num = 0
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device)[:,None,:,:]
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('output:',outputs.shape)
        loss = calc_loss(outputs,target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        delta = torch.argmax(outputs,dim=1)-torch.argmax(target,dim=1)
        acc += len(torch.argwhere(delta==0))
        num += inputs.shape[0]
    print('[%d]-th train loss:%.3f' % (epoch + 1, loss_sum))
    print('[%d]-th train acc:%.3f' % (epoch + 1, acc/num))

def test_one_epoch(model,epoch,device,test_data):
    loss_sum = 0
    acc = 0
    num = 0
    eva = evaluate(cls_num=3)
    model.eval()
    for idx,(inputs,target) in enumerate(test_data):
        inputs = inputs.float().to(device)[:,None,:,:]
        target = target.to(device)
        outputs = model(inputs)
        delta = torch.argmax(outputs,dim=1)-torch.argmax(target,dim=1)
        acc += len(torch.argwhere(delta==0))
        num += inputs.shape[0]
        eva.calculation(target,outputs)
    print('[%d]-th test acc:%.3f' % (epoch + 1, acc/num))
    eva.eval()
    return acc/num

def val_one_epoch(model,epoch,device,val_data):
    loss_sum = 0
    acc = 0
    num = 0
    eva = evaluate(cls_num=3)
    model.eval()
    for idx,(inputs,target) in enumerate(val_data):
        inputs = inputs.float().to(device)[:,None,:,:]
        target = target.to(device)
        outputs = model(inputs)
        delta = torch.argmax(outputs,dim=1)-torch.argmax(target,dim=1)
        acc += len(torch.argwhere(delta==0))
        num += inputs.shape[0]
        eva.calculation(target,outputs)
    print('[%d]-th val acc:%.3f' % (epoch + 1, acc/num))
    eva.eval()
    return acc/num

def train(epoches, model,lr, device, save_path,
           argument = True,data_path='../Data/BgRemoved/center1/',
          dataset_info_path='dataset_seg_info_file_dlrb_1.npy',):

    train_data = trainloader(batch_size=16, dataset_info_path = 'dataset_seg_info_file_da_dlrb.npy')
    train_data_o = trainloader(batch_size =16, dataset_info_path=dataset_info_path,headpath=data_path)
    # test_data = testloader(batch_size = 16, dataset_info_path = 'dataset_seg_info_file_da_dlrb.npy')
    test_data_o = testloader(batch_size = 16, dataset_info_path = dataset_info_path,headpath=data_path)
    val_data_o = valloader(batch_size = 16, dataset_info_path = dataset_info_path,headpath=data_path)
    model = model.to(device)
    val_acc_saver = []
    train_acc_saver = []
    model_paras_saver = []
    for epoch in range(epoches):
        optimizer_a = optim.Adam(model.parameters(), lr=lr/10)
        optimizer_o = optim.Adam(model.parameters(), lr=lr)
        # scheduler_o = StepLR(optimizer=optimizer_o, step_size=10, gamma=0.96)
        # scheduler_a = StepLR(optimizer=optimizer_a, step_size=10, gamma=0.96)
        # test_one_epoch(embed_model, model, epoch,
        #                 device, test_data)

        val_acc_saver.append(val_one_epoch( model, epoch,
                       device, test_data_o))
        if epoch% 10 == 0: lr = lr*0.98
        if argument:
            train_one_epoch(model, epoch,
                            optimizer_a, device, train_data)
        train_one_epoch(model, epoch,
                        optimizer_o, device, train_data_o)
        if epoch % 1 == 0: torch.save(model.state_dict(), save_path)
    print(train_acc_saver)
    print(val_acc_saver)
    print(max(val_acc_saver))
    for i in range(len(val_acc_saver)):
        max_val = np.sort(val_acc_saver)[-i]
        max_idx = np.argsort(val_acc_saver)[-i]
        if train_acc_saver[max_idx] > 0.7:
            print('max val acc:',max_val)
            print('max train acc:',train_acc_saver[max_idx])
            model.load_state_dict(model_paras_saver[max_idx])
            break
    test_acc = test_one_epoch(model, 0, device, test_data_o)
    torch.save(model.state_dict(), save_path)

