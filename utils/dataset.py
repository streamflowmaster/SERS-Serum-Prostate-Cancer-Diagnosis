import torch.utils.data as tud
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def read_dataset_info(info_path='dataset_info_file.npy'):
    info = np.load(info_path,allow_pickle = True).item()
    return info

class SERSdataset(tud.Dataset):

    def __init__(self,headpath='../Data/BgRemoved/center1/',type='test',dataset_info_path = 'dataset_seg_info_file_arb.npy'):
        self.headpath = headpath
        self.type = type
        info_dict = read_dataset_info(dataset_info_path)
        self.filelist = info_dict[type]

    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, item):
        if '.pt' in self.filelist[item][0]: data = (torch.load(self.headpath+self.filelist[item][0]))[:200,:724]
        data = torch.tensor(torch.load(self.headpath+self.filelist[item][0]))[:200,:724]
        label = torch.tensor(self.filelist[item][1])
        label = torch.nn.functional.one_hot(label, num_classes=3).float()
        data = torch.nn.functional.normalize(data,dim=1,p=2)
        return data,label
    def augumentation(self,x):
        noisem = torch.normal(1,0.025,x.shape)
        noisep = torch.normal(0,0.1,x.shape)
        x = x*noisem+noisep
        return x


class CAMdataset(tud.Dataset):

    def __init__(self,headpath='../Data/BgRemoved/center1/',type='test',dataset_info_path = 'dataset_seg_info_file_arb.npy'):
        self.headpath = headpath
        self.type = type
        info_dict = read_dataset_info(dataset_info_path)
        self.filelist = info_dict[type]

    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, item):
        if '.pt' in self.filelist[item][0]: data = (torch.load(self.headpath+self.filelist[item][0]))[:200,:724]
        data = torch.tensor(torch.load(self.headpath+self.filelist[item][0]))[:200,:724]
        label = torch.tensor(self.filelist[item][1])
        # print(self.filelist)

        label = torch.nn.functional.one_hot(label, num_classes=3).float()
        data = data.reshape(20, 10, -1).permute(2,0,1) # chw
        # if self.type=='train': data = self.augumentation(data)
        # print(self.filelist[item],data.shape)
        # internsity = torch.sqrt(torch.sum(torch.square(data),dim=0))
        # print(internsity.shape)
        # data = data/internsity
        data = torch.nn.functional.normalize(data,dim=0,p=2)
        return data,label,self.filelist[item][0]

    def augumentation(self,x):
        # x = x[:, torch.randperm(x.size(1))]
        noisem = torch.normal(1,0.025,x.shape)
        noisep = torch.normal(0,0.1,x.shape)
        x = x*noisem+noisep
        return x

class FeatScatteringDataset(tud.Dataset):

    def __init__(self,headpath='../Data/BgRemoved/center1/',type='test',dataset_info_path = 'dataset_seg_info_file_arb.npy'):
        self.headpath = headpath
        self.type = type
        # self.filelist = os.listdir(headpath+type+'/')
        info_dict = read_dataset_info(dataset_info_path)
        self.filelist = info_dict[type]
        # print(self.filelist)

    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, item):
        if '.pt' in self.filelist[item][0]: data = (torch.load(self.headpath+self.filelist[item][0]))[:200,:724]
        data = torch.tensor(torch.load(self.headpath+self.filelist[item][0]))[:200,:724]
        label = torch.tensor(self.filelist[item][1])
        # print(self.filelist)
        psa_pth = os.path.join(self.headpath,'PSA_data/',self.filelist[item][0][-13:])
        psa = torch.load(psa_pth)[0]
        # print(psa)
        label = torch.nn.functional.one_hot(label, num_classes=3).float()
        data = torch.nn.functional.normalize(data,dim=1,p=2)
        return data,label,psa
    def augumentation(self,x):
        # x = x[:, torch.randperm(x.size(1))]
        noisem = torch.normal(1,0.025,x.shape)
        noisep = torch.normal(0,0.1,x.shape)
        x = x*noisem+noisep
        return x

def testloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):
    sers = SERSdataset(type='test',dataset_info_path=dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers, batch_size=batch_size, shuffle=False)

def trainloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):
    sers = SERSdataset(type='train',dataset_info_path = dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers,batch_size=batch_size,shuffle=True)

def valloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):
    sers = SERSdataset(type='validation',dataset_info_path = dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers,batch_size=batch_size,shuffle=True)

def CAMtestloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):
    sers = CAMdataset(type='test',dataset_info_path=dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers, batch_size=batch_size, shuffle=True)

def CAMtrainloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):
    sers = CAMdataset(type='train',dataset_info_path = dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers,batch_size=batch_size,shuffle=True)

def FeatScatteringTestloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):

    sers = FeatScatteringDataset(type='test',dataset_info_path=dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers, batch_size=batch_size, shuffle=False)

def FeatScatteringTrainloader(batch_size,dataset_info_path = 'dataset_seg_info_file_dlrb.npy',headpath='../Data/BgRemoved/center1/'):
    sers = FeatScatteringDataset(type='train',dataset_info_path = dataset_info_path,headpath=headpath)
    return tud.DataLoader(dataset=sers,batch_size=batch_size,shuffle=True)

if __name__ == '__main__':
    loader = trainloader(1)
    channel = 724
    # loader = testloader(1)
    # loader = testloader(1,dataset_info_path='dataset_seg_info_file_arb.npy')
    colorbar = ['r', 'g', 'b']
    aver = torch.zeros((3, channel))

    count = torch.zeros((3,1))
    for idx, data in enumerate(loader):
        hsi, label = data
        print(hsi.shape)
        label = torch.argmax(label).item()
        aver[label ] += hsi[0].reshape(channel,-1).mean(1)
        count[label ] += 1
        print(aver.shape)
        print(count.shape)
        # plt.plot(hsi[0].reshape(583, -1)[1:], colorbar[label])
        # for i in range(200):
        #     plt.plot(hsi[0].reshape(channel,-1)[1:,i],colorbar[label])
        # plt.show()
        # plt.plot(aver,c=colorbar[label-1])
    plt.plot((aver / count).T)
    plt.show()
