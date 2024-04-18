import os.path
import torch
import models,dataset,train_models


def demo(embed_name,backbone_name,embedding_dim,spectral_dim,
         head_path = '../WorkSpace1/', device='cuda:3',batch_size = 16,
         num_works=2,dataset_info_path='dataset_seg_info_file'):

    os.makedirs(head_path, exist_ok=True)
    if backbone_name in ['Resnet1d_']:
        model = models.Resnet1d(inplanes=embedding_dim)
    elif backbone_name in ['Resnet2d_without_embed_']:
        model = models.resnet50_without_embed_(False,num_classes=3)
    model_save_path =  head_path+str(backbone_name)+embed_name+str(spectral_dim)+'_'+str(embedding_dim)+'.pt'
    if os.path.exists(model_save_path):
        # print(model_save_path)
        print(model_save_path + ' is existed')
        model.load_state_dict(torch.load(model_save_path))
        train_models.train(epoches=40, model=model, lr=1e-4, device='cuda:1',
                           save_path=model_save_path,
                           dataset_info_path=dataset_info_path,argument=True)
    else:
        print(model_save_path+' is not existed')
        print('Training_122 ' + backbone_name + ' on ' + device)
        batch_size = batch_size
        train_models.train(epoches=40, model=model,  lr=1e-4, device='cuda:1',
                           save_path=model_save_path,
                           dataset_info_path=dataset_info_path,argument=True)

if __name__ == '__main__':

    spectral_dim, embedding_dim = 724,20
    device = 'cuda:0'
    data_path = '../Data/BgRemoved/center1/'
    backbone_name_set = ['Resnet2d_without_embed_','cnn_base_mini_','cnn_base_']
    # embedding_dim_set = [20,40,80,160,320]
    # backbone_name_set = ['cnn_base_mini_', 'cnn_base_']
    embedding_name_set = ['raw_','embed_']
    # embedding_dim_set = [320,200,160,80,40,20]
    embedding_dim_set = [724]
    for embedding_dim in embedding_dim_set:
        for embedding_name in embedding_name_set:
            for backbone_name in backbone_name_set:
                if os.path.exists('log/'+str(embedding_dim)+'_'+embedding_name+backbone_name+'.txt'): pass

                else:

                    demo(embed_name=embedding_name, backbone_name=backbone_name,
                         embedding_dim=embedding_dim, spectral_dim=spectral_dim,
                         device=device, batch_size=64, head_path='../WorkSpace_dlrb_r1/',
                         dataset_info_path='dataset_seg_info_file_dlrb_1.npy')

