import os
import torch
from PCaNet_final_Submit.utils import models

def test_script(test_dir = '../center1_test_data', device = 'cpu'):
    # laad the model
    model = models.resnet50_without_embed_(False,num_classes=3)
    model_save_path =  '../Resnet2d_without_embed_raw_724_724.pt'
    if os.path.exists(model_save_path):
        print(model_save_path + ' is existed')
        model.load_state_dict(torch.load(model_save_path))
    model.to(device).eval()

    # load the data
    test_list = os.listdir(test_dir)
    with torch.no_grad():
        for item in test_list:
            data = torch.tensor(torch.load(os.path.join(test_dir, item)))[:200,:724].to(device)
            label = torch.tensor(int(item[1])).to(device)
            label = torch.nn.functional.one_hot(label-1, num_classes=3).float()
            data = torch.nn.functional.normalize(data, dim=1, p=2)
            data = data[None, None, :, :]
            # predict
            output = model(data)
            print('predict cls:', torch.argmax(output), 'true cls:', torch.argmax(label))


test_script(test_dir='../center1_test_data')