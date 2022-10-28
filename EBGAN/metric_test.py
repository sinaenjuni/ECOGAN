from metric.inception_net import EvalModel
import torch
from utils.dataset import DataModule_

eval_model = EvalModel().to(torch.device('cuda:4'))

dm = DataModule_(path_train='/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train', batch_size=128)
dm.setup('train')

for batch in dm.train_dataloader():
    with torch.no_grad():
        img, label = batch[0].to(torch.device('cuda:4')), batch[1].to(torch.device('cuda:4'))

        output = eval_model.get_outputs(img, True)
        print(output[0].size())





