import numpy as np
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from models import Generator
import importlib

dataset_ori = getattr(m, 'CIFAR10_LT')(is_extension=False)

for i in dataset_ori:
    print(i)


import h5py
base_path = Path('/shared_hdd/sin/gen/gen.h5py')




# api = wandb.Api()
# project = api.project('MYTEST0')
# project
#
# runs = api.runs(path="MYTEST0")
#
# for run in runs:
#     artifact_name = '/'.join(run.path) + ':v0'
#     print(artifact_name)
#     artifact = api.artifact(artifact_name)
#
# [i.path for i in runs]
#
# [i for i in api.artifact_type(['model'])]
#
#
# dir(runs[0])
# dir(runs[0].client)
#




m = importlib.import_module('utils.datasets')
batch_size = 128


api = wandb.Api()
runs = api.runs(path="MYTEST0")

with h5py.File(base_path, 'w') as hf:
    for idx, run in enumerate(runs):
        imgs_gen = []

        print(idx)
        # if idx != 0:
        #     break
        artifact_dir = run.logged_artifacts()[0].download()
        # artifact_dir = artifact.download()
        ch = torch.load(Path(artifact_dir) / "model.ckpt", map_location=torch.device('cuda'))
        g_ch = {'.'.join(k.split('.')[1:]): w for k, w in ch['state_dict'].items() if 'G' in k}
        G = Generator(**ch['hyper_parameters']).cuda()
        ret = G.load_state_dict(g_ch)

        id = run.id
        data_name = ch['hyper_parameters']['data_name']
        if 'imb_' in data_name:
            data_name = data_name.split('imb_')[1] + '_LT'
        print(data_name)

        dataset_ori = getattr(m, data_name)()
        classes_ori, num_classes_ori = np.unique(dataset_ori.targets, return_counts=True)

        num_classes_gen = (num_classes_ori.max() - num_classes_ori).astype(np.int32)
        gen_dict = dict(zip(classes_ori, num_classes_gen))
        labels = torch.tensor(np.concatenate([[k] * v for k, v in gen_dict.items()])).to(torch.long)

        hf.create_group(f'gened_data/{data_name}/{id}')

        pbar = tqdm(iterable=range(len(labels)))
        # gen dataset
        for idx in range(0, (len(labels) // batch_size) + 1):
            batch_label = labels[idx * batch_size: (idx + 1) * batch_size]
            z = torch.randn(batch_label.size(0), 128).cuda()
            img_gen = G(z, batch_label.cuda())
            img_gen = img_gen.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            imgs_gen.append(img_gen)
            pbar.update(len(batch_label))

        hf.create_dataset(f'gened_data/{data_name}/{id}/data', data=torch.cat(imgs_gen))
        hf.create_dataset(f'gened_data/{data_name}/{id}/targets', data=labels)

    # model_name = run.name
    # img_dim = ch['hyper_parameters']['img_dim']
    # num_classes = ch['hyper_parameters']['num_classes']
    # img_size = ch['hyper_parameters']['img_size']

    # print(model_name)
    # print(data_name)

f = h5py.File('/shared_hdd/sin/gen/gen.h5py', 'r')
f['gened_data'].keys()
f['gened_data']['Places_LT']['1vdnnadg'].keys()
f['gened_data']['Places_LT']['1vdnnadg']['targets']
f.close()

for i in tqdm(f['gened_data']['Places_LT']['1vdnnadg']['data']):
    pass


with h5py.File('/shared_hdd/sin/data.h5py', 'r') as file_object:

  dataset = file_object['img_align_celeba']
  image = numpy.array(dataset['6.jpg'])
  plt.imshow(image, interpolation='none')





classes = range(1, 10, 1)
rate = 1000
labels = torch.cat([torch.ones(1000) * cls for cls in classes]).to(torch.long)
imgs_gen = []
# gen_dict = {1:rate, 2:rate, 3:rate, 4:rate, 5:rate, 6:rate, 7:rate, 8:rate, 9:rate}
# label = torch.tensor(np.concatenate([[k] * v for k, v in gen_dict.items()]))


    # imgs = imgs.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)

    # for idx_, (img, cls) in enumerate(zip(imgs, batch_label)):
    #     # print(idx * batch_size + idx_)
    #     save_path = base_path / str(rate) / name / str(cls.item())
    #     if not save_path.exists():
    #         save_path.mkdir(exist_ok=True, parents=True)
    #     if img.size(0) == 1:
    #         img = img.squeeze()
    #         img = Image.fromarray(img.numpy())
    #     else:
    #         img = Image.fromarray(img.permute(1, 2, 0).numpy())
    #
    #     img.save(save_path / f'{idx * batch_size + idx_}.png')


