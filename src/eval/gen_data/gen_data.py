import torch
from gen_model import Generator
import config
from tqdm import tqdm

contraGAN_cfgs = config.Configurations("/home/dblab/git/ECOGNA/src/eval/ContraGAN_div4.yaml")
ECOGAN_cfgs = config.Configurations("/home/dblab/git/ECOGNA/src/eval/OurGAN_div4.yaml")


contraGAN_path = "/home/dblab/git/PyTorch-StudioGAN/save_files/imbcifar10/contra_loss_div4/checkpoints/imbcifar10_contra_loss_div4-ContraGAN_div4-train-2022_08_29_07_16_19/model=G-best-weights-step=42000.pth"

ECOGAN_path = "/home/dblab/git/PyTorch-StudioGAN/save_files/imbcifar10/our_loss_div4/checkpoints/imbcifar10_our_loss_div4-OurGAN_div4-train-2022_08_26_16_57_30/model=G-best-weights-step=96000.pth"

contraGAN_weight = torch.load(contraGAN_path)
ECOGAN_weight = torch.load(ECOGAN_path)


contraGAN = Generator(z_dim=contraGAN_cfgs.MODEL.z_dim,
                      g_shared_dim=contraGAN_cfgs.MODEL.g_shared_dim,
                      img_size=contraGAN_cfgs.DATA.img_size,
                      g_conv_dim=contraGAN_cfgs.MODEL.g_conv_dim,
                      apply_attn=contraGAN_cfgs.MODEL.apply_attn,
                      attn_g_loc=contraGAN_cfgs.MODEL.attn_g_loc,
                      g_cond_mtd=contraGAN_cfgs.MODEL.g_cond_mtd,
                      num_classes=contraGAN_cfgs.DATA.num_classes,
                      g_init=contraGAN_cfgs.MODEL.g_init,
                      g_depth=contraGAN_cfgs.MODEL.g_depth,
                      mixed_precision=False,
                      MODULES=contraGAN_cfgs.MODULES,
                      MODEL=contraGAN_cfgs.MODEL)
contraGAN.eval()
contraGAN.load_state_dict(contraGAN_weight['state_dict'])


ECOGAN = Generator(z_dim=ECOGAN_cfgs.MODEL.z_dim,
                      g_shared_dim=ECOGAN_cfgs.MODEL.g_shared_dim,
                      img_size=ECOGAN_cfgs.DATA.img_size,
                      g_conv_dim=ECOGAN_cfgs.MODEL.g_conv_dim,
                      apply_attn=ECOGAN_cfgs.MODEL.apply_attn,
                      attn_g_loc=ECOGAN_cfgs.MODEL.attn_g_loc,
                      g_cond_mtd=ECOGAN_cfgs.MODEL.g_cond_mtd,
                      num_classes=ECOGAN_cfgs.DATA.num_classes,
                      g_init=ECOGAN_cfgs.MODEL.g_init,
                      g_depth=ECOGAN_cfgs.MODEL.g_depth,
                      mixed_precision=False,
                      MODULES=ECOGAN_cfgs.MODULES,
                      MODEL=ECOGAN_cfgs.MODEL)
ECOGAN.eval()
ECOGAN.load_state_dict(ECOGAN_weight['state_dict'])


def denorm(img):
    img = img * 0.5 + 0.5
    img = img * 255.
    img = img.astype(np.uint8)
    return img


from pathlib import Path
from PIL import Image
import numpy as np


images, labels = [], []
train_path = Path("/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train")
for path_ in list(train_path.glob('*/*.JPEG')):
    cls = path_.parts[-2]
    print(cls, path_)
    img = np.array(Image.open(path_))
    images.append(img)
    labels.append(int(cls[-1]))
images = np.array(images)
labels = np.array(labels)

classes, num_per_classes = np.unique(labels, return_counts=True)
num_classes = len(classes)
num_samples = len(images)
maximum_classes = num_per_classes[np.argmax(num_per_classes)]
num_mask_samples = maximum_classes - num_per_classes


save_path = Path("./save_files/ECOGAN_restore_data/")

for target_labels_, num_data_ in enumerate(num_mask_samples):
    print(target_labels_, num_data_)
    path = save_path / f"{target_labels_}"
    path.mkdir(parents=True, exist_ok=True)

    for i_ in tqdm(range(num_data_)):
        z = torch.randn(1, contraGAN_cfgs.MODEL.z_dim)
        labels = torch.tensor((target_labels_,)).to(torch.long)
        gened_img = ECOGAN(z, labels)
        gened_img = gened_img.squeeze().permute(1,2,0)
        gened_img = denorm(gened_img.detach().cpu().numpy())
        gened_img = Image.fromarray(gened_img)
        gened_img.save(path / f"{target_labels_}_{i_}.JPEG")



#
# EBGAN_gened_data_path = "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data"
# EBGAN_gened_data = ImageFolder(EBGAN_gened_data_path, transform=Compose([ ToTensor(),
#                                                                      Normalize((0.5, 0.5, 0.5),
#                                                                               (0.5, 0.5, 0.5))]))
#
# ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
# ori_img_dataset = ImageFolder(ori_imb_data_path, transform=Compose([ ToTensor(),
#                                                                      Normalize((0.5, 0.5, 0.5),
#                                                                               (0.5, 0.5, 0.5))]))
#
# concat_dataset = ConcatDataset([ori_img_dataset, EBGAN_gened_data])
#
# ori_img_loader = DataLoader(ori_img_dataset, batch_size=64)
#
# batch = iter(ori_img_dataset).__next__()
#
#
# for i, batch in enumerate(concat_dataset):
#     print(i)
#

