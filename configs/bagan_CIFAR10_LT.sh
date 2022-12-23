python3 /home/dblab/git/ECOGAN/src/training.py \
--model bagan \
--gpus 0 1 2 3 \
--data_name CIFAR10_LT \
--path save_files/ \
--epoch_ae 150 \
--steps 100000 \
--logger true \
--img_dim 3 \
--latent_dim 128 \
--lr 0.00005 \
--beta1 0.5 \
--beta2 0.999 \
--batch_size 128 \
--is_save true