nohup python3 -u /home/dblab/git/ECOGAN/src/eval_cls.py \
--model resnet18 \
--gpus 6 \
--data_seq 6 \
--lr 0.001 \
--steps 100000 \
--batch_size 128 > log/cls_test6.log &