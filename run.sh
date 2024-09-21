export PYTHONPATH=$(pwd)
cd src

python main.py tracking --exp_id mot17_fulltrain --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --load_model ../models/dla_aff.pth --lr 0.00002 --batch_size 2
cd ..

