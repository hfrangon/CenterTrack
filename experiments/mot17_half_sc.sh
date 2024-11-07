export PYTHONPATH=$(pwd)

cd src
# train
python main.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halftrain --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --batch_size=2 --load_model ../models/crowdhuman.pth
# val
python test.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume
cd ..