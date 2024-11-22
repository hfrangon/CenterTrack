export PYTHONPATH=$(pwd)

cd src
# train
python main.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halftrain --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --batch_size=2 --load_model ../models/crowdhuman.pth --trainval

# val
python test.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --new_thresh 0.5 --out_thresh 0.2 --pre_thresh 0.5 --resume --load_model ../models/dla_aff_simple_60.pth --trainval

#val algorithm
python test.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --new_thresh 0.4 --out_thresh 0.2 --pre_thresh 0.4 --resume --load_model ../models/mot17_half.pth --arch dlav0_34 --trainval

#test
python test.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17test --ltrb_amodal --track_thresh 0.4 --new_thresh 0.5 --out_thresh 0.2 --pre_thresh 0.5 --resume --load_model ../models/dla_aff_simple_60.pth

cd ..