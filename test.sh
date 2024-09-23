export PYTHONPATH=$(pwd)

cd src

python test.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.3 --pre_thresh 0.5 --load_model ../models/model_last-Copy1.pth --gpus 0 --init

cd ..

