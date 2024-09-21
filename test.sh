export PYTHONPATH=$(pwd)

cd src

python test.py tracking --exp_id mot17_half_sc --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/crowdhuman.pth

cd ..

