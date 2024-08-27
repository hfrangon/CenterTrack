mkdir -p ../../data/mot17
cd ../../data/mot17
wget https://motchallenge.net/data/MOT17.zip #这一步是下载
unzip MOT17.zip #解压
rm MOT17.zip #删除源文件
mkdir annotations
cd ../../src/tools/
python convert_mot_to_coco.py
python convert_mot_det_to_results