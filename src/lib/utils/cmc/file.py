import numpy as np

import json
class GmcFile:
    @staticmethod
    def apply(img_info, trainval):
        if not trainval:
            filePath = r'../data/Ecc/MOT17_ECC_test.json'
        else:
            filePath = r'../data/Ecc/MOT17_ECC_val.json'
        "MOT17-01-DPM/img1/000001.jpg"
        img_info = img_info.split('/')
        seqName = img_info[0]
        imgId = str(int(img_info[2][:-4]))
        eye = np.eye(3)
        with open(filePath, 'r') as file:
            gmcFile = json.load(file)
        if seqName in gmcFile and imgId in gmcFile[seqName]:
            matrix = np.array(gmcFile[seqName][imgId])
            dist = np.linalg.norm(eye - matrix)
            if dist < 100:
                return matrix
        return eye
