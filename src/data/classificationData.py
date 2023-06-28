import cv2, shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

RESULT_PATH = 'E:/Program/Python/Colorization/result/_csv/results_TM2_L.csv'
COLOR_PATH = 'E:/Program/Python/Colorization/dataset/TM2/Color'
GRAY_PATH = 'E:/Program/Python/Colorization/dataset/TM0/Gray'

REMAIN_KEY = 'j'
REMOVE_KEY = 'k'
UNDO_KEY = 'l'


def main():
    df = pd.read_csv(RESULT_PATH)

    if not 'result' in df.columns:
        df['result'] = 0

    start_index = int(input('START INDEX: '))
    end_index = int(input('END INDEX: '))

    i= start_index
    print(i, min(end_index, len(df)))
    while (i<min(end_index, len(df))):
        label = df['label'][i]

        color_image = cv2.resize(cv2.imread(get_color_path(label)), (256, 256))
        gray_image = cv2.resize(cv2.imread(get_gray_path(label)), (256, 256))
        
        imgs = np.concatenate((color_image, gray_image), axis = 1)
        
        cv2.imshow('image', imgs)
        
        while True:
            key = cv2.waitKey()
            if key == ord(REMAIN_KEY):
                df.loc[i ,'result'] = 1
                break
            elif key == ord(REMOVE_KEY):
                df.loc[i ,'result'] = 0
                break
            elif key == ord(UNDO_KEY):
                i = i-2
                break
        if i%100 == 0:
            shutil.copy('E:/Program/Python/Colorization/result/_csv/results_TM2_L.csv', 'E:/Program/Python/Colorization/result/_csv/backup.csv')
        
        df.to_csv(RESULT_PATH, index=False)
        print(i)
        i+=1
    df.to_csv(RESULT_PATH, index=False)


def get_color_path(label: str) -> str:
    return f"{COLOR_PATH}/{label}_4.png"


def get_gray_path(label: str) -> str:
    return f"{GRAY_PATH}/{label}_2.jpg"


def get_ssim_diff_image(color_image: np.ndarray, gray_image: np.ndarray) -> np.ndarray:
    color_image = cv2.resize(color_image, gray_image.shape)

    (_, diff) = ssim(color_image, gray_image, full=True)
    
    return diff


if __name__=='__main__':
    main()