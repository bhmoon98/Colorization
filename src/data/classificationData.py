import cv2, shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

RESULT_PATH = 'E:/Program/Python/Colorization/result/_csv/results_TM3.csv'
COLOR_PATH = 'E:/Program/Python/Colorization/dataset/data3/data3_TM3'
GRAY_PATH = 'E:/Program/Python/Colorization/dataset/data3/Gray'

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
        maxVal = df['maxVal'][i]
        color_image = cv2.resize(cv2.imread(get_color_path(label)), (256, 256))
        gray_image = cv2.resize(cv2.imread(get_gray_path(label)), (256, 256))
        diff_image = get_ssim_diff_image(color_image, gray_image)
        rec = np.zeros((30, 256*3, 3), np.uint8) + 255
        imgs = np.concatenate((color_image, gray_image, diff_image), axis=1)
        imgs = np.concatenate((rec, imgs), axis=0)
        font = cv2.FONT_ITALIC
        imgs = cv2.putText(imgs, label, (0, 20), font, 0.5, (0 ,0 ,0), 1)
        imgs = cv2.putText(imgs, str(maxVal), (500, 20), font, 0.5, (0 ,0 ,0), 1)
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
            shutil.copy(RESULT_PATH, 'E:/Program/Python/Colorization/result/_csv/backup.csv')
        
        df.to_csv(RESULT_PATH, index=False)
        print(i)
        i+=1
    df.to_csv(RESULT_PATH, index=False)


def get_color_path(label: str) -> str:
    return f"{COLOR_PATH}/{label}_4.png"


def get_gray_path(label: str) -> str:
    return f"{GRAY_PATH}/{label}_2.png"


def get_ssim_diff_image(color_image: np.ndarray, gray_image: np.ndarray) -> np.ndarray:
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    color_image = cv2.resize(color_image, gray_image.shape)

    kernel = np.ones((3, 3), np.float32) / 25
    denoised = color_image
    denoised = np.where(color_image < 170, 0, 255)
    gray_image = np.where(gray_image < 170, 0, 255)
    denoised = denoised.astype(np.float32)

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int8)

    kernel = np.ones(9).reshape(3, 3) / 9
    # denoised = cv2.filter2D(denoised, -1, kernel)

    diff = np.absolute(denoised - gray_image)
    diff = 255 - diff
    
    diff = cv2.merge([diff, diff, diff])
    diff = diff.astype(np.uint8)
    return diff


if __name__=='__main__':
    main()