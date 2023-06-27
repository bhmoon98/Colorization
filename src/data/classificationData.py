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

    i = start_index
    print(i, min(end_index, len(df)))
    while (i<min(end_index, len(df))):
        def on_key_press(event):
            nonlocal df, i
            print(i)
            if event.key == REMAIN_KEY:
                df.loc[i ,'result'] = 1
                plt.close()
            elif event.key == REMOVE_KEY:
                df.loc[i ,'result'] = 0
                plt.close()
            elif event.key == UNDO_KEY:
                if i < 1:
                    return
                i = i-2
                plt.close()
            else:
                return

        fig = plt.figure(num=0)
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        label = df['label'][i]

        color_image = cv2.imread(get_color_path(label))
        gray_image = cv2.imread(get_gray_path(label))

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        plt.subplot(121)
        plt.xlabel('COLOR IMAGE')
        plt.imshow(color_image, cmap='gray')

        plt.subplot(122)
        plt.xlabel('GRAY IMAGE')
        plt.imshow(gray_image, cmap='gray')

        plt.title(f"SSIM={df['ssim'][i]:.1f} | PSNR={df['psnr'][i]:.1f}")

        plt.show()

        df.to_csv(RESULT_PATH, index=False)
        if i%100 == 0:
            shutil.copyfile('E:/Program/Python/Colorization/result/_csv/results_TM2_L.csv', 'E:/Program/Python/Colorization/result/_csv/backup.csv')
            
        i+=1


def get_color_path(label: str) -> str:
    return f"{COLOR_PATH}/{label}_4.png"


def get_gray_path(label: str) -> str:
    return f"{GRAY_PATH}/{label}_2.jpg"


if __name__=='__main__':
    main()

        
   
        