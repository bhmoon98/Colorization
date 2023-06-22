import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    start = int(input("START INDEX: "))
    end = int(input("END INDEX: "))

    results = pd.read_csv('results.csv') 
    results = results[results['result'] == 0].reset_index(drop = True)

    print(results['result'])
    for i in range(start, end):
        if results['result'][i] == 1:
           continue
    
        color, gray = get_images(results['label'][i])

        color, gray = cv2.imread(color), cv2.imread(gray)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        def on_press(event):
            nonlocal results, i
            if event.key == '1':
                results['result'][i] = 'a'
                plt.close()
            elif event.key == '2':
                results['result'][i] = 'b'
                plt.close()
            elif event.key == '3':
                results['result'][i] = 'c'
                plt.close()
            elif event.key == '4':
                results['result'][i] = 'd'
                plt.close()
            elif event.key == 'j':
                results['result'][i] = 1
            elif event.key == 'l':
                print('save')
                results.to_csv('E:/PEA/program/fast/Python/Colorizer/src/data/synapseimaging-datapreprocessing/results.csv', index=False)

        fig = plt.figure(figsize=(10, 5))
        fig.canvas.mpl_connect('key_press_event', on_press)

        plt.subplot(1, 2, 1)
        plt.imshow(color)
        plt.xlabel(f"Image {i+1} / {len(results)} Label {results['label'][i]}")

        plt.subplot(1, 2, 2)
        plt.imshow(gray)
        plt.xlabel(f"PSNR: {results['psnr'][i]}, SSIM: {results['ssim'][i]}")

        plt.show()

        # Checkpointing
        if i % 10 == 0:
            results.to_csv('results.csv', index=False)


def get_images(label):
    if os.path.exists(f"../../dataset/Color/test/{label}_4.png"):
        color = f"../../dataset/Color/test/{label}_4.png"
        gray = f"../../dataset/Gray/test/{label}_2.jpg"
    elif os.path.exists(f"../../dataset/Color/train/{label}_4.png"):
        color = f"../../dataset/Color/train/{label}_4.png"
        gray = f"../../dataset/Gray/train/{label}_2.jpg"
    elif os.path.exists(f"../../dataset/Color/val/{label}_4.png"):
        color = f"../../dataset/Color/val/{label}_4.png"
        gray = f"../../dataset/Gray/val/{label}_2.jpg"
    else:
        print('Error File_path not exists')
    return (color, gray)


def load_checkpoint():
    try:
        with open("checkpoint.txt", "r") as f:
            result = f.read()

            li = []
            for ch in result:
                if ch == '1':
                    li.append(1)
                elif ch == '0':
                    li.append(0)
                else:
                    continue

            return li 
    except:
        return []




if __name__ == '__main__':
    main()
