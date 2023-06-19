import cv2
import os, glob
import numpy as np



if __name__ == '__main__':
    sharpening_fliter_1 = np.array([[-1, -1, -1], 
                                  [-1, 9, -1], 
                                  [-1, -1, -1]])
    sharpening_fliter_2 = np.array([[-1, -1, -1, -1, -1], 
                                  [-1, 2, 2, 2, -1],
                                  [-1, 2, 8, 2, -1],
                                  [-1, 2, 2, 2, -1], 
                                  [-1, -1, -1, -1, -1]])/8.0
    sharpening_fliter_3 = np.array([[-1, -1, -1, -1, -1, -1, -1], 
                                  [-1, 2, 2, 2, 2, 2, -1],
                                  [-1, 2, 4, 4, 4, 2, -1],
                                  [-1, 2, 4, 8, 4, 2, -1],
                                  [-1, 2, 4, 4, 4, 2, -1], 
                                  [-1, 2, 2, 2, 2, 2, -1],
                                  [-1, -1, -1, -1, -1, -1, -1]])
    sum=0
    for i in range(len(sharpening_fliter_3)):
        for j in range(len(sharpening_fliter_3[i])):
            sum += sharpening_fliter_3[i][j]
    sharpening_fliter_3 = sharpening_fliter_3/float(sum)
    sharpening_fliter_4 = np.array([[1, 1, 1], 
                                  [1, -7, 1], 
                                  [1, 1, 1]])
    
    data_dir = 'dataset/test'
    save_dir = 'dataset/blur_test_1'
    list = glob.glob(os.path.join(data_dir, '*.*'))
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    list.sort()
    list_len = len(list)
    i = 0
    print('start:', list_len)
    for dir in list:
        img = cv2.imread(dir)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.filter2D(img_rgb, -1, sharpening_fliter_3)
        img_rgb = cv2.filter2D(img_rgb, -1, sharpening_fliter_1)
        sharpened_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, '1_'+dir.split('/')[-1]), sharpened_image)
        if i==0:
            break
            pass
        i+=1
        
    
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l_channel = lab[:, :, 0]
    # laplacian = cv2.Laplacian(l_channel, cv2.CV_64F)
    # sharpened_l_channel = np.uint8(np.clip(l_channel - laplacian, 0, 255))
    # lab[:, :, 0] = sharpened_l_channel
    # sharpened_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l_channel = img_lab[:, :, 0]
    # img_blur = cv2.blur(l_channel, (5, 5))
    # sharpened_l_channel = l_channel* 2 -img_blur
    # img_lab[:, :, 0] = sharpened_l_channel
    # sharpened_image = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
