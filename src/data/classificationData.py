import os
import sys
import shutil
import random



if __name__=="__main__":
    img_dir = sys.argv[1]
    img_color_dir = os.path.join(img_dir, 'Color')
    img_gray_dir = os.path.join(img_dir, 'Gray')
    
    img_color_list = os.listdir(img_color_dir)
    img_color_list.sort()
    img_gray_list = os.listdir(img_gray_dir)
    img_gray_list.sort()
    
    imgs_list = list(zip(img_color_list, img_gray_list))
    print(len(imgs_list))
    
    train = random.sample(imgs_list, 40000)
    train_dir = os.path.join(img_dir, 'train')
    train_color_dir = os.path.join(train_dir, 'color')
    train_gray_dir = os.path.join(train_dir, 'gray')
    os.mkdir(train_dir)
    os.mkdir(train_color_dir)
    os.mkdir(train_gray_dir)
    
    for imgs in train:
        imgs_list.remove(imgs)
        img_color = imgs[0]
        img_gray = imgs[1]
        
        shutil.move(os.path.join(img_color_dir, img_color), os.path.join(train_color_dir, img_color))
        shutil.move(os.path.join(img_gray_dir, img_gray), os.path.join(train_gray_dir, img_gray))
        train = random.sample(imgs_list, 5000)
        
    test_dir = os.path.join(img_dir, 'test')
    train_color_dir = os.path.join(train_dir, 'color')
    train_gray_dir = os.path.join(train_dir, 'gray')
    os.mkdir(train_dir)
    os.mkdir(train_color_dir)
    os.mkdir(train_gray_dir)
    
    for imgs in train:
        imgs_list.remove(imgs)
        img_color = imgs[0]
        img_gray = imgs[1]
        
        shutil.move(os.path.join(img_color_dir, img_color), os.path.join(train_color_dir, img_color))
        shutil.move(os.path.join(img_gray_dir, img_gray), os.path.join(train_gray_dir, img_gray))
        
   
        