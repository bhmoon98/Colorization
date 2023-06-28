import os, sys, cv2
sys.path.append("/mnt/e/Program/Python/Colorization")



if __name__=="__main__":
    img_color = cv2.imread("dataset/TM0/Color/001_001_1_4.png", cv2.IMREAD_COLOR)
    img_gray = cv2.imread("dataset/TM0/Gray/001_001_1_2.jpg", cv2.IMREAD_COLOR)
    
    img_color = cv2.resize(img_color, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    img_gray = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2LAB)
    
    img_gray[:, :, 0] = img_gray[:, :, 0]*0.6
    
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite('gray.png',img_color)
    