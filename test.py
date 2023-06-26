import sys, os, csv, cv2
sys.path.append("/mnt/e/Program/Python/Colorization")



if __name__=='__main__':
    test = cv2.imread('dataset/Colorization/Color/001_001_1_4.png', cv2.IMREAD_COLOR)
    cv2.imshow('test', test)
    cv2.waitKey(0)