import os, sys, csv, cv2
sys.path.append("/mnt/e/Program/Python/Colorization")
import numpy as np



def main():
    ab = np.load("./utils/gamut_pts.npy")
    imgs_dir = "../../../dataset/TM2/"
    lab = []
    for l in range(0, 101, 10):
        for a, b in ab:
            lab.append([l, a, b])
    counts = np.zeros(len(lab))
    imgs_list = os.listdir(imgs_dir)
    n = 0
    for img_path in imgs_list:
        img_bgr = cv2.imread(os.path.join(imgs_dir, img_path), cv2.IMREAD_COLOR)
        img_bgr = cv2.resize(img_bgr,(256, 256), interpolation=cv2.INTER_CUBIC)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        for i in range(256):
            for j in range(256):
                l = round(img_lab[i][j][0]/256*100, -1)
                a = round((img_lab[i][j][1]-128), -1)
                b = round(img_lab[i][j][2]-128, -1)
                counts[lab.index([l, a, b])] += 1
                
        n = n+1
        print(n)
        
    counts = counts/sum(counts)
    
    print(counts)
    
    lab = np.array(lab)
    counts = np.array(counts)
    np.save("./utils/gamut_pts_lab.npy", lab)
    np.save("./utils/gamut_probs_lab.npy", counts)
    temp = np.load("./utils/gamut_pts_lab.npy")
    print(temp)
    print("---------------------------")
    temp = np.load("./utils/gamut_probs_lab.npy")
    print(temp)
    
    

if __name__=="__main__":
    main()