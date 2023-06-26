import sys, os, csv
sys.path.append("/mnt/e/Program/Python/Colorization")
import cv2



def _save_img_list(img_dir: str) -> list:
    img_list = os.listdir(img_dir)
    save_file = img_dir.split('/')[-1]+'_list.csv'
    header = ['gray', 'color']
    with open(save_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        i=0
        while i<len(img_list):
            writer.writerow([img_list[i+1], img_list[i+3]])
            i+=4 
    
        


def _get_img_list(img_dir: str, save_file: str) -> list:
    img_list = []
    with open(save_file, "r") as csv_file:
        reader = csv.writer(csv_file)
        for line in reader:
            temp = {'gray': os.path.join(img_dir, line[0]), 
                    'color': os.path.join(img_dir, line[1])}
            img_list.append(temp)
    
    return img_list


def _template_matching(img_list: list, save_dir: str):
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    for imgs in img_list:
        img_gray = cv2.imread()
    pass
    
    
    
if __name__=='__main__':
    _save_img_list(f'dataset/raw')