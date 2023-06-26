import os, sys, csv, shutil
sys.path.append("/mnt/e/Program/Python/Colorization")



def cut_thV(GT_dir, imgs_dir, output_file):
    if not(os.path.exists(imgs_dir)):
        os.makedirs(imgs_dir)
    else:
        shutil.rmtree(imgs_dir)
        os.makedirs(imgs_dir)
    n = 0
    with open(output_file, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for line in reader:
            if not(line[3]==0):
                n+=1
                print(line[0])
                shutil.copy(os.path.join(GT_dir, line[0]+'_4.png'), os.path.join(imgs_dir, line[0]+'_4.png'))
    print(n)



if __name__=='__main__':
    GT_dir = 'dataset/TM2/Color'
    imgs_dir = 'dataset/TM2_THV/Color'
    output_file = "result/_csv/results_TM2.csv"
    cut_thV(GT_dir, imgs_dir, output_file)