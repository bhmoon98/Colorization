import os, sys, csv, shutil
sys.path.append("/mnt/e/Program/Python/Colorization")



def make_pair(GT_dir, imgs_dir, output_file):
    pair_dir = 'pair/img'
    psnr = []
    ssim = []
    if os.path.exists(pair_dir):
        shutil.rmtree(pair_dir)
    os.makedirs(pair_dir)

    with open(output_file, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for line in reader:
            psnr.append(float(line[2]))
            ssim.append(float(line[5]))
            if (float(line[2])<25.5):
                print(line[0])
                shutil.copy(os.path.join(GT_dir, line[0][:-5]+'4.png'), os.path.join(pair_dir, line[0][:-5]+'4.png'))
                shutil.copy(os.path.join(imgs_dir,line[0]), os.path.join(pair_dir, line[0]))



if __name__=='__main__':
    GT_dir = 'dataset/TM2/Color'
    imgs_dir = 'result/UGATIT/TM2'
    output_file = "result/metrics_UGATIT_TM2.csv"
    make_pair(GT_dir, imgs_dir, output_file)