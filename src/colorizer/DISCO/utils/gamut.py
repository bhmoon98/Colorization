import os, sys, csv, cv2, argparse
sys.path.append("/mnt/e/Program/Python/Colorization")
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch



def argparser(parser):
    parser.add_argument('--data_dir', default='', type=str, help='Colorization dir')
    return parser



def main(data_dir):
    dir_types = ["train", "val", "test"]

    l = np.arange(0, 101, 10, dtype=np.int32)
    a_min = -110
    a_max = 110
    a = np.linspace(a_min, a_max, num=23, dtype=np.int32)
    b = np.linspace(a_min, a_max, num=23, dtype=np.int32)

    lab = np.array(np.meshgrid(l, a, b)).T.reshape(-1, 3)
    counts = np.zeros(lab.shape[0])

    n = 0
    imgs_len = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs_path_list = []
    dir_list = []
    for dir_type in dir_types:
        temp_dir = os.path.join(data_dir, dir_type)
        temp_dir = os.path.join(temp_dir, "Color")
        temp = os.listdir(temp_dir)
        imgs_len += len(temp)
        imgs_path_list.append(temp)
        dir_list.append(temp_dir)

    print(imgs_len)
    for imgs_list, imgs_dir in zip(imgs_path_list, dir_list):
        for img_path in imgs_list:
            img_bgr = cv2.imread(os.path.join(imgs_dir, img_path), cv2.IMREAD_COLOR)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

            h, w, _ = img_lab.shape

            img_lab_tensor = torch.from_numpy(img_lab.transpose((2, 0, 1))).to(device)
            l = (torch.round(img_lab_tensor[0] / 255 * 10) * 10).to(torch.int32)
            a = (torch.round((img_lab_tensor[1].to(torch.int32) - 128)/10)*10).to(torch.int32)
            b = (torch.round((img_lab_tensor[2].to(torch.int32) - 128)/10)*10).to(torch.int32)

            combined = torch.stack((l.flatten(), a.flatten(), b.flatten()), dim=1)
            unique_colors, counts_tensor = torch.unique(combined, return_counts=True, dim=0)

            unique_colors_np = unique_colors.cpu().numpy()
            counts_np = counts_tensor.cpu().numpy()

            for i in range(unique_colors_np.shape[0]):
                color = unique_colors_np[i]
                count = counts_np[i]
                index = np.where((lab == color).all(axis=1))[0][0]
                counts[index] += count

            n += 1
            print(f">> [{n}/{imgs_len}]")

    counts = counts / np.sum(counts)
    np.save("./utils/gamut_pts_lab.npy", lab)
    np.save("./utils/gamut_probs_lab.npy", counts)



if __name__ == "__main__":
    print("-------------make gamut-------------")
    parser = argparse.ArgumentParser()
    parser = argparser(parser)
    args = parser.parse_args()

    main(args.data_dir)
