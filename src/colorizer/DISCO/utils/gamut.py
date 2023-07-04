import os, sys, csv, cv2
sys.path.append("/mnt/e/Program/Python/Colorization")
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch



def main():
    imgs_dir = "../../../dataset/RAW2/"
    lab = []
    counts = []
    imgs_list = os.listdir(imgs_dir)
    n = 0
    imgs_len = int(len(imgs_list) / 4)
    
    print("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for img_path in imgs_list:
        if img_path.split('_')[-1] == '4.png':
            img_bgr = cv2.imread(os.path.join(imgs_dir, img_path), cv2.IMREAD_COLOR)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

            h, w, _ = img_lab.shape

            img_lab_tensor = torch.from_numpy(img_lab.transpose((2, 0, 1))).to(device)
            l = torch.round(img_lab_tensor[0] / 255 * 100).to(torch.int32)
            a = torch.round(img_lab_tensor[1] - 128).to(torch.int32)
            b = torch.round(img_lab_tensor[2] - 128).to(torch.int32)

            combined = torch.stack((l.flatten(), a.flatten(), b.flatten()), dim=1)
            unique_colors, counts_tensor = torch.unique(combined, return_counts=True, dim=0)

            unique_colors_np = unique_colors.cpu().numpy()
            counts_np = counts_tensor.cpu().numpy()

            for i in range(unique_colors_np.shape[0]):
                color = unique_colors_np[i]
                count = counts_np[i]

                if list(color) not in lab:
                    lab.append(list(color))
                    counts.append(count)
                else:
                    index = lab.index(list(color))
                    counts[index] += count

            n = n + 1
            print(f"{n}/{imgs_len}")
            if n >= 10:
                break

    counts = np.array(counts) / np.sum(counts)
    lab = np.array(lab)
    counts = np.array(counts)
    np.save("./utils/gamut_pts_lab.npy", lab)
    np.save("./utils/gamut_probs_lab.npy", counts)

    temp = np.load("./utils/gamut_pts_lab.npy")
    print(temp)
    print("---------------------------")
    temp = np.load("./utils/gamut_probs_lab.npy")
    print(temp)



if __name__ == "__main__":
    main()
