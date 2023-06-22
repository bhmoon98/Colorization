# -*- coding: utf-8 -*-

import os
import csv, cv2
import numpy as np
from skimage import io, transform
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def calculate_metrics(image_folder1, image_folder2):
    image_list1 = sorted(os.listdir(image_folder1))
    image_list2 = sorted(os.listdir(image_folder2))
    metrics_list = []

    total_images = len(image_list1)
    progress = 0

    l_channel = np.zeros((256, 256))+50.0

    for image_name1, image_name2 in zip(image_list1, image_list2):
        image_path1 =os.path.join(image_folder1, image_name1)
        image_path2 = os.path.join(image_folder2, image_name2)

        # 이미지 로드
        image1 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
        image2 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
        
        image1 = cv2.resize(image1,(256, 256), interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, (256, 256), interpolation=cv2.INTER_CUBIC)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        l_channel = image1[:,:,0]
        image2[:,:,0] = l_channel

        image1 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

        # SSIM 및 PSNR 계산
        ssim_score = ssim(image1, image2, multichannel=True)
        psnr_score = psnr(image1, image2)

        # 결과를 리스트에 추가
        metrics_list.append((image_name1, image_name2, ssim_score, psnr_score))

        # 진행 상황 업데이트
        progress += 1
        print(f"진행 상황: {progress}/{total_images}")

    return metrics_list


def generate_top_bottom_images(metrics, folder1, folder2, n=5):
    # SSIM 및 PSNR 점수를 기준으로 상위 n개 및 하위 n개 이미지 가져오기
    ssim_scores = [item[2] for item in metrics]
    psnr_scores = [item[3] for item in metrics]

    top_n_indices_ssim = np.argsort(ssim_scores)[-n:]
    bottom_n_indices_ssim = np.argsort(ssim_scores)[:n]

    top_n_indices_psnr = np.argsort(psnr_scores)[-n:]
    bottom_n_indices_psnr = np.argsort(psnr_scores)[:n]

    # 상위 n개 및 하위 n개 이미지 플롯 생성
    fig, axes = plt.subplots(nrows=4, ncols=n, figsize=(20, 16))

    for i, (idx_ssim, idx_psnr) in enumerate(zip(top_n_indices_ssim, top_n_indices_psnr)):
        image1_ssim, image2_ssim, ssim_score, _ = metrics[idx_ssim]
        image1_psnr, image2_psnr, _, psnr_score = metrics[idx_psnr]
        image_path1_ssim = os.path.join(folder1, image1_ssim)
        image_path2_ssim = os.path.join(folder2, image2_ssim)
        image_path1_psnr = os.path.join(folder1, image1_psnr)
        image_path2_psnr = os.path.join(folder2, image2_psnr)
        image1_ssim = io.imread(image_path1_ssim)
        image2_ssim = io.imread(image_path2_ssim)
        image1_psnr = io.imread(image_path1_psnr)
        image2_psnr = io.imread(image_path2_psnr)

        axes[0, i].imshow(image1_ssim)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"SSIM: {ssim_score:.4f}")

        axes[1, i].imshow(image2_ssim)
        axes[1, i].axis("off")

        axes[2, i].imshow(image1_psnr)
        axes[2, i].axis("off")
        axes[2, i].set_title(f"PSNR: {psnr_score:.4f}")

        axes[3, i].imshow(image2_psnr)
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"top_{n}_images.png")
    plt.show()

    
    # 상위 n개 및 하위 n개 이미지 플롯 생성
    fig, axes = plt.subplots(nrows=4, ncols=n, figsize=(20, 16))

    for i, (idx_ssim, idx_psnr) in enumerate(zip(bottom_n_indices_ssim, bottom_n_indices_psnr)):
        image1_ssim, image2_ssim, ssim_score, _ = metrics[idx_ssim]
        image1_psnr, image2_psnr, _, psnr_score = metrics[idx_psnr]
        image_path1_ssim = os.path.join(folder1, image1_ssim)
        image_path2_ssim = os.path.join(folder2, image2_ssim)
        image_path1_psnr = os.path.join(folder1, image1_psnr)
        image_path2_psnr = os.path.join(folder2, image2_psnr)
        image1_ssim = io.imread(image_path1_ssim)
        image2_ssim = io.imread(image_path2_ssim)
        image1_psnr = io.imread(image_path1_psnr)
        image2_psnr = io.imread(image_path2_psnr)

        axes[0, i].imshow(image1_ssim)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"SSIM: {ssim_score:.4f}")

        axes[1, i].imshow(image2_ssim)
        axes[1, i].axis("off")

        axes[2, i].imshow(image1_psnr)
        axes[2, i].axis("off")
        axes[2, i].set_title(f"PSNR: {psnr_score:.4f}")

        axes[3, i].imshow(image2_psnr)
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"bottom_{n}_images.png")
    plt.show()



# 이미지가 저장된 폴더 경로 설정
folder1 = "./../../Circuit/Crop/Color/test"
folder2 = "results/image/disco_crop_1/model_best"

# 이미지의 SSIM 및 PSNR 계산
metrics = calculate_metrics(folder1, folder2)

# 결과 출력 및 CSV 파일 저장
output_file = "withoutL.csv"
header = ["이미지 1", "이미지 2", "SSIM", "PSNR"]

with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

    for image1, image2, ssim_score, psnr_score in metrics:
        print(f"이미지 1: {image1}, 이미지 2: {image2}")
        print(f"SSIM: {ssim_score:.4f}")
        print(f"PSNR: {psnr_score:.4f}")
        print()

        writer.writerow([image1, image2, ssim_score, psnr_score])

print(f"결과가 {output_file}에 저장되었습니다.")

ssim_scores = [item[2] for item in metrics]
psnr_scores = [item[3] for item in metrics]

mean_ssim = np.mean(ssim_scores)
mean_psnr = np.mean(psnr_scores)

print(f"SSIM 평균: {mean_ssim:.4f}")
print(f"PSNR 평균: {mean_psnr:.4f}")

# 평균값을 CSV 파일에 기록
with open(output_file, "a", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([])
    writer.writerow(["평균", "", mean_ssim, mean_psnr])

print(f"결과가 {output_file}에 저장되었습니다.")

generate_top_bottom_images(metrics, folder1, folder2, n=5)