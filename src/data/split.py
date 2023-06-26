import os, sys
sys.path.append("/mnt/e/Program/Python/Colorization")
import shutil
from sklearn.model_selection import train_test_split

# 데이터셋을 로드
image_folder = "dataset/TM2/Color"

# 데이터를 저장할 폴더들의 경로
train_dir = "dataset/TM2/train"
val_dir = "dataset/TM2/val"
test_dir = "dataset/TM2/test"

# 폴더를 생성할 때 기존 폴더가 있으면 삭제하고 다시 생성
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)


# 이미지 파일 목록 가져오기
image_files = sorted(os.listdir(image_folder))

# train, val, test로 데이터 분할
train_files, valtest_files = train_test_split(image_files, test_size=0.2, random_state=100)
val_files, test_files = train_test_split(valtest_files, test_size=0.5, random_state=100)

# 분할된 파일들을 해당 폴더로 복사
for file_name in train_files:
    src_path = os.path.join(image_folder, file_name)
    dst_path = os.path.join(train_dir, file_name)
    shutil.copy(src_path, dst_path)

for file_name in val_files:
    src_path = os.path.join(image_folder, file_name)
    dst_path = os.path.join(val_dir, file_name)
    shutil.copy(src_path, dst_path)

for file_name in test_files:
    src_path = os.path.join(image_folder, file_name)
    dst_path = os.path.join(test_dir, file_name)
    shutil.copy(src_path, dst_path)
