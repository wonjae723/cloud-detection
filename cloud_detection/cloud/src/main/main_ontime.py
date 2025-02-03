import sys
import os

# `src` 폴더 경로를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src/training"
sys.path.append(src_path)
from ontime_train import train_model_for_all_suffixes

def train_all_suffixes():
    base_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/processed/20240807"   # 훈련할 날짜 입력 ex)20240807
    input_folders = ["ir105_sw038_btd", "ir105_wv063_btd", "ir105_wv073_btd", "ir105_ir087_btd", "ir105_ir123_btd", "ir105_ir133_btd"]
    label_folder = "cloudmask"
    model_save_dir = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/output/models/ontime/20240807"   # 훈련할 날짜 입력 ex)20240807
    os.makedirs(model_save_dir, exist_ok=True)

    for hour in range(24):
        time_suffix = str(hour).zfill(2)
        print(f"\n[INFO] Training for time suffix {time_suffix}")
        train_model_for_all_suffixes(base_path, input_folders, label_folder, model_save_dir, num_repeats=3)


if __name__ == "__main__":
    train_all_suffixes()