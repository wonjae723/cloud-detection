import sys
import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.dataset import CloudDataset
from utils.model import SimpleDNN
from training.model_update import retrain_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_all_suffixes():
    # 경로 설정
    previous_model_dir = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/output/models/ontime/20240731"   # 기존 모델 날짜 입력 ex)20240731
    new_base_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/processed/20240807"   # 기존 모델에 새롭게 학습시키고 싶은 날짜 입력 ex)20240807
    input_folders = ['ir105_ir087_btd', 'ir105_ir123_btd', 'ir105_ir133_btd', 'ir105_wv063_btd',
                     'ir105_wv073_btd', 'ir105_sw038_btd']
    label_folder = 'cloudmask'
    model_save_dir = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/output/updated_models/20240807"  # 기존 모델에 새롭게 학습시키고 싶은 날짜 입력 ex)20240807
    os.makedirs(model_save_dir, exist_ok=True)

    # 00부터 23시까지 모델을 순차적으로 재학습
    for hour in range(24):
        time_suffix = str(hour).zfill(2)
        print(f"\n[INFO] Retraining model for time suffix {time_suffix}")
        retrain_model(previous_model_dir, new_base_path, input_folders, label_folder, time_suffix, model_save_dir)


if __name__ == "__main__":
    train_all_suffixes()