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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(data):
    """Replace NaN and infinite values in the input data."""
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


# 학습 함수
def train_model_for_all_suffixes(base_path, input_folders, label_folder, model_save_dir, num_repeats=3, epochs_per_cycle=10, learning_rate=0.001):


    time_suffixes = [str(i).zfill(2) for i in range(24)]  # 시간대 목록 (00~23)

    os.makedirs(model_save_dir, exist_ok=True)

    for time_suffix in time_suffixes:
        print(f"\nTraining model for time suffix {time_suffix}")

        # 파일 목록 필터링
        label_files = sorted([f for f in os.listdir(os.path.join(base_path, label_folder))
                              if f.endswith('.nc') and f.split('_')[-1].split('.')[0][-4:-2] == time_suffix])
        input_files_dict = {
            folder: sorted([f for f in os.listdir(os.path.join(base_path, folder)) if
                            f.endswith('.nc') and f.split('_')[-1].split('.')[0][-4:-2] == time_suffix])
            for folder in input_folders
        }

        # 모델 초기화
        model = SimpleDNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for repeat in range(num_repeats):
            print(f"\nStarting training cycle {repeat + 1}/{num_repeats}")

            # 파일 조합
            combined_files = [
                (label_file, *[input_files_dict[folder][i] for folder in input_folders])
                for i, label_file in enumerate(label_files)
            ]

            for i, (label_file, *input_files) in enumerate(combined_files, 1):
                print(f"\nTraining on label file: {label_file} ({i}/{len(label_files)})")

                # 입력 데이터 로드 및 전처리
                input_data_list = []
                for j, folder in enumerate(input_folders):
                    file_path = os.path.join(base_path, folder, input_files[j])
                    ds = xr.open_dataset(file_path)
                    input_data = preprocess_data(ds['image_pixel_value'].values)
                    input_data_list.append(input_data)

                # 입력 데이터가 6개 모두 있는지 확인
                if len(input_data_list) == 6:
                    combined_input_data = np.stack(input_data_list, axis=-1)

                    # 라벨 데이터 로드 및 전처리
                    label_path = os.path.join(base_path, label_folder, label_file)
                    ds_label = xr.open_dataset(label_path)
                    labels = preprocess_data(ds_label['CLD'].values)

                    # 데이터셋 및 데이터로더 정의
                    dataset = CloudDataset(combined_input_data, labels)
                    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

                    # 모델 학습
                    for epoch in range(epochs_per_cycle):
                        running_loss = 0.0
                        for inputs, labels in dataloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                        print(f"  Epoch {epoch + 1}/{epochs_per_cycle}, Loss: {running_loss / len(dataloader)}")

        # 모델 저장
        model_save_path = os.path.join(model_save_dir, f"spring_ontime{time_suffix}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")