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
# 데이터 전처리 함수
def preprocess_data(data):
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


# 모델 재학습 함수
def retrain_model(previous_model_dir, new_base_path, input_folders, label_folder, time_suffix, model_save_dir):
    # 이전 모델 불러오기
    previous_model_path = os.path.join(previous_model_dir, f"{time_suffix}.pth")
    model = SimpleDNN().to(device)
    model.load_state_dict(torch.load(previous_model_path))
    model.train()

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 새로운 데이터셋으로 재학습
    input_files_dict = {
        folder: sorted(
            [os.path.join(new_base_path, folder, f) for f in os.listdir(os.path.join(new_base_path, folder)) if
             f.endswith('.nc')]
        )
        for folder in input_folders
    }
    label_files = sorted(
        [os.path.join(new_base_path, label_folder, f) for f in os.listdir(os.path.join(new_base_path, label_folder)) if
         f.endswith('.nc')]
    )

    # 시간대에 맞는 파일 조합
    combined_files = [
        (label_file, *input_files)
        for label_file, *input_files in zip(label_files, *[input_files_dict[folder] for folder in input_folders])
        if label_file.split('_')[-1].split('.')[0][-4:-2] == time_suffix
    ]

    # 학습 주기
    num_repeats = 3
    epochs_per_cycle = 10
    for repeat in range(num_repeats):
        print(f"\nStarting retraining cycle {repeat + 1}/{num_repeats}")
        for i, (label_file, *input_files) in enumerate(combined_files, 1):
            print(f"\nTraining on label file: {os.path.basename(label_file)} ({i}/{len(combined_files)})")

            # 데이터 로드 및 전처리
            input_data_list = []
            for j, folder in enumerate(input_folders):
                file_path = os.path.join(new_base_path, folder, input_files[j])
                ds = xr.open_dataset(file_path)
                input_data = preprocess_data(ds['image_pixel_value'].values)
                input_data_list.append(input_data)

            if len(input_data_list) == 6:
                combined_input_data = np.stack(input_data_list, axis=-1)

                # 라벨 데이터 로드
                label_path = os.path.join(new_base_path, label_folder, label_file)
                ds_label = xr.open_dataset(label_path)
                labels = preprocess_data(ds_label['CLD'].values)

                # 데이터셋 및 데이터로더 정의
                dataset = CloudDataset(combined_input_data, labels)
                dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

                # 모델 학습
                for epoch in range(epochs_per_cycle):
                    epoch_loss = 0.0
                    for inputs, labels in dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    print(f"  Epoch {epoch + 1}/{epochs_per_cycle}, Loss: {epoch_loss / len(dataloader):.4f}")

    # 모델 저장 경로 설정 및 저장
    retrained_model_save_path = os.path.join(model_save_dir, f"{time_suffix}.pth")
    torch.save(model.state_dict(), retrained_model_save_path)
    print(f"Retrained model saved to {retrained_model_save_path}")