import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr
import sys
# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from utils.model import SimpleDNN
from utils.dataset import CloudDataset

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_test_files(base_path, time_stamp, input_folders, label_folder):
    """
    Load input and label files for a given timestamp.
    """
    test_input_data_list = []

    # Input 데이터 로드
    for folder in input_folders:
        folder_path = os.path.join(base_path, folder)
        matching_file = next((f for f in os.listdir(folder_path) if time_stamp in f), None)
        if matching_file:
            file_path = os.path.join(folder_path, matching_file)
            ds = xr.open_dataset(file_path)
            test_input_data_list.append(np.nan_to_num(ds['image_pixel_value'].values))
        else:
            return None, None  # 입력 데이터 누락 시 반환

    # Label 데이터 로드
    label_folder_path = os.path.join(base_path, label_folder)
    matching_label_file = next(
        (f for f in os.listdir(label_folder_path) if f.endswith(f"{time_stamp}.nc")), None
    )
    if not matching_label_file:
        print(f"[INFO] Label file not found: {label_folder_path} with timestamp {time_stamp}")
        return None, None

    label_file_path = os.path.join(label_folder_path, matching_label_file)
    ds_label = xr.open_dataset(label_file_path)
    test_labels = np.nan_to_num(ds_label['CLD'].values)

    # Ensure all input data is present
    if len(test_input_data_list) == len(input_folders):
        combined_test_input_data = np.stack(test_input_data_list, axis=-1)
        return combined_test_input_data, test_labels

    return None, None




def test_ontime_hours(base_folder, model_path, input_folders, label_folder, date):
    """
    Test Ontime Models on Hourly Data.
    """
    base_path = os.path.join(base_folder, date)
    if not os.path.exists(base_path):
        print(f"Skipping date {date} as the directory does not exist.")
        return

    accuracies = []
    for hour in range(24):  # 24시간
        time_suffix = f"{hour:02}"
        model_file = os.path.join(model_path, f"{time_suffix}.pth")  # 시간대별 모델 경로

        # 모델 파일 존재 여부 확인
        if not os.path.exists(model_file):
            print(f"[INFO] Model file {model_file} not found. Skipping.")
            continue

        # 모델 로드
        model = SimpleDNN().to(device)
        model.load_state_dict(torch.load(model_file))
        model.eval()

        # 테스트 데이터 로드
        time_stamp = f"{date}{time_suffix}00"
        test_input_data, test_labels = get_test_files(base_path, time_stamp, input_folders, label_folder)
        if test_input_data is None or test_labels is None:
            print(f"[INFO] Missing data for timestamp {time_stamp}. Skipping.")
            continue

        # 데이터 준비
        test_dataset = CloudDataset(test_input_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        # 테스트 실행
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 정확도 계산
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        accuracies.append((time_suffix, accuracy))
        print(f"Accuracy for model {time_suffix}.pth on {time_stamp}: {accuracy * 100:.2f}%")

    # 평균 정확도 계산
    if accuracies:
        average_accuracy = np.mean([acc for _, acc in accuracies])
        print(f"\nAverage Accuracy for {date}: {average_accuracy * 100:.2f}%")