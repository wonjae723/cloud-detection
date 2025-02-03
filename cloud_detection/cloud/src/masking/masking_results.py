import os
import numpy as np
import torch
import sys
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.dataset import CloudDataset
from utils.model import SimpleDNN

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 데이터 전처리 함수
def preprocess_data(data):
    """Preprocess the data by replacing NaN and infinite values."""
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


# 테스트 데이터 로드 함수
def get_test_files(base_path, time_stamp, input_folders, label_folder):
    """
    Load input and label files for a given timestamp.
    """
    test_input_data_list = []

    for folder in input_folders:
        matching_file = next((f for f in os.listdir(os.path.join(base_path, folder)) if time_stamp in f), None)
        if matching_file:
            file_path = os.path.join(base_path, folder, matching_file)
            ds = xr.open_dataset(file_path)
            test_input_data = ds['image_pixel_value'].values
            test_input_data_list.append(test_input_data)
        else:
            return None, None  # If input data is missing

    if len(test_input_data_list) == len(input_folders):
        combined_test_input_data = np.stack(test_input_data_list, axis=-1)
        label_file = next((f for f in os.listdir(os.path.join(base_path, label_folder)) if time_stamp in f), None)
        if label_file:
            label_path = os.path.join(base_path, label_folder, label_file)
            ds_label = xr.open_dataset(label_path)
            test_labels = ds_label['CLD'].values
            return combined_test_input_data, test_labels
    return None, None


# 라벨링된 마스크 시각화 함수
def plot_label_mask(predictions, save_path=None):
    """
    Visualize the predicted cloud mask with colors:
    0: white (cloud), 1: gray (probably cloud), 2: green (clear)
    """
    label_colors = {
        0: [255, 255, 255],  # white (cloud)
        1: [169, 169, 169],  # gray (probably cloud)
        2: [0, 0, 0]  # black (clear)
    }

    # Create color image from predictions
    colored_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        colored_image[predictions == label] = color

    # Convert to PIL Image
    image = Image.fromarray(colored_image)

    # Display the image
    plt.imshow(image)
    plt.title("Predicted Cloud Mask")
    plt.axis('off')

    # Save or display image
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# 모델 평가 및 시각화 함수
def evaluate_model(model_file, test_time_stamp, base_path, input_folders, label_folder, model_save_dir):
    """
    Evaluate model and visualize the predictions.
    """
    # 모델 로드
    model = SimpleDNN().to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_dir, model_file)))
    model.eval()

    # 테스트 데이터 로드
    test_input_data, test_labels = get_test_files(base_path, test_time_stamp, input_folders, label_folder)

    if test_input_data is not None and test_labels is not None:
        test_dataset = CloudDataset(test_input_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())

        # 예측 결과를 900x900 형태로 변환
        all_predictions = np.array(all_predictions).reshape(test_labels.shape)

        # 저장 경로 생성 (시간별 폴더 및 파일)
        save_dir = os.path.join("C:/Users/kwj/PycharmProjects/weather/code/cloud/data/output/masking", test_time_stamp[:8])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{test_time_stamp}.png")

        # 라벨 마스크 시각화 후 저장
        plot_label_mask(all_predictions, save_path=save_path)
        print(f"Mask saved to {save_path}")

    else:
        print(f"Not enough input data for {test_time_stamp}. Skipping...")
