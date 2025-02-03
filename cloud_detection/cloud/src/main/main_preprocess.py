import sys
import os

# `src` 폴더 경로를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src/preprocess"
sys.path.append(src_path)
from data_loader import load_file_list, load_nc_file
from calibration import load_calibration_coefficients, apply_calibration
from preprocessing import calculate_difference, save_as_netcdf


def preprocess_channel_pairs(base_folder, output_base_folder, channel_pairs, calibration_file):
    for folder1_name, folder2_name in channel_pairs:
        print(f"Processing channels: {folder1_name} and {folder2_name}")

        folder1_path = os.path.join(base_folder, folder1_name)
        folder2_path = os.path.join(base_folder, folder2_name)
        output_folder_name = f"{folder1_name}_{folder2_name}_btd"
        output_folder_path = os.path.join(output_base_folder, output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        files = load_file_list(folder1_path, folder2_path)

        channel_index1 = get_channel_index(folder1_name)  # 채널 매핑 로직 필요
        channel_index2 = get_channel_index(folder2_name)
        coefficients1 = load_calibration_coefficients(calibration_file, channel_index1)
        coefficients2 = load_calibration_coefficients(calibration_file, channel_index2)

        for file1, file2 in zip(files['folder1'], files['folder2']):
            dataset1 = load_nc_file(file1)
            dataset2 = load_nc_file(file2)

            if dataset1 and dataset2:
                img_array1 = dataset1['image_pixel_values'].values
                img_array2 = dataset2['image_pixel_values'].values

                calibrated1 = apply_calibration(img_array1, coefficients1)
                calibrated2 = apply_calibration(img_array2, coefficients2)

                diff = calculate_difference(calibrated1, calibrated2)

                # 기존 코드 수정
                file_suffix = os.path.basename(file1).split('.')[0]  # 현재 파일명에서 확장자를 제거
                file_suffix = file_suffix.split('_')[-1]  # 언더스코어로 구분된 마지막 부분(202407310000)만 추출
                save_as_netcdf(output_folder_path, file_suffix, diff)


def get_channel_index(channel_name):
    """채널 이름에 따라 인덱스를 반환"""
    mapping = {
        "sw038": 0, "wv063": 1, "wv073": 3, "ir087": 4, "ir105": 6, "ir123": 8,  "ir133": 9,
    }
    return mapping.get(channel_name)


if __name__ == "__main__":
    BASE_FOLDER = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/raw/20240810"   # 전처리하고 싶은 날짜 입력 ex)20240810
    OUTPUT_BASE_FOLDER = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/processed/20240810"  # 전처리하고 싶은 날짜 입력 ex)20240810
    CALIBRATION_FILE = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/calibration/gk2a_calibration_coeff.txt"
    CHANNEL_PAIRS = [
        ("ir105", "sw038"),
        ("ir105", "wv063"),
        ("ir105", "wv073"),
        ("ir105", "ir087"),
        ("ir105", "ir123"),
        ("ir105", "ir133")
    ]
    preprocess_channel_pairs(BASE_FOLDER, OUTPUT_BASE_FOLDER, CHANNEL_PAIRS, CALIBRATION_FILE)
    print("Preprocessing complete.")
