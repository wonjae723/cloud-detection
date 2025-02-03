import os
import sys
# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from masking.masking_results import evaluate_model


def run_model_evaluation():
    # 경로 설정
    base_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/processed/20240807"   # masking할 날짜 입력 ex)20240807
    input_folders = ['ir105_ir087_btd', 'ir105_ir123_btd', 'ir105_ir133_btd', 'ir105_wv063_btd', 'ir105_wv073_btd',
                     'ir105_sw038_btd']
    label_folder = 'cloudmask'
    model_save_dir = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/output/models/ontime/20240731"   # masking에 이용할 모델의 날짜 입력 ex)20240731

    # 사용할 모델 파일명과 테스트 타임스탬프
    model_file = "00.pth"  # 사용할 모델 파일명 ex)00.pth
    test_time_stamp = "202408070000"  # 테스트할 날짜와 시간대 ex)202408070000

    print(f"\n[INFO] Evaluating model {model_file} for timestamp {test_time_stamp}")

    # 모델 평가 함수 호출
    evaluate_model(model_file, test_time_stamp, base_path, input_folders, label_folder, model_save_dir)


if __name__ == "__main__":
    run_model_evaluation()
