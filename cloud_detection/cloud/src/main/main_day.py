import os
import sys

# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from training.day_train import train_day

# 경로 설정
base_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/processed/20240807"   # 훈련할 날짜 입력 ex)20240807
input_folders = [
    'ir105_ir087_btd', 'ir105_ir123_btd', 'ir105_ir133_btd',
    'ir105_wv063_btd', 'ir105_wv073_btd', 'ir105_sw038_btd'
]
label_folder = 'cloudmask'
model_save_dir = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/models/day"
model_name = "day_model_20240807.pth"   # 훈련할 날짜 입력 ex)20240807

# 모델 저장 디렉토리 생성
os.makedirs(model_save_dir, exist_ok=True)

# 하루 전체 학습 실행
train_day(
    base_path=base_path,
    input_folders=input_folders,
    label_folder=label_folder,
    model_save_dir=model_save_dir,
    model_name=model_name
)
