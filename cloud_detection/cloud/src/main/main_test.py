import os
import sys
from datetime import datetime

# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 테스트 모듈 임포트
from testing.ontime_model.test_hours_data import test_ontime_hours
from testing.ontime_model.test_full_data import test_ontime_full
from testing.day_model.test_hours_data import test_day_hours
from testing.day_model.test_full_data import test_day_full

def main():
    # 테스트 방식 선택 {시간대별 모델 (정각시간대만 test , 시간대에 해당하는 데이터들 모두 test) , 단일 모델 ( 정각시간대만 test, 모든 시간대 test)}
    print("Select a test type:")
    print("1. Ontime Models - Hourly Data")
    print("2. Ontime Models - Full Data")
    print("3. Day Model - Hourly Data")
    print("4. Day Model - Full Data")
    choice = input("Enter (1-4): ").strip()

    test_type_map = {
        "1": ("ontime", test_ontime_hours),
        "2": ("ontime", test_ontime_full),
        "3": ("day", test_day_hours),
        "4": ("day", test_day_full),
    }
    selected_test_info = test_type_map.get(choice)
    if not selected_test_info:
        print("[ERROR] Invalid choice. Exiting.")
        sys.exit()

    model_type, selected_test = selected_test_info

    # 테스트할 날짜와 불러올 모델 날짜 입력
    test_date = "20240807"
    model_date = "20240731"

    # Configure paths
    base_folder = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/processed"
    model_save_dir = "C:/Users/kwj/PycharmProjects/weather/code/cloud/data/output/models/"
    input_folders = ['ir105_ir087_btd', 'ir105_ir123_btd', 'ir105_ir133_btd', 'ir105_wv063_btd', 'ir105_wv073_btd', 'ir105_sw038_btd']
    label_folder = 'cloudmask'


    print(f"\nTesting date: {test_date} | Model date: {model_date}")
    if model_type == "ontime":
        model_path = os.path.join(model_save_dir, "ontime", model_date)
    elif model_type == "day":
        model_path = os.path.join(model_save_dir, "day", f"day_model_{model_date}.pth")
    else:
        print("[ERROR] Unknown model type. Exiting.")
        return


    selected_test(base_folder, model_path, input_folders, label_folder, test_date)

if __name__ == "__main__":
    main()
