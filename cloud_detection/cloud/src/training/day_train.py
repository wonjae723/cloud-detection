import os
import sys

# `src/utils` 폴더를 Python 경로에 추가
src_path = "C:/Users/kwj/PycharmProjects/weather/code/cloud/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import CloudDataset
from utils.model import SimpleDNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(data):
    """Replace NaN and infinite values in the input data."""
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def train_day(base_path, input_folders, label_folder, model_save_dir, model_name, epochs=10, num_repeats=3, learning_rate=0.001):
    """
    Train a model using all the data available for a specific day.

    Args:
        base_path (str): Base path containing input and label data.
        input_folders (list): List of input folder names.
        label_folder (str): Folder containing label files.
        model_save_dir (str): Directory to save the trained model.
        model_name (str): Name of the model file to be saved.
        epochs (int): Number of epochs for each repeat.
        num_repeats (int): Number of training repeats.
        learning_rate (float): Learning rate for the optimizer.
    """
    if not os.path.exists(base_path):
        raise ValueError(f"Base path {base_path} does not exist.")
    for folder in input_folders:
        if not os.path.exists(os.path.join(base_path, folder)):
            raise ValueError(f"Input folder {folder} does not exist.")
    if not os.path.exists(os.path.join(base_path, label_folder)):
        raise ValueError(f"Label folder {label_folder} does not exist.")

    # Prepare input and label files
    input_files_dict = {folder: sorted(os.listdir(os.path.join(base_path, folder))) for folder in input_folders}
    label_files = sorted(os.listdir(os.path.join(base_path, label_folder)))

    # Initialize model, criterion, and optimizer
    model = SimpleDNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for repeat in range(num_repeats):
        print(f"\n[INFO] Starting training cycle {repeat + 1}/{num_repeats}")

        combined_files = list(zip(label_files, *[input_files_dict[folder] for folder in input_folders]))

        for i, (label_file, *input_files) in enumerate(combined_files, 1):
            print(f"\n[INFO] Training on label file: {label_file} ({i}/{len(combined_files)})")

            # Prepare input data
            input_data_list = []
            for j, folder in enumerate(input_folders):
                file_path = os.path.join(base_path, folder, input_files[j])
                ds = xr.open_dataset(file_path)
                input_data_list.append(preprocess_data(ds['image_pixel_value'].values))

            if len(input_data_list) != len(input_folders):
                print(f"[WARNING] Skipping label file {label_file} due to incomplete data.")
                continue

            combined_input_data = np.stack(input_data_list, axis=-1)

            # Prepare label data
            label_path = os.path.join(base_path, label_folder, label_file)
            ds_label = xr.open_dataset(label_path)
            labels = preprocess_data(ds_label['CLD'].values)

            dataset = CloudDataset(combined_input_data, labels)
            dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

            # Training loop
            for epoch in range(epochs):
                running_loss = 0.0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"[INFO] Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    # Save the model
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")
