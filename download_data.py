import gdown
import zipfile
import os
import shutil


file_id = '1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP'
file_name = 'time-series-dataset.zip'

# Generate the Google Drive URL for downloading
url = f'https://drive.google.com/uc?id={file_id}&export=download'
gdown.download(url, file_name, quiet=False)

# Unzip the file
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall()

# Rename the dataset folder to 'datasets'
extracted_folder = 'dataset'  # Original folder name after extraction
new_folder_name = 'datasets'
if os.path.exists(extracted_folder):
    os.rename(extracted_folder, new_folder_name)

# Remove the downloaded zip file
os.remove(file_name)