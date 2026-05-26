import os
import urllib.request
from pathlib import Path

def download_chb01():
    base_url = "https://physionet.org/files/chbmit/1.0.0/chb01/"
    data_dir = Path("c:/work/explainable-AI/data/external/chbmit/chb01")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        "chb01-summary.txt",
        "chb01_01.edf", "chb01_02.edf", "chb01_03.edf", "chb01_04.edf",
        "chb01_05.edf", "chb01_06.edf", "chb01_14.edf", "chb01_15.edf",
        "chb01_16.edf", "chb01_17.edf", "chb01_18.edf", "chb01_19.edf",
        "chb01_20.edf", "chb01_21.edf", "chb01_22.edf", "chb01_23.edf",
        "chb01_24.edf", "chb01_25.edf", "chb01_26.edf", "chb01_27.edf"
    ]
    
    for filename in files_to_download:
        filepath = data_dir / filename
        url = base_url + filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists, skipping.")
            
if __name__ == "__main__":
    download_chb01()
