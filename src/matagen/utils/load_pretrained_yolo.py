import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?id="+id
    cmd = "gdown %s -O %s"%(URL, destination)
    os.system(cmd)  
    

model_path = "checkpoints_1"
os.umask(0)
os.makedirs(model_path, mode=0o777, exist_ok=True)

download_file_from_google_drive('1baAJd8WDN55usWSn8YEgQ9fij72A8mzI', f"{model_path}/yolov11_finetuned_augmentation_best.pt")