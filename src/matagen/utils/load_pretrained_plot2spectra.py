# The pretrained weights were adapted from MaterialsEyes: https://github.com/MaterialEyes/Plot2Spec
import os
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?id="+id
    cmd = "gdown %s -O %s"%(URL, destination)
    os.system(cmd)  
    

model_path = "checkpoints_1"
os.umask(0)
os.makedirs(model_path, mode=0o777, exist_ok=True)
os.makedirs(f"{model_path}/axis_alignment", mode=0o777, exist_ok=True)
os.makedirs(f"{model_path}/plot_data_extraction", mode=0o777, exist_ok=True)

download_file_from_google_drive('1hmOLFkeR2q2bRa1TQixP7gTLxuLWN0fM', f"{model_path}/axis_alignment/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py")
download_file_from_google_drive('18Ypstzm9kO6LlVJblHP6ceFXlDrBHWSY', f"{model_path}/axis_alignment/craft_mlt_25k.pth")
download_file_from_google_drive('1G-zrIXk4zbUQ_9nycdlaxyOHItJht7WG', f"{model_path}/axis_alignment/craft_refiner_CTW1500.pth")
download_file_from_google_drive('13T3xGdbJgJr_76Jb36rUUUqY2lnrZKLY', f"{model_path}/axis_alignment/epoch_200.pth")
download_file_from_google_drive('1EojUgM0vTGHjfMdDiWOLzOaB98xhau1Y', f"{model_path}/axis_alignment/TPS-ResNet-BiLSTM-Attn.pth")
download_file_from_google_drive('1qofgH2CSbZZ_VfLSvQnMC5ra4F58LsJD', f"{model_path}/plot_data_extraction/checkpoint_0999.pth")