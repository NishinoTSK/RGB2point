
from model import PointCloudNet
from utils import predict
import torch
import torch.nn as nn



model_save_name = "/content/RGB2point/pc1024_three.pth"

model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
state = torch.load(model_save_name)
if "model" in state:
    model.load_state_dict(state["model"])
else:
    model.load_state_dict(state)
model.eval()  

image_path = "/content/RGB2point/img/fish.png"
save_path = "result/fish.ply"

predict(model, image_path, save_path)
