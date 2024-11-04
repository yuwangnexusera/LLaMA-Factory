import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

# 将克隆的 ESRGAN 路径添加到 Python 路径
sys.path.append("/path/to/ESRGAN")

from RRDBNet_arch import RRDBNet
from utils import load_pretrained_model

# 定义 ESRGAN 模型
model = RRDBNet(3, 3, 64, 23, gc=32)
model_path = "/path/to/ESRGAN/experiments/pretrained_models/RRDB_ESRGAN_x4.pth"  # 更新为你的模型路径

# 加载预训练模型
model = load_pretrained_model(model, model_path)
model.eval()

# 打开图像
image_path = "/root/LLM/data/a6d6c5854c9841468d743fa4088fe477.jpg"
img = Image.open(image_path).convert("RGB")

# 图像预处理
preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 将图像转为张量  # 标准化
)
input_tensor = preprocess(img).unsqueeze(0)

# 使用模型进行超分辨率处理
with torch.no_grad():
    output_tensor = model(input_tensor)

# 后处理并保存结果
output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
output_image.save("/root/LLM/data/enhanced_image.jpg")
