from color_torch import *
from torchvision.transforms import ToTensor
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = cv2.imread("D:\\MSI_NB\\source\\data\\flowl.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(img).unsqueeze(0)
    assert img_tensor.dim() == 4 and img_tensor.shape[1] == 3, "tensor shape should be like B x 3 x H x W"
    hsv_tensor = rgb2hsv_torch(img_tensor)
    hsvback = hsv2rgb_torch(hsv_tensor)
    hsl_tensor = rgb2hsl_torch(img_tensor)
    hslback = hsl2rgb_torch(hsl_tensor)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img_tensor[0].permute(1, 2, 0).numpy())
    axes[1].imshow(hsvback[0].permute(1, 2, 0).numpy())
    axes[2].imshow(hslback[0].permute(1, 2, 0).numpy())
    plt.show()
