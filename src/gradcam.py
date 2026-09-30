import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from medmnist import PathMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import build_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = build_model(num_classes=9)
    model.load_state_dict(torch.load("pathmnist_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()

    # Target layer3 instead of layer4 to retain spatial dimensions (2x2 vs 1x1)
    target_layers = [model.layer3]

    # 2. Load Sample Image
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_dataset = PathMNIST(split='test', transform=data_transform, download=True)
    img_tensor, label = test_dataset[0]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # 3. Generate Grad-CAM Heatmap
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Get model prediction class
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # 4. Prepare normalized RGB background image (values between 0 and 1)
    orig_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    orig_img = (orig_img * 0.5 + 0.5)
    orig_img = np.clip(orig_img, 0, 1)

    # 5. Overlay Heatmap on Image
    visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)

    # 6. Plot Results
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original (True Label: {label[0]})")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title(f"Grad-CAM (Pred Class: {pred_class})")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/gradcam_result.png")
    print("Grad-CAM visualization saved successfully to outputs/gradcam_result.png")
    plt.show()