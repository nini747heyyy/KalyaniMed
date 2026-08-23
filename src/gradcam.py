import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import build_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx=None):
        self.model.eval()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)
        return cam

# Generate visualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=9)
model.load_state_dict(torch.load("pathmnist_resnet18.pth", map_location=device))
model.to(device)

grad_cam = GradCAM(model, model.layer4)

# Example visualization logic: Run on a sample image and use cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) to overlay on the cell sample.