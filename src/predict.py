
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image


class SteelDefectPredictor:

    def __init__(self, model_path="models/best_resnet18.pth"):

        self.device = torch.device("cpu")

        self.class_names = [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches"
        ]

        self.model = models.resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            len(self.class_names)
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs, dim=1
            )[0]

            confidence, predicted_idx = torch.max(
                probabilities, dim=0
            )

        return {
            "predicted_class": self.class_names[predicted_idx.item()],
            "confidence": confidence.item(),
            "probabilities": {
                name: probabilities[i].item()
                for i, name in enumerate(self.class_names)
            }
        }
