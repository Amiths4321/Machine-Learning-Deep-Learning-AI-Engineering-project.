import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "Scratches"
]


class SteelDefectPredictor:

    def __init__(self, model_path):

        self.device = torch.device("cpu")

        self.model = models.resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            6
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Grayscale(num_output_channels=3),

            transforms.Resize(256),

            transforms.CenterCrop(224),

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
                outputs,
                dim=1
            )

        predicted_index = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[0, predicted_index].item()

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: probabilities[0, i].item()
                for i in range(len(CLASS_NAMES))
            }
        }