import torch
import torch.nn as nn
import torch.optim as optim
import timm  # PyTorch Image Models library
from torchvision import transforms

# 1. Define data transformations
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Load pre-trained Vision Transformer model for 10 classes
NUM_CLASSES = 10
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)

# 3. Define loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
