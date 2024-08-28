import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=128):
        super(CNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, 120, 160)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 30, 40)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (128, 8, 10)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (256, 2, 3)

        # Adjusted fully connected layer based on the output size from the previous layers
        self.fc = nn.Linear(256 * 2 * 2, self.embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ImageEncoder, self).__init__()
        # Use a pretrained ResNet backbone (you can replace it with any other model)
        resnet = models.resnet18(pretrained=True)
        layers = list(resnet.children())[:-2]  # Remove the last two layers (adaptive pooling and FC)

        self.embedding_dim = embedding_dim
        self.backbone = nn.Sequential(*layers)

        # Multi-scale feature fusion layer (optional)
        self.fusion = nn.Conv2d(512, embedding_dim, kernel_size=1)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear layer for dimensionality reduction
        self.fc = nn.Linear(embedding_dim, embedding_dim)

        # Normalization
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Pass through the backbone
        features = self.backbone(x)

        # Apply multi-scale feature fusion
        fused_features = self.fusion(features)

        # Global average pooling
        pooled_features = self.global_avg_pool(fused_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Final embedding layer
        embedding = self.fc(pooled_features)

        # Apply normalization
        embedding = self.norm(embedding)

        return embedding


# Example usage
if __name__ == "__main__":
    encoder = CNNEncoder(input_channels=3, embedding_dim=128)
    count_parameters(encoder)
    # Example input tensor with shape (Batch, Channels, Height, Width)
    input_tensor = torch.randn(16, 3, 480, 640)  # Batch of 16 RGB images of size 480x640

    embedding = encoder(input_tensor)
    print("Output Embedding Shape:", embedding.shape)  # Expected: torch.Size([16, 128])

    image_encoder = ImageEncoder(embedding_dim=256)
    input_image = torch.randn(1, 3, 180, 240)  # Example input image
    embedding = image_encoder(input_image)
    print(embedding.shape)  # Should be (1, 256)
    count_parameters(image_encoder)