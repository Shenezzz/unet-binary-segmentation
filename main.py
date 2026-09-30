import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)

IMG_SIZE = 128


class PetSegmentationDataset(Dataset):
    def __init__(self, split="trainval", img_size=IMG_SIZE):
        self.base_dataset = torchvision.datasets.OxfordIIITPet(
            root="./data",
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.img_size = img_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # scales to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # standard ImageNet
                    std=[0.229, 0.224, 0.225],
                ),  # normalization stats
            ]
        )

        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, trimap = self.base_dataset[idx]
        image = self.image_transform(image)

        trimap = self.mask_resize(trimap)
        trimap = torch.from_numpy(
            np.array(trimap, dtype=np.int64)
        )  # (H, W), values in {1,2,3}

        # 1 if trimap value is 1, 0 elsewhere
        binary_mask = (trimap == 1).float().unsqueeze(0)  # (1, H, W)

        return image, binary_mask


print("Loading Oxford-IIIT Pet dataset ...")
full_dataset = PetSegmentationDataset(split="trainval")

# train/val split
val_fraction = 0.1
val_size = int(len(full_dataset) * val_fraction)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels  # using 32 as the base (instead of the paper's 64)
        # keeps parameter count reasonable for a CPU-friendly demo

        # encoder
        self.enc1 = DoubleConv(in_channels, c)  # -> (B, c, 128, 128)
        self.enc2 = DoubleConv(c, c * 2)  # -> (B, 2c, 64, 64)
        self.enc3 = DoubleConv(c * 2, c * 4)  # -> (B, 4c, 32, 32)
        self.enc4 = DoubleConv(c * 4, c * 8)  # -> (B, 8c, 16, 16)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)  # -> (B, 16c, 8, 8)

        # decoder
        # ConvTranspose2d does the upsampling; DoubleConv processes
        # the result AFTER concatenating the matching skip connection.
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(
            c * 16, c * 8
        )  # c*16 in-channels: c*8 (upsampled) + c*8 (skip)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        # Final 1x1 convolution: maps the c channels down to
        # out_channels (1 here, for binary segmentation). We output
        # raw logits (no sigmoid) -- BCEWithLogitsLoss applies
        # sigmoid internally, which is more numerically stable.
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder, saving each stage's output for its skip connection ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(e4))

        # --- Decoder, concatenating the matching encoder feature map at each step ---
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)  # concatenate along the CHANNEL dimension
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1) # 3*128*128 -> 1*128*128 (the final feature map martix itself is considered as logits)


model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def dice_coefficient(preds, targets, threshold=0.5, eps=1e-7):
    """
    Dice coefficient: 2*|intersection| / (|preds| + |targets|).
    Ranges from 0 (no overlap) to 1 (perfect overlap). This is the
    standard segmentation quality metric -- much more informative
    than raw pixel accuracy here, since ~70% of pixels are
    background, so a model that predicts "all background" would
    still get ~70% pixel accuracy while being completely useless.
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def run_one_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(logits.detach(), masks)
            n_batches += 1

    return total_loss / n_batches, total_dice / n_batches


def train_unet(num_epochs=30):
    print("\nStarting training...\n")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_dice = run_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_dice = run_one_epoch(model, val_loader, criterion, optimizer=None)
        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}, Dice: {train_dice:.3f} | "
            f"Val loss: {val_loss:.4f}, Dice: {val_dice:.3f}"
        )


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def visualize_predictions(dataset, n=4, save_path="unet_predictions.png"):

    # model = UNet()
    # model.load_state_dict(torch.load("unet_pets.pt"))
    # model.to(device)
    model.eval()
    fig, axes = plt.subplots(5, n, figsize=(n * 2, 9))

    for i in range(n):
        image, mask = dataset[i]
        with torch.no_grad():
            logit = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) > 0.5).float().cpu().squeeze()

        # Undo normalization to display the original-looking photo.
        img_display = image * IMAGENET_STD + IMAGENET_MEAN
        img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()

        axes[0, i].imshow(img_display)
        axes[0, i].set_title("Image")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask.squeeze().numpy(), cmap="gray")
        axes[1, i].set_title("Ground truth mask")
        axes[1, i].axis("off")

        axes[2, i].imshow(pred.numpy(), cmap="gray")
        axes[2, i].set_title("U-Net prediction")
        axes[2, i].axis("off")
        
        axes[3, i].imshow(img_display * pred.numpy()[..., None], cmap="gray")
        axes[3, i].set_title("Prediction Region")
        axes[3, i].axis("off")
        
        axes[4, i].imshow(img_display * mask.squeeze().numpy()[..., None], cmap="gray")
        axes[4, i].set_title("Masked Region")
        axes[4, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    train_unet(num_epochs=15)
    visualize_predictions(val_dataset)
    torch.save(model.state_dict(), "unet_pets.pth")
    print("Model weights saved to unet_pets.pth")
