"""Custom 4-block CNN for CIFAR-10, written in PyTorch.

Slot B (trained from scratch) of the Computer Vision final project.
The architecture below is the starting point -- read every line, understand
it, and adjust it if you can justify the change (that is what the "own
architecture" rubric points reward). You must be able to explain every line
in the Q&A.

Architecture (input 32x32 RGB):
  Block 1: Conv(3->32) -> BN -> ReLU -> MaxPool  -> 16x16x32
  Block 2: Conv(32->64) -> BN -> ReLU -> MaxPool -> 8x8x64
  Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool -> 4x4x128
  Block 4: Conv(128->256) -> BN -> ReLU -> AdaptiveAvgPool -> 256
  Head: Linear(256 -> 10)
Total: ~0.39M parameters (far below ResNet-18's ~11M).
"""
import numpy as np
import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Four convolutional blocks + a linear classifier head."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Each block: 3x3 convolution (padding 1 keeps the spatial size),
        # batch normalization (stabilizes training), ReLU (nonlinearity).
        # MaxPool halves the spatial size after the first three blocks;
        # the last block collapses to 1x1 with adaptive average pooling.
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)   # flatten the 1x1x256 map
        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One full pass over the training loader. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()            # clear last step's gradients
        outputs = model(images)          # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                  # backpropagation
        optimizer.step()                 # gradient descent update
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Loss, accuracy and all predictions on a loader (no gradients)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(labels.cpu().numpy())
    return (total_loss / total, correct / total,
            np.concatenate(all_pred), np.concatenate(all_true))


def fit(model, train_loader, val_loader, device, epochs=30, lr=1e-3,
        weight_decay=5e-4, verbose=True):
    """Train with Adam + cosine schedule; keep the best-validation model.

    Returns a history dict with per-epoch train/val loss and accuracy
    (used for the training-curves plot the brief requires).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs)
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        # Keep the checkpoint with the best validation accuracy
        # (this is how we "address overfitting": we never pick by test).
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}", flush=True)

    model.load_state_dict(best_state)  # restore the best-val weights
    print(f"  best val accuracy: {best_acc:.4f}", flush=True)
    return history


def plot_curves(history, save_path):
    """Training/validation loss + accuracy curves (required by the brief)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Loss curves"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy curves"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics (self-contained; no sklearn dependency)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def accuracy_and_macro_f1(cm):
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp),
                          where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp),
                       where=true_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(tp.sum() / cm.sum()), float(f1.mean()), f1


def plot_confusion_matrix(cm, class_names, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
