"""ResNet-18 fine-tuning for CIFAR-10.

Slot C (pretrained, modern) of the Computer Vision final project.
Loads ImageNet-pretrained ResNet-18 weights from torchvision and fine-tunes
the whole network on CIFAR-10. Read every line and understand it -- you must
be able to explain every line in the Q&A.

Why this works (the one-paragraph story for the report): ResNet-18 was
trained on 1.2M ImageNet photos, so its early layers already detect edges,
textures and shapes. Fine-tuning keeps that knowledge and only adapts it to
CIFAR-10's 10 classes, which is why a pretrained model beats a from-scratch
one on small data. The residual (skip) connections are what let an 18-layer
network train at all: each block learns a *residual* F(x) added to its input
x, so gradients flow straight through even in deep stacks.

Preprocessing (this is the "correct preprocessing" rubric point):
  - train: Resize to 224, random horizontal flip, ImageNet normalization
  - eval:  Resize to 224, ImageNet normalization (no augmentation)
CIFAR-10 images are 32x32; ResNet-18 expects 224x224 inputs.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=10, pretrained=True):
    """ResNet-18 with its final layer replaced for 10 CIFAR-10 classes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # The original head is Linear(512 -> 1000) for ImageNet; swap it.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)          # forward pass (once)
        loss = criterion(outputs, labels)
        loss.backward()                  # backpropagation
        optimizer.step()                 # gradient descent update
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def fit(model, train_loader, val_loader, device, epochs=12, lr=1e-4,
        weight_decay=1e-4, verbose=True):
    """Fine-tune with Adam at a small lr (pretrained weights are precious).

    A small learning rate matters: the pretrained filters are already good,
    so we nudge them instead of relearning from scratch. Returns the
    per-epoch history for the training-curves plot.
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
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    print(f"  best val accuracy: {best_acc:.4f}", flush=True)
    return history


def plot_curves(history, save_path):
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
