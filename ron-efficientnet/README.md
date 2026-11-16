# Ron — EfficientNet-B0 fine-tuning (Slot E, pretrained, second)

ImageNet-pretrained EfficientNet-B0 (2019), fine-tuned on CIFAR-10. Same
recipe as Christine's ResNet-18 slot: Resize 224 + ImageNet normalisation,
Adam lr 1e-4, cosine decay, 12 epochs, best-val checkpoint.

## Files

- `efficientnet_finetune.py` — model, training loop, evaluation, metrics
- `train_efficientnet.py` — run this (or the notebook) to train + evaluate
- `train_efficientnet.ipynb` — same pipeline as a notebook, with outputs
- `results/` — metrics.json, confusion matrix, training curves, model.pt,
  training_log.txt

## How to run

On Colab with a GPU (recommended — ~30 min):

```bash
python train_efficientnet.py
```

On CPU the same script works but takes hours — the model was verified
end-to-end on CPU here with a short run; the full 12-epoch training was
done on this machine in the background (see training_log.txt).
