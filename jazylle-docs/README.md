# Jazylle — Documentation & Presentation Lead

## Deliverables checklist (from the project brief)

- [ ] **PDF report** — sections: Introduction, Methods, Experiments, Results.
- [ ] **Slides** — 10-minute class presentation (+ 5 min Q&A).
- [ ] **Contribution statement** — one paragraph per member listing what
      each person did, **including how AI assistants were used**
      (allowed for debugging/plotting, must be disclosed).
- [ ] **Demo video** (optional, 3 min) — all three methods on the same new
      inputs. Coordinate with Kurt.
- [ ] Final zip `GroupNo_CVProject.zip`: one executed notebook (outputs
      visible) + one `.py` module per method + data link + PDF + slides +
      contribution statement. Due end of Week 5. **Presentation: Nov 18.**

## Report skeleton

- **Introduction**: task (CIFAR-10 image classification), research question —
  *how do hand-designed features (HOG), learned-from-scratch features (custom
  CNN), and transferred features (fine-tuned ResNet-18) compare on accuracy,
  speed, and model size?*
- **Methods**: one subsection per method. Sources: Ron's
  `ron-hog-svm/results/TECHNICAL_NOTES.md`, Duke's and Christine's notes
  (ask them to write the same kind of file when their training finishes).
- **Experiments**: canonical 45k/5k/10k splits (shared!), metrics per slot
  (accuracy, macro F1, confusion matrix + ms/image, params, train time).
- **Results**: Kurt's comparison table + charts, plus the honest
  strengths/weaknesses story per method.

## Inputs you'll collect from the group

- Ron: `ron-hog-svm/results/` — metrics.json, confusion_matrix.png,
  TECHNICAL_NOTES.md (all ready).
- Duke: `duke-custom-cnn/results/` — same set when his training finishes.
- Christine: `christine-resnet18/results/` — same set when hers finishes.
- Kurt: comparison table + system screenshots + demo video.

## House style

All-black text, no blue, clean modern formatting (same style as the proposal).
Keep figures labeled with units; every claim about "why" a method wins/loses
needs a number behind it — the Analysis section is worth 20 points.
