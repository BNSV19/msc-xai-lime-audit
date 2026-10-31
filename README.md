# Explainable AI: LIME Interpretability Audit of a U-Net

MSc module project, INM708 Explainable Artificial Intelligence, City St George's, University of London (May 2026). Author: Badrinarayan Nandakumar.

This notebook loads the custom U-Net pet-segmentation model (see `msc-unet-segmentation`) and applies LIME. Several predictions relied on background pixels rather than the animal. This showed that a strong IoU can hide shortcut learning, and it motivated adding explanation checks to the training loop. The trained model weights (`.pth`, 209 MB) are not included because of GitHub's file-size limit.

**Skills:** Python · PyTorch · LIME · Explainable AI · Shortcut-learning detection
