# Image Distillation for Safe Data Sharing in Histopathology

This repository provides a PyTorch implementation of the paper [Image Distillation for Safe Data Sharing in Histopathology.](https://arxiv.org/abs/2406.13536) accepted at MICCAI 2024.

Tested with:

- PyTorch 1.13.1
- Python 3.10.13

### Training:
* Dataset is downloaded automatically in code at the first time run ([here](https://medmnist.com/))

* To distill the small dataset and train classifier, run 
  
  `python main.py --dataset=medsyn --cluster=embedinfo --contrastive`. 




### Citation:

If you use the code, please cite

    Zhe Li, Bernhard Kainz.
    Image Distillation for Safe Data Sharing in Histopathology.
    The International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2024
    
