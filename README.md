
This repository provides a PyTorch implementation of the paper [Image Distillation for Safe Data Sharing in Histopathology](https://arxiv.org/abs/2406.13536) accepted at MICCAI 2024.

Tested with:

- PyTorch 1.13.1
- Python 3.10.13

### Distillation and Training:
* Dataset is downloaded automatically in code at the first time run ([here](https://medmnist.com/))

* Pre-trained classifiers for embedding can be downloaded ([here](https://drive.google.com/drive/folders/15xvSnOT8FHEVO4Yd9k9-oEhKAOz6NFJy?usp=sharing))

* To distill the small dataset and train classifier on images with size 256, run 
  
  ```
  # run with contrastive loss
  python main_bound.py --dataset=medsyn --cluster=embedinfo --contrastive

  # run only with ce loss
  python main.py --dataset=medsyn --cluster=embedinfo
  ```

* To distill the small dataset and train classifier on images with size 64, run 
  
  ```
  # run with contrastive loss
  python main_smallbound.py --dataset=medsyn --cluster=embedinfo --contrastive

  # run only with ce loss
  python main_small.py --dataset=medsyn --cluster=embedinfo
  ```




### Citation:

If you use the code, please cite

    Zhe Li, Bernhard Kainz.
    Image Distillation for Safe Data Sharing in Histopathology.
    The International Conference on Medical Image Computing and Computer Assisted Intervention 
    (MICCAI), 2024
    
