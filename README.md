## Riemann Sum Optimization for Accurate Integrated Gradients Computation

This repository contains the official implementation of the following paper accepted at [NeurIPS 2024 Workshop](https://interpretable-ai-workshop.github.io/):

> **Riemann Sum Optimization for Accurate Integrated Gradients Computation**<br>
> Shree Singhi, Swadesh Swain <br>
> https://www.arxiv.org/abs/2409.09043
>
> **Abstract:** *Integrated Gradients (IG) is a widely used algorithm for attributing the outputs of a deep neural network to its input features. Due to the absence of closed-form integrals for deep learning models, inaccurate Riemann Sum approximations are used to calculate IG. This often introduces undesirable errors in the form of high levels of noise, leading to false insights in the model's decision-making process. We introduce a framework, RiemannOpt, that minimizes these errors by optimizing the sample point selection for the Riemann Sum. Our algorithm is highly versatile and applicable to IG as well as its derivatives like Blur IG and Guided IG. RiemannOpt achieves up to 20% improvement in Insertion Scores. Additionally, it enables its users to curtail computational costs by up to four folds, thereby making it highly functional for constrained environments.*

### Usage: 

Create a folder `<dataroot>/val/images` and download and place all 50K validation images of ImageNet

Copy and paste `val_annotations.txt` inside `<dataroot>/val/`

Run `blur.py` to generate blurred images for PIC score calculations.

Each set of saliencies takes ~15GB of space
For each model and each attribution method the results need to be evaluated on, do the following:
1. Run `class_filterer.py` to filter out correctly predicted images for a model
2. Run `riemann_opt.py` with an underlying method (like IG, BlurIG or GIG) to perform pre-computation for RiemannOpt
    - Necessary if the user wants to use the RiemannOpt
    - Uses only 1 image per class (~3% of the data)
3. Run `proba_filterer.py` to filter out correctly predicted images with confidence >= 80% for a model
4. Run `script.py` to generate saliencies for a model for a chosen attribution method
5. Run `script_insertion.py` to get the insertion scores in `./results/`
6. Run `script_pic.py` to get the AUC AIC and AUC SIC scores in `./results/`
7. Run `script_completeness.py` to get the completeness scores in `./results/`

#### Authors: 
[Shree Singhi](https://github.com/ShreeSinghi), [Swadesh Swain](https://github.com/)