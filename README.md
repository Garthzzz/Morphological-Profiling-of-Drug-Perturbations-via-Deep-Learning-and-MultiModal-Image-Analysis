# Morphological Profiling of Drug Perturbations via Deep Learning and Multi-Modal Image Analysis

## Introduction
This project explores how to extract and interpret phenotypic signals from high-content cell imaging data subjected to various drug (chemical) perturbations. The primary goal is to evaluate different representations of cell morphology—ranging from handcrafted CellProfiler features to deep neural network embeddings—and investigate how well they can distinguish and characterize phenotypic effects.

The key research questions are:
1. **RQ1**: Can image-based features capture and distinguish the phenotypic effects of different perturbations? In particular, how do modern deep learning features compare to traditional handcrafted features in classifying treatments from cell images?
2. **RQ2**: Does integrating CellProfiler-derived features with deep neural network representations improve the accuracy or consistency of phenotypic classification and clustering? What additional value (if any) do classical features provide when combined with deep features?
3. **RQ3**: How effective are unsupervised deep learning embeddings (e.g., variational autoencoders) at representing phenotypic differences, relative to supervised CNN features? Can combining unsupervised embeddings with CellProfiler features enhance the separation of perturbation effects?
4. **RQ4**: Are the CNN models interpretable in terms of biological relevance? For example, can we identify which cellular structures the model focuses on for different perturbations, and do these correspond to known phenotypic changes?

## Repository Contents
- **`code.ipynb`**  
  Main Jupyter Notebook with the core code for data loading, model training (CNN, VAE), evaluation metrics (accuracy, silhouette score, mAP), and Grad-CAM visualization.

- **`Integrating Deep Representations and CellProfiler Features to Decode Phenotypic Effects from Cell Imaging.pdf`**  
  This is our full paper detailing our integrated approach for decoding cell phenotypic effects. In this work, we combine deep learning representations (from CNNs and VAEs) with classical morphological features extracted via CellProfiler. The paper describes our data integration process, methodologies for feature extraction and fusion, as well as unsupervised clustering and interpretability analyses (including Grad-CAM). It addresses key research questions such as whether deep image-based features can capture and differentiate perturbation effects, the value added by combining these with CellProfiler features, and if the models provide interpretable insights on underlying biological changes. For full details, please refer to the PDF [here](https://github.com/Garthzzz/Morphological-Profiling-of-Drug-Perturbations-via-Deep-Learning-and-MultiModal-Image-Analysis).


- **`output image/`**  
  A folder intended for saving or storing representative output figures (e.g., Grad-CAM heatmaps, UMAP plots).


## Brief Summary of Results
- **Classification**: A CNN fine-tuned on microscopy images generally outperforms logistic regression on purely handcrafted features, though the task is challenging (250 classes with limited samples each).
- **Feature Fusion**: Combining deep features with CellProfiler features modestly improves both accuracy and clustering metrics.  
- **VAE vs. CNN**: Unsupervised VAE embeddings alone do not strongly align with drug treatments, but they become more discriminative when fused with CellProfiler features.  
- **Interpretability (Grad-CAM)**: The CNN focuses on biologically relevant structures (e.g., nuclei, cytoplasm) for certain perturbations, providing interpretable insights into how treatments alter cell morphology.

Overall, the results suggest that multi-modal feature integration can yield more robust phenotypic characterization than either deep or handcrafted features alone, and that interpretability methods like Grad-CAM help validate and explain these findings.

---
