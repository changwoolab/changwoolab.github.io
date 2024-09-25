---
layout: post
title: "Paper Review 18: A Neural Algorithm of Artistic Style"
date: 2024-09-24 19:55:29 +0900
categories: paper-review
---

## Summary

Uses CNN features that contains only “style” or “content” representations, not exact pixel location, and Mix them

<img src="/public/img/styletransfer.png" style="display: block; margin: auto;" width="100%" />

## Method

### Content Loss

- visualize directly from feature map of CNN
    - Higher layer = high level content
    - Lower layer = Exact pixel values
- Use Gradient descent from white noise image to match content representation
- **Notation**
    
    $$
    \vec{p}:\text{original image}\\
    \vec{x}:\text{generated image}\\
    P^l, F^l: \text{feature representation in layer }l
    $$
    
- **Loss**: squared-error loss between 2 feature representations
    
    $$
    L_{content}(\vec{p},\vec{x},l)=\frac{1}{2}\sum_{i,j}(F^l_{ij}-P^l_{ij})^2
    $$
    
- **Derivative of the loss**
    
    $$
    \frac{\partial \mathcal{L}_{\text{content}}}{\partial F_{ij}^l} =
    \begin{cases}
        (F_{ij}^l - P_{ij}^l) & \text{if } F_{ij}^l > 0 \\
        0 & \text{if } F_{ij}^l < 0
    \end{cases}
    $$
    

### Style Loss

- **Style** = Correlation between feature map channels of the layer    
Why use correlation? → To remove pixel value’s impact and get real style!
    - Represented by: Gram Matrix (G)
    ( inner product between vectorized feature map *i, j* in layer *l* )
        
        $$
        G^l_{ij}=\sum_kF^l_{ik}F^l_{jk}
        $$
        
- Use Gradient descent from white noise image to match style representation
- **Notation**
    
    $$
    \vec{a}:\text{original image}\\
    \vec{x}:\text{generated image} \\
    A^l, G^l:\text{style representation in layer }l
    $$
    
- **Loss**: minimizing mean-squared distance between Gram matrix of original image and Gram matrix of generated image
    - **Contribution of the layer to total loss**
        
        $$
        E_l=\frac{1}{4N^2_lM^2_l}\sum_{i,j}(G^l_{ij}-A^l_{ij})^2
        $$
        
    - **Total Loss** (w_l = weighting factors)
        
        $$
        L_{style}(\vec{a},\vec{x})=\sum_{l=0}^Lw_lE_l
        $$
        
- **Derivative of E_l**
    
    $$
    \frac{\partial E_l}{\partial F_{ij}^l} =\begin{cases}    \frac{1}{N_l^2 M_l^2} \left( (F^l)^\top (G^l - A^l) \right)_{ji} & \text{if } F_{ij}^l > 0 \\    0 & \text{if } F_{ij}^l < 0\end{cases}
    $$
    

### Total Loss

$$
L_{total}(\vec{p},\vec{a},\vec{x})=⍺L_{content}(\vec{p},\vec{x})+βL_{style}(\vec{a},\vec{x})
$$
