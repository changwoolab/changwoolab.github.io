---
layout: post
title: "Entropy, Cross-entropy, KL Divergence and JSD"
date: 2024-10-03 11:55:29 +0900
categories: ML-Basics
---

## Entropy (Information Theory)

Measures expected amount of information needed to describe the state of the variable, considering the distribution of probabilities across all potential states.

$$
H = \sum_ip_i\log_2\frac{1}{p_i}=-\sum_ip_i\log_2{p_i}
$$

- The amount of information needed decreases when probability goes to 1

<img src="/public/img/kl_divergence_and_jsd.png" style="display: block; margin: auto;" width="200" />

## Cross-Entropy

Similar to Entropy, but with estimated probability distribution Q not real distribution P.

Cross-entropy of Q relative to P can be defined as

$$
H(p, q) = \sum_{i} p_i \log_2 \frac{1}{q_i} = - \sum_{i} p_i \log_2 q_i
$$

- Why use Cross-Entropy loss in machine learning?
→ Because it calculates the entropy (amount of information) of estimated distribution relative to real distribution.

## KL divergence (Relative entropy from Q to P)

Measures how one reference probability distribution P is different from probability distribution Q.

It is measured by calculating the difference of entropy, which can be defined with Entropy and Cross-entropy.

$$
\begin{align*}
D_{KL}(P \parallel Q) 
&=H(p,q)-H(p)
\\&= \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
\end{align*}
$$

Now, we can understand the characteristics of KL divergence

1. KL divergence ≥ 0 because
    
    $$
    H(p,q) \ge H(p)
    $$
    
2. KL divergence is asymmetric because
    
    $$
    D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)
    $$
    
- Why not use KL divergence in loss function?
    - Because H(p) is real distribution it is a constant value, which can be omitted. So we simply use the Cross-Entropy loss.
    - Because it is asymmetric, it is not a “distance”. To use KL divergence as distance, we need to make it symmetric

## JSD (Jenson-Shannon divergence)

Symmetric version of KL divergence. It can be used as a distance.

$$
\begin{align*}
JSD(p \parallel q) 
&= \frac{D_{KL}(p\parallel M)+D_{KL}(q\parallel M)}{2}\\
& \text{where, }M=\frac{p+q}{2}
\end{align*}
$$