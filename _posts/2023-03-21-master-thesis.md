---
layout: post
title: "Causal Inference Using Random Forests and LASSO"
subtitle: "Machine learning methods for estimating treatment effects"
background: '/img/posts/master-thesis/normality.png'
---

My interest in machine learning for causal inference began in a computational statistics course at the University of Bonn in 2021. The course compared machine learning models such as LASSO, random forests, XGBoost and support vector machines, and covered training and test splits, cross-validation and the bias-variance trade-off, where machine learning methods typically accept a little bias in exchange for lower variance.

In economics, however, the main concern is usually causality, where bias, variance and valid confidence intervals are the key properties of an estimator. This raises a practical question: how can the flexibility of machine learning in large covariate spaces be used for high-dimensional problems in statistical inference? The question matters more as data become more abundant and more granular.

I focused on this topic through further courses in econometrics and statistics, and it became the subject of my master's thesis, "Causal Inference Using Random Forests and LASSO". The thesis evaluates machine learning methods for estimating causal effects through simulations and empirical applications. This post summarizes the methods, the simulations and two empirical applications.

## Introduction and Theoretical Background

### Research Motivation

Economic datasets are increasingly large and complex, and traditional econometric approaches can struggle with high-dimensional data and complex nuisance parameters. The thesis examines how machine learning techniques, in particular random forests and LASSO, can improve causal inference in econometric models.

### Prior Work

The thesis builds on work by Susan Athey (Stanford) and Victor Chernozhukov (MIT) on machine learning for the estimation of treatment effects. It first reviews existing methods and their limitations, and then introduces three approaches: Generalized Random Forests (GRF), Double Machine Learning (DML) and post-double-selection. These methods were chosen for their potential to improve the accuracy and reliability of treatment effect estimates in complex econometric models.

### Methods

In brief:

- **Generalized Random Forests (GRF)** adapt the random forest algorithm to estimate treatment effects, using a local linear approach that accommodates many covariates and complex interactions.
- **Double Machine Learning (DML)** separates the estimation of nuisance parameters from the target parameter, which reduces bias and makes the causal estimates more robust.
- **Post-double-selection** uses LASSO to select the variables most relevant to the outcome and the treatment, which guides variable selection in high-dimensional settings.

### Simulation Studies

A series of Monte Carlo simulations tests the performance of these estimators in settings ranging from randomized controlled trials to strong confounding, designed to mimic the complexity of real-world data.

### Empirical Applications

The methods are then applied to two empirical cases:

- **Microcredit in Bosnia and Herzegovina:** the economic effects of microcredit availability, using data from a prior study.
- **401(k) participation and financial assets:** how eligibility for and participation in 401(k) plans affect net financial assets.

### Findings

In the simulations, GRF and DML performed well in complex, high-dimensional settings compared with traditional estimators. The empirical applications show how these methods can be used in policy evaluation.

### Conclusion

The thesis compares machine learning and traditional econometric approaches to estimating causal effects and supports the use of these methods in academic and policy research, while pointing to open questions for future work in econometrics and statistics.

This was my master's thesis in economics at the University of Bonn.
