---
layout: post
title: "Causal Inference Using Random Forests and LASSO"
subtitle: "Leveraging ML methods for estimation of treatment effects."
background: '/img/posts/master-thesis/normality.png'
---


It is no secret that I'm a big supporter of using machine learning for causal inference. My interest in this field can be traced back to a ***Computational Statistics*** course I took with Marina Khismatullina at Uni Bonn in 2021. I was intrigued by how well machine learning method peformed in predictive scenarios with a large number of covariates and enjoyed learning comparing different ML models such as LASSO, Random Forest, XGBoost, and Support Vector Machines. We also learned about model training/test, cross-validation, and the bias-variance trade-off, where ML methods tend to sacrifice a little bit of bias for lower variance in their point estimates.

But in economics, we generally care about issues of causality, where bias, variance, and valid confidence intervals are the key considerations for any estimator. So my question was, how can we re-purpose the strength and flexibility of ML in large covariate spaces for high-dimensional problems in statistical inference? This issue is becoming even more important nowadays since there has been a drastic increase in the availability and granularity of data. 

Over the course of my studies, I concentrated heavily on this topic and took several classes in econometrics and statistics to better understand the potential merger between machine learning's superior predictive capabilities on one hand and the causal framework of traditional econometrics on the other. The result of this two-year effort was my master thesis "Causal Inference Using Random Forests and LASSO,", where I explored innovative methods leveraging machine learning algorithms to evaluate causal effects, specifically through simulations and empirical applications. In this post, I would like to show the promise of these methods in specific econometric and highlight use-cases how this could help your research.

## Introduction and Theoretical Background
### Research Motivation

The study was motivated by the growing need for robust statistical methods capable of handling large, complex datasets often encountered in economics. Traditional econometric approaches sometimes fall short when addressing high-dimensional data and highly complex nuisance parameters. This research focused on integrating machine learning techniques, particularly Random Forests and LASSO, to enhance causal inference in econometric models.

### Prior Work
My research was heavily inspired by the extensive works of Susan Athey (Stanford) and Victor Chernozhukov (MIT) in the space of ML for estimation of treatment effects. To that end, I first provided a comprehensive review of existing methods and their limitations, setting the stage for the introduction of Generalized Random Forests (GRF), Double Machine Learning (DML), and Post-Double-Selection approaches. These methods were hand=picked for their potential to improve the accuracy and reliability of treatment effect estimations in complex econometric models.

### Methodological Approach
As a very cursory glance at these methods:
- Generalized Random Forests (GRF): This non-parametric method adapts the random forest algorithm to estimate treatment effects, utilizing a local linear approach that accommodates various covariates and complex interactions.
- Double Machine Learning (DML): This technique separates the estimation of nuisance parameters from the target parameter, reducing bias that I mentioned earlier and enhancing the robustness of causal inferences.
- Post-Double Selection: Leveraging LASSO for variable selection, this method aims to improve model accuracy by focusing only on the most relevant variables affecting the outcome. This is especially useful for guiding variable selection. 

### Simulation Studies
I used a series of Monte Carlo simulations to test the performance of these estimators under different scenarios—ranging from randomized controlled trials to settings with strong confounding. These simulations were crucial for demonstrating the practical applicability and robustness of the ML methods in mimicking real-world data complexities.

### Empirical Applications
The theoretical insights and simulation results were applied to two empirical cases:

#### Microcredit's Impact in Bosnia and Herzegovina: Analysis focused on the economic effects of microcredit availability using data from a prior study.
401(k) Participation and Financial Assets: This study evaluated how eligibility and participation in 401(k) plans affect net financial assets, showcasing the practical implications of these ML techniques in policy evaluation.
#### Key Insights
Robustness of ML Estimators: The ML-based methods, particularly GRF and DML, showed strong performance in handling complex, high-dimensional datasets and produced more reliable causal inferences compared to traditional methods.
#### Practical Implications: 
The empirical applications illustrated how these advanced statistical techniques could be used to inform policy decisions and improve economic outcomes.
#### Theoretical Contributions
The thesis extended existing theories in econometrics by integrating and enhancing machine learning algorithms for better causal inference, providing a new framework that can be adopted in various economic and policy research areas.

### Conclusion
This research bridged the gap between traditional econometric approaches and modern machine learning techniques, offering substantial improvements in the estimation of causal effects. The findings not only support the use of these advanced methods in academic and policy-making contexts but also open avenues for future research in econometrics and statistics.

This thesis, supervised by Prof. Dr. Christoph Breunig at Rheinische Friedrich-Wilhelms-Universität Bonn, represents a significant step forward in the use of machine learning to enhance causal inference, demonstrating the potential of integrating cutting-edge statistical techniques in economic research.