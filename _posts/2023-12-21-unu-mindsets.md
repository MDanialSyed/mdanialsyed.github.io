---
layout: post
title: "Cluster Analysis for Urban Sustainability Transformation"
subtitle: "Applied machine learning techniques in the field of urban sustainability and climate action."
background: '/img/posts/unu-mindsets/cover5.jpg'
---


In 2022/23, I had the great privilege of joining the United Nations University's Institute of Environment and Human Security (UNU-EHS) in Bonn, Germany, as a consultant researcher. At the time, I was a Uni Bonn student working on my master thesis on machine learning for causal inference. The Transformative Urban Coalitions (TUC) team at UNU-EHS was conducting research on mindsets regarding urban sustainability transformation in Latin America and was looking for someone to generate insights on this through survey analysis. And so, they brought me on onboard in July 2022. 

This post describes the application of advanced machine learning for statistical survey analysis. It is exemplified by the research I conducted in TUC under the guidance of my mentor, and later co-author, Dr. Guerra, which culminated in the ***Sustainability Science*** publication titled "Identifying Mindsets for Urban Sustainability Transformation: Insights from Urban Labs". 

As a quick background, my colleagues and I wanted to understand how different mindsets influence urban sustainability initiatives. We gathered data from participants in Urban Labs across Argentina, Brazil, and Mexico, employing advanced machine learning techniques to analyze the data and identify distinct stakeholder personas. Here's a detailed overview of our methodology, focusing particularly on our innovative use of Hierarchical Clustering on Principal Components (HCPC) for data analysis and the deep insights we gained into the distinct personas involved in urban sustainability.

## **Data Analysis and Methodological Approach**

### **Survey Design and Data Collection:** 
We conducted a detailed survey involving 153 participants from five Urban Labs, which included diverse stakeholders such as government officials, community members, researchers, and representatives from the private sector. The data collection was executed through computer-assisted telephone interviews over two rounds, ensuring a comprehensive view of different perspectives.

### **Questionnaire:**
The survey was carefully designed with 40 questions aimed at capturing a range of mindset dimensions—beliefs, emotions, perceptions, attitudes, values, and worldviews—related to climate change and urban sustainability. These questions were crafted based on a review of existing literature and were adapted to fit the context of our study, using both quantitative and categorical variables.

### **Cluster Analysis:**
Our data analysis involved a two-tiered approach:
1. **Descriptive Statistics**: We first explored the dataset through descriptive statistics to understand basic trends and distributions across the various Urban Labs.
2. **Hierarchical Clustering on Principal Components (HCPC)**: We applied HCPC, a sophisticated machine learning technique that combines Principal Component Analysis (PCA) with hierarchical clustering. This method is particularly effective for datasets like ours that contain mixed types of data, allowing us to identify naturally occurring groups in our data based on shared characteristics in mindset-related responses.

![Book logo](/img/posts/unu-mindsets/cluster_map.png)

### **Detailed HCPC Analysis:**
- **Principal Component Analysis (PCA)**: Before clustering, we used PCA to reduce the dimensionality of our data, which clarifies the relationships between variables and enhances the interpretability of our clustering analysis. PCA helps to condense the data into principal components that capture the most variance and patterns within the dataset.
- **Hierarchical Clustering**: Following PCA, hierarchical clustering was applied to the principal components. This method sorts data points into clusters based on their similarity, creating a dendrogram that illustrates how each cluster is related. Hierarchical clustering is particularly useful for revealing the hierarchical relationships between clusters, making it easier to understand the grouping logic.
- **Optimizing Cluster Selection**: We employed methods such as the elbow method and silhouette analysis to determine the optimal number of clusters. These techniques help to assess the cohesion and separation of clusters, ensuring that each cluster is both distinct from others and internally homogeneous.
- **Multivariate Cluster Analysis**: The HCPC combines PCA and hierarchical clustering, enhancing our ability to handle the complex, multi-dimensional data collected. By using multiple factor analysis (MFA) before hierarchical clustering, we grouped variables by related themes, which improved the quality and interpretability of our clusters.

### **Identification of Personas:**
The HCPC enabled us to discern three distinct personas among the study participants, each representing a unique combination of beliefs, values, and attitudes towards urban sustainability and climate change:

1. **The Skeptical Activist**: This persona is characterized by a strong commitment to community-driven initiatives but exhibits skepticism towards government-led interventions. They prioritize grassroots movements and local actions, believing that true change happens at the community level.

2. **The Optimist Technocrat**: Stakeholders in this group have a strong faith in technological solutions and government policies. They believe that structured, well-funded governmental programs, supported by the latest technological advancements, are the key to achieving sustainable urban environments.

3. **The Bystander with Mixed Feelings**: Individuals in this category show varied levels of engagement and often possess conflicting feelings about their roles in sustainability efforts. They acknowledge the importance of sustainability but feel disconnected from the direct impact of their actions, often due to a lack of clear direction or overwhelming climate change narratives.

## **Insights and Implications:**
Our advanced machine learning-based analysis, particularly the use of HCPC, revealed critical insights into how diverse mindsets can influence the effectiveness of urban sustainability initiatives. By identifying and understanding these personas, urban planners and policymakers can tailor their strategies to better align with the motivations and perceptions of different stakeholder groups, potentially enhancing engagement and the efficacy of sustainability measures.

## **Conclusion:**
We demonstrated the application of sophisticated machine learning techniques, particularly HCPC, in the field of urban sustainability, providing a methodological blueprint for future research. This study underscores the importance of integrating advanced data analytics to deeply understand the complex interplay of human factors in urban environmental initiatives. The insights gained not only shed light on the current state of urban sustainability mindsets but also pave the way for targeted interventions that can drive more profound and effective transformations in urban settings worldwide.
