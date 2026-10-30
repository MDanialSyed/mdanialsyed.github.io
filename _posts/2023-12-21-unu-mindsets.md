---
layout: post
title: "Cluster Analysis for Urban Sustainability Transformation"
subtitle: "Applying machine learning to survey data on urban sustainability and climate action"
background: '/img/posts/unu-mindsets/cover5.jpg'
---

In July 2022, I joined the United Nations University Institute for Environment and Human Security (UNU-EHS) in Bonn, Germany, as a research assistant, while completing my master's thesis at the University of Bonn on machine learning for causal inference. The Transformative Urban Coalitions (TUC) team at UNU-EHS was studying mindsets on urban sustainability transformation in Latin America and needed survey analysis to inform this work.

This post describes how machine learning can be applied to survey analysis, using the research I contributed to at TUC as an example. The work was published in *Sustainability Science* as "Identifying Mindsets for Urban Sustainability Transformation: Insights from Urban Labs".

The team set out to understand how different mindsets influence urban sustainability initiatives. We collected data from participants in Urban Labs across Argentina, Brazil and Mexico and used machine learning to identify distinct stakeholder personas. The sections below outline the methodology, focusing on Hierarchical Clustering on Principal Components (HCPC), and the personas it identified.

## Data and Methods

### Survey Design and Data Collection

The survey covered 153 participants from five Urban Labs, including government officials, community members, researchers and private-sector representatives. Data were collected through computer-assisted telephone interviews in two rounds.

### Questionnaire

The questionnaire had 40 questions covering several dimensions of mindset (beliefs, emotions, perceptions, attitudes, values and worldviews) related to climate change and urban sustainability. The questions were based on a review of the existing literature, adapted to the context of the study, and included both quantitative and categorical variables.

### Cluster Analysis

The analysis had two stages:

1. **Descriptive statistics** to explore basic trends and distributions across the Urban Labs.
2. **Hierarchical Clustering on Principal Components (HCPC)**, which combines principal component analysis (PCA) with hierarchical clustering. The method suits datasets with mixed types of data, like this one, and groups respondents by shared characteristics in their mindset-related answers.

![Factor map of survey respondents grouped into three clusters on the first two principal dimensions](/img/posts/unu-mindsets/cluster_map.png)

### HCPC in Detail

- **Principal component analysis (PCA):** reduces the dimensionality of the data before clustering, condensing it into components that capture most of the variance and making the clusters easier to interpret.
- **Hierarchical clustering:** groups the principal components by similarity, producing a dendrogram that shows how the clusters relate to each other.
- **Choosing the number of clusters:** the elbow method and silhouette analysis assess the cohesion and separation of clusters, so that each cluster is internally similar and distinct from the others.
- **Multiple factor analysis (MFA):** grouping variables by theme before clustering improved the quality and interpretability of the clusters.

### Personas

HCPC identified three distinct personas among the participants, each with a different combination of beliefs, values and attitudes towards urban sustainability and climate change:

1. **The Skeptical Activist:** committed to community-driven initiatives but skeptical of government-led interventions. This group prioritizes grassroots movements and local action and believes change happens at the community level.
2. **The Optimist Technocrat:** confident in technological solutions and government policy. This group sees well-funded public programs, supported by new technology, as the key to sustainable cities.
3. **The Bystander with Mixed Feelings:** varied engagement and conflicting feelings about their role in sustainability efforts. This group acknowledges the importance of sustainability but feels disconnected from the impact of their own actions, often because of unclear direction or overwhelming climate narratives.

## Implications

Understanding these personas can help urban planners and policymakers tailor their strategies to the motivations and perceptions of different stakeholder groups, which may improve engagement with sustainability measures and their effectiveness.

## Conclusion

The study shows how HCPC can be applied to survey data in urban sustainability research. The personas it identified describe current mindsets on urban sustainability and can inform more targeted interventions.
