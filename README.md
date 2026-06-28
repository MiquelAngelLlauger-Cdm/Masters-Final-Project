# Topological Data Analysis for Clinical Trial Data: A study of Treatment Retention in Opioid Use Disorder

## *Master's Final Project*


This repository contains all the code used for the thesis as well as the datasets for the trial CTN-0051 X:BOT.

- **Author**: Miquel Àngel Llauger Suau
- **Program**: MSc in Fundamental Principles of Data Science
- **Institution**: University of Barcelona
- **Advisor**: Prof. Dr.Kostyiantin Drach


### Abstract 

The Mapper algorithm of Topological Data Analysis (TDA) captures the shape of a point cloud as a graph, but in doing so it collapses many data points into each node. This thesis introduces the Metric Graph, a variant of Mapper in which every point is retained as a vertex and a projection cover is used only to constrain the edges of an underlying ε-neighbourhood graph in a given metric space. Because each vertex now corresponds to a single subject, we enable node-level prediction via Graph Neural Networks (GNNs).
We apply the method to the National Institute on Drug Abuse (NIDA) CTN- 0051 Extended-Release Naltrexone vs. Buprenorphine-Naloxone (X:BOT) study for
Opioid Use Disorder (OUD), focusing on patient retention. We build Metric Graphs at trial weeks 8 and 12 using baseline and engagement features and train Graph Convolutional Networks (GCNs) to predict dropout, benchmarking against non-graph classifiers. We report the experimental results, discuss the parameter choice and assess whether GNNs outperform classical models when they operate on the Metric Graph.

### Contact

Feel free to contact me to discuss any issues, questions or comments. 

- Email: [mallaugers@gmail.com](mailto:mallaugers@gmail.com)
- Github: [MiquelAngelLlauger-Cdm](https://github.com/MiquelAngelLlauger-Cdm)


### Data

Both raw and clean data can be found in the data folder, algonside the data cleaning Python notebooks. 


### Metric Graph pipeline

In the Algorithms/mapper_pipeline folder lies the Metric Graph implementation. For running a demo, go to run_mapper.ipynb and run the first cells for an inmediate visualization. It is recommended to activate the TFM_env Python environment (see TFM_env.yml file). 

The w8 and w12 folders contain previous experiments of the Metric Graph for a wide range of parameter combinations. 

### Graphs

The two chosen Metric Graph configurations A and B are stored in the graphs folder, one for each of the checkpoint weeks 8 and 12. 

### Models

The models folder contains the GCN implementation with validation related methods as a Python file. Notebooks a and b contain the training and validation for the GCNs on the four graphs. In the baseline folder there are validation statistics for the non-graph baseline models Rnadom Forest and Logistic Regression. The stats notebook just plots the results obtained in the other notebooks. 




