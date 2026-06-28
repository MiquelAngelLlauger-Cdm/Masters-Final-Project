# Topological Data Analysis for Clinical Trial Data: A study of Treatment Retention in Opioid Use Disorder

## *Master's Final Project*


This repository contains all the code used for the thesis as well as the datasets for the trial CTN-0051 X:BOT

### Data

Both raw and clean data can be found in the data folder, algonside the data cleaning Python notebooks. 


### Metric Graph pipeline

In the Algorithms/mapper_pipeline folder lies the Metric Graph implementation. For running a demo, go to run_mapper.ipynb and run the first cells for an inmediate visualization. It is recommended to activate the TFM_env Python environment (see TFM_env.yml file). 

The w8 and w12 folders contain previous experiments of the Metric Graph for a wide range of parameter combinations. 

### Graphs

The two chosen Metric Graph configurations A and B are stored in the graphs folder, one for each of the checkpoint weeks 8 and 12. 

### Models

The models folder contains the GCN implementation with validation related methods as a Python file. Notebooks a and b contain the training and validation for the GCNs on the four graphs. In the baseline folder there are validation statistics for the non-graph baseline models Rnadom Forest and Logistic Regression. The stats notebook just plots the results obtained in the other notebooks. 




