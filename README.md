# Code - Job Market Paper

### Title: **Balancing Economic Activity and Carbon Emissions: A Study of Fuel Substitution in India**

## Emmanuel Murray Leclair

## Institution: University of Western Ontario, Department of Economics

## Contact: emurrayl@uwo.ca

### Last updated: August 2023
### The following details are applicable to the August 17th 2023 version of the paper, available here: [emurraylJMP_aug2023](https://drive.google.com/file/d/1J4am43imZd3f3-ruPvSuxbK_UZN_2tfD/view?usp=sharing)

This repository contains the code for my job market paper.

- In "Stata Do-files", you will find all the code necessary from importing the data to estimating the production function. Each do-file is in order.

- In "Julia Code", you will find all the code necessary to estimate the dynamic parameters of the model and perform counterfactuals. Note that most of the code-files are written for a cluster, and require access to an NVIDIA GPU. Each code file is in order.

In the following code-files, you will be able to generate all figures and tables appearing in the paper

	--------------------------------- Created in Stata --------------------------------------
	1) 7a.GraphTable_Evidence.do
	  Table 1, Table 2, Figure 1, Table 3, Figure 2, Table 4
	
	2) 7b.GraphTable_SpatialEvidence.do
	  Figure 3, Table 5
	
	3) 8.Program_OuterPFE
	  Table 7
	
	4) 9.Program_EnergyPFE
	  Table 8, Table 9, Figure 4, Table 11
	
	5) 10.Graphs_PostDynamics.do
	  Figure 5
	
	6) 11.Appendix_GraphsTables.do
	  Tables and figures in the appendix, excl. counterfactual
	
	--------------------------------- Created in Julia --------------------------------------
	7) 2.PostEstimation_ModelFit.jl
	  Table 10, Table 12, Figure 5, Figure 6, Figure 7
	  Appendix: Figure 27, Figure 28, Figure 29
	
	8) 4.Counterfactual_CompareModels-graphs.jl
	  Figure 8, Figure 9, Table 14, Table 15


