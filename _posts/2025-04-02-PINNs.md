# PINN_Examples
Solve PDEs with PINN method_deepXDE package:


Here (https://pinns-karkheiran.streamlit.app) some examples of solving PDE systems using physics informed neural networks (PINN) are given and deployed . In these examples we will:

1. Solve various PDE/ODEs with different types of boundary conditions,
2. Enforce boundary conditions either directly or by hard constraints,
3. Simulate a situations where an experimenter collect some data (with errors), and identify the paramters of the underlying theory by PINN,
4. In the main Jupyter notebook, the preformance analysis of each case is given.

The models are deployed by streamlit, and anyone can change the input paramters to check the solutions. The link can be found in the main page. All examples here are based on deepXDE. In another project (https://github.com/mohsenkarkheiran/Finance) we used another code for PINN (given to students in PINN course of UAlberta) which is written from scratch. In that project we used custom activation functions and automatic differentiation to compute the risk factors of market (as well as paramter identification for finding implied volatility).

The examples here are part of the exam of the PINN course in University of Alberta, presented by Prof. V. Putkaratze.
