---
layout: post
title:  "Moore-Penrose Pseudoinverse"
date:   2023-03-19 21:59:00 +0000
categories: Scientific Computing
---

The Moore-Penrose Pseudoinverse provides a method for approximating the inverse of a non-invertible matrix. This post briefly discusses the theory and applications behind this important tool in linear algebra.


### Overview

For a matrix to be **invertible** it must be **square** (have an equal number of rows and columns), and it must be **non-singular** (have a non-zero determinant). The satisfaction of  these conditions by a matrix $$\textbf{A} \in M_{n \times n}$$ implies that there exists another matrix $$\textbf{B} \in M_{n \times n}$$ such that:

\\[\textbf{A} \textbf{B} = \textbf{B} \textbf{A} = \textbf{I}_{n}, \\]

where $$\textbf{I}_{n}$$ denotes the $$n$$-by-$$n$$ identity matrix and $$\textbf{B}$$ is the inverse of $$\textbf{A}$$, denoted by $$\textbf{A}^{-1}$$. For a square matrix, the inverse can be determined as:

\\[ \textbf{A}^{-1} = \frac{\boldsymbol{1}}{\det{\boldsymbol{A}}} \text{adj}(\boldsymbol{A}), \\]

where $$\text{adj}(\boldsymbol{A})$$ is the [adjoint](https://byjus.com/maths/adjoint-of-a-matrix/) of the matrix. For an $$n$$-by-$$n$$ matrix, the adjoint can be calculated exactly using methods such as [Gauss-Jordan Elimination](https://mathworld.wolfram.com/Gauss-JordanElimination.html) or [LU Decomposition](https://mathworld.wolfram.com/LUDecomposition.html). When a matrix is non-invertible, **singular value decomposition** (SVD) provides a method for constructing the inverse. This theorem states that any matrix $$\textbf{A}$$ can be decomposed as: 

\\[\textbf{A} = \textbf{U} \boldsymbol{\Sigma}\textbf{V}^{T}, \\]

where $$\textbf{U}$$ and $$\textbf{V}$$ are orthogonal matrices and $$\boldsymbol{\Sigma}$$ is a diagonal matrix. The inverse can then be determined from:

\\[\textbf{A}^{-1} = \textbf{V} \boldsymbol{\Sigma}^{-1} \textbf{U}^{T}. \\]

This simplifies the problem as $$\boldsymbol{\Sigma}^{-1}$$ is zero-valued except for along the diagonal and therefore its inverse can be computed by taking the reciprocal of the non-zero values in $$\boldsymbol{\Sigma}$$. SVD is also applicable for calculating the pseudoinverse of a matrix $$\boldsymbol{A}$$, denoted by $$\boldsymbol{A}^{+}$$. This can be done by replacing the inverse matrices in the above formula with their pseudoinverse and by identifying that $$\boldsymbol{\Sigma}^{+}$$ can be computed by transposing $$\boldsymbol{\Sigma}$$ and taking the reciprocal of its non-zero elements as before. A summary of the how SVD is performed can be found [here](https://towardsdatascience.com/simple-svd-algorithms-13291ad2eef2).

### Applications

Matrix inversion is well-utilised across a range of topics in science and mathematics, most notably in solving systems of linear equations. Suppose there exists a system described by: 

\\[\textbf{A} \textbf{x} = \textbf{b}, \\]

where $$\textbf{A}$$ and $$\textbf{b}$$ are known matrices and $$\textbf{x}$$ is an unknown vector which can be computed from $$\textbf{x} = \textbf{A}^{-1} \textbf{b}$$. If the matrix $$\textbf{A}$$ is square, then system can be solved exactly as the number of linear equations in the system and variables in $$\textbf{x}$$ are equal. However, in most real-world systems these quantities are not equal and therefore there exists infinitely many or no solutions to the system of linear equations. The **Moore-Penrose Pseudoinverse** provides a means of approximating the matrix inversion and hence findng a solution to the linear system. 