---
layout: post
title:  "Neural Ordinary Differential Equations"
date:   2022-05-18 21:44:59 +0100
categories: machine-learning
---

<!--------------------------------------------------------
TODO:
- summarise the function of the network
- explain in more detail why this method is worth knowing

---------------------------------------------------------->

**Neural ordinary differential equations** are a powerful tool for continuous time-series modelling. They hold strong advantages over more traditional methods of forecasting, such as recurrent neural networks, as they can be trained on samples of irregularly-spaced data and are robust to missing values. 


## 1. Mathematics

<!-----------------------------------------------------------
TODO:
- introduce ODEs -> DONE
- may want to discuss why solve for dy/dx and not y? 
- explain similarity Euler's equation and NN hidden structure -> DONE
- explain Reverse-mode automatic differentiation

------------------------------------------------------------->

# 1.1 Ordinary Differential Equations

A typical *ordinary differential equation* (ODE) can be defined by the following equation:

$$ 
\begin{equation}
	\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0.
\end{equation}
$$

In most instances, ODEs cannot be solved analytically and require numerical methods to find an approximate solution at discrete points. One of the simplest and most prominent examples of these methods is [**Euler's method**][EulerMethod]. This method is based on the idea that a curve can be approximated as a series of tangential line elements at fixed intervals, $$\delta$$:

$$ 
\begin{equation}
	\frac{dy}{dt} \approx \frac{y(t + \delta) - y(t)}{\delta}.
\end{equation}
$$

The above expression can be rearranged to the following form to allow the explicit approximation of $$y$$ at future timesteps:

$$ 
\begin{equation}
	y_{n+1} = y_n + \delta f(t_n, y_n), \quad n = 0, 1, 2, ...
\end{equation}
$$

# 1.2 Neural Ordinary Differential Equations

Neural ODEs were primarily inspired by the observed similarity between the above equation for approximating ODEs and the underlying structure of neural networks. The hidden state of a *residual neural network* can be expressed as:

$$
\begin{equation}
	h_{t+1} = h_t + f(h_t, \theta_t), 
\end{equation}
$$

where $$t$$ is the network depth and $$f$$ is a neural layer with parameters, $$ \theta_t $$. Just like Euler's method can be seen as discretisation of the continuous relationship between the inputs and outputs of an ODE, a neural network of this form can be treated as the discretisation of hidden states in a latent space. This suggests that in a similar way to how an analytical solution to an ODE may be obtained, each discrete layer in a network can be considered in the continuous limit as:

$$
\begin{equation}
	\frac{dh(t)}{dt} = f(t, h(t), \theta_t),
\end{equation}
$$ 

where the hidden state is parametrised as a continuous function of layer depth. In this limit the value of the hidden state can effectively be evaluated at any depth by computing:

$$
\begin{equation}
	h(t) = \int^{t_1}_{t_0} f(t, h(t), \theta_t) dt,
\end{equation}
$$

where $$h(t_0) = x$$ and $$h(t_1) = y$$. A solution to the above integral is then obtainable by using standard numerical ODE solver with the free parameters $$h(t_0)$$, $$t_0$$, $$t_1$$, $$f$$ and $$\theta$$. 

# 1.3 Back-propagation in a Continuous Depth Network 

The most complex aspect of training a neural ODE is performing back-propagation (*reverse-mode automatic differentiation*) through the ODE solver. In this instance, the gradients are computed by solving an additional ODE backwards in time, in a method referred to as the [**adjoint sensitivity method**][AdjointMethod]. The loss of the model can be described by:

$$
\begin{equation}

L(\boldsymbol{z}(t_1)) = 

L\left( \boldsymbol{z}(t_0) + \int^{t_1}_{t_0} f(\boldsymbol{z}(t_1), t, \theta)dt \right) = ODESolve(\boldsymbol{z}(t_0), f, t_0, t_1, \theta), 


\end{equation}
$$

where $$\boldsymbol{z}$$ is the hidden state. To optimise this loss, the gradient of the function must initially be computed with respect to the hidden state at each instance.


## Implementation

<!-----------------------------------------------------------
TODO:


------------------------------------------------------------->

[NODEPaper]: https://arxiv.org/abs/1806.07366
[NCDEPaper]: https://arxiv.org/abs/2005.08926
[EulerMethod]: https://tutorial.math.lamar.edu/classes/de/eulersmethod.aspx
[AdjointMethod]: https://towardsdatascience.com/the-story-of-adjoint-sensitivity-method-from-meteorology-906ab2796c73


[NODEIntro]: https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html



