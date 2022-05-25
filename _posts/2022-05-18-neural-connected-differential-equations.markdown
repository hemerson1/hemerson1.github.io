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
- label all the equations with numbers and refer to them in the text.
- add greater spacing between subsections
- add some images to help illustrate the points.

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
	
	\tag{1}\label{eq:one}	
	
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

The most complex aspect of training a neural ODE is performing back-propagation (or *reverse-mode automatic differentiation*) through the ODE solver. In this instance, the gradients are computed by solving an additional ODE backwards in time, in a method referred to as the [**adjoint sensitivity method**][AdjointMethod]. The loss of the model can be described by:

$$
\begin{equation}

L(\boldsymbol{z}(t_1)) = 

L\left( \boldsymbol{z}(t_0) + \int^{t_1}_{t_0} f(\boldsymbol{z}(t_1), t, \theta)dt \right) = ODESolve(\boldsymbol{z}(t_0), f, t_0, t_1, \theta), 


\end{equation}
$$

where $$\boldsymbol{z}$$ is the hidden state. To optimise this loss, the gradient of the function must initially be computed with respect to the hidden state at each instance. This quantity defines the **adjoint state** and is given by:

$$
\begin{equation}

\boldsymbol{a}(t) = \frac{dL}{d\boldsymbol{z}(t)}.

\end{equation}
$$

The dynamics of the adjoint state can be determined by considering a transformation of the hidden state under a change in time, $$\epsilon$$.

$$
\begin{equation}

z(t + \epsilon) = \int^{t + \epsilon}_t f(\boldsymbol{z}(t), t, \theta) + \boldsymbol{z}(t) = T_{\epsilon}(\boldsymbol{z}(t), t), 

\end{equation}
$$

where $$\frac{d\boldsymbol{z}(t)}{dt} = f(\boldsymbol{z}(t), t, \theta)$$. By using the chain rule and the above equation the adjoint state can be redefined as:

$$
\begin{equation}
  \boldsymbol{a}(t) = \frac{dL}{d\boldsymbol{z}(t)} 
                    
                    = \frac{dL}{d\boldsymbol{z}(t + \epsilon)} \frac{d\boldsymbol{z}(t + \epsilon)}{d\boldsymbol{z}(t)} 
                    
                    = \boldsymbol{a}(t + \epsilon) \frac{\partial T_\epsilon(\boldsymbol{z}(t), t,)}{\partial \boldsymbol{z}(t)}.
  
\end{equation}
$$

The derivative of the adjoint state then follows from the above equation:

$$
\begin{align*}
  
  \frac{d\boldsymbol{a}(t)}{dt} &= \lim_{\epsilon \to 0^+} \frac{\boldsymbol{a}(t + \epsilon) - \boldsymbol{a}(t)}{\epsilon} \\  
  
  &= \lim_{\epsilon \to 0^+} \frac{\boldsymbol{a}(t + \epsilon) - \boldsymbol{a}(t + \epsilon) \frac{\partial}{\partial \boldsymbol{z}(t)} T_\epsilon(\boldsymbol{z}(t))}{\epsilon} && \text{(from the above equation)} \\ 
    
  &= \lim_{\epsilon \to 0^+} \frac{\boldsymbol{a}(t + \epsilon) - \boldsymbol{a}(t + \epsilon) \frac{\partial}{\partial \boldsymbol{z}(t)} \left( \boldsymbol{z}(t) + \epsilon f(\boldsymbol{z}(t), t, \theta) + \mathcal{O}(\epsilon^2) \right)}{\epsilon} && \text{(Taylor series expansion)} \\
  
  &= - \boldsymbol{a}(t) \frac{\partial f(\boldsymbol{z}(t), t, \theta)}{\partial \boldsymbol{z}(t)} && \text{(Rearranging and taking the limit)}
  
\end{align*}
$$

To obtain a solution to the adjoint state for a given time, the above ODE needs to be solved backwards in time with respect to the last time point.

$$
\begin{equation}

\boldsymbol{a}(t_0) = \frac{dL}{d\boldsymbol{z}(t_N)} - \int^{t_0}_{t_1} \boldsymbol{a}(t)^T \frac{\partial f(\boldsymbol{z}(t), t, \theta)}{\partial \boldsymbol{z}(t)}

\end{equation}
$$

The above equation can be generalised to determine the derivate of the loss with respect to $$\theta$$ by defining $$\theta$$ and $$t$$ as states with constant differential equations:

$$
\begin{equation}

\frac{\partial \theta (t)}{\partial t} = \boldsymbol{0} \qquad \frac{dt(t)}{dt} = 1.

\end{equation}
$$

These differential equations can be combined with $$z$$ to create an augmented state: 

$$
\begin{equation}

\frac{d}{dt} \begin{bmatrix} \boldsymbol{z} \\ \boldsymbol{0} \\ 1 \end{bmatrix} (t) = f_{aug}([\boldsymbol{z}, \theta, t]) := \begin{bmatrix} f([z, \theta, t]) \\ \boldsymbol{0} \\ 1 \end{bmatrix},  

\boldsymbol{a}_{aug} := \begin{bmatrix} \boldsymbol{a} \\ \boldsymbol{a}_\theta \\ \boldsymbol{a}_t \end{bmatrix}, 

\boldsymbol{a}_\theta := \frac{dL}{d\theta(t)}, 

\boldsymbol{a}_t := \frac{dL}{dt(t)}.

\end{equation}
$$

From the above definition it is clear that the Jacobian of the augmented state is defined as:

$$
\begin{equation}

\frac{df_{aug}}{d[\boldsymbol{z}, \theta, t]} = 

\begin{bmatrix} 
\frac{\partial f}{\partial \boldsymbol{z}} & \frac{\partial f}{\partial \theta} & \frac{\partial f}{\partial t} \\
\boldsymbol{0} & \boldsymbol{0} & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{0} & \boldsymbol{0} \\

\end{bmatrix}

\end{equation}
$$

The Jacobian can then be combined with the above equation to yield:


$$
\begin{equation}

\frac{d \boldsymbol{a}_{aug}(t)}{dt} = 

-\begin{bmatrix} 
\boldsymbol{a}\frac{\partial f}{\partial \boldsymbol{z}} &

\boldsymbol{a}\frac{\partial f}{\partial \theta} &

\boldsymbol{a}\frac{\partial f}{\partial t} 
\end{bmatrix}

\end{equation}
$$

Taking the second term of this equation and defining $$\boldsymbol{a}_\theta = \boldsymbol{0}$$ yields the expression:


$$
\begin{equation}

\frac{dL}{d\theta} = - \int^{t_0}_{t_N} \boldsymbol{a}(t) \frac{\partial f(\boldsymbol{z}(t), t, \theta)}{\partial \theta} dt

\end{equation}
$$

The final component of the above equation can then be used to determine the gradients of the remaining free parameters $$t_0$$ and $$t_N$$:

$$
\begin{equation}

\frac{dL}{dt_N} = \boldsymbol{a}(t_N) f(\boldsymbol{z}(t_N), t_N, \theta), \quad

\frac{dL}{dt_0} = \boldsymbol{a}(t_N) - \int^{t_0}_{t_N} \boldsymbol{a}(t) \frac{\partial f(\boldsymbol{z}(t), t, \theta)}{\partial t} dt.

\end{equation}
$$

## 2. Implementation

<!-----------------------------------------------------------
TODO:


------------------------------------------------------------->

[NODEPaper]: https://arxiv.org/abs/1806.07366
[NCDEPaper]: https://arxiv.org/abs/2005.08926
[EulerMethod]: https://tutorial.math.lamar.edu/classes/de/eulersmethod.aspx
[AdjointMethod]: https://towardsdatascience.com/the-story-of-adjoint-sensitivity-method-from-meteorology-906ab2796c73


[NODEIntro]: https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html



