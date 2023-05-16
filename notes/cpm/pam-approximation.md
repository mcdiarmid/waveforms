# PAM Approximation of CPM Signals

From the binary CPM case for PAM representation:

$$s_{bin}(t;\boldsymbol{\gamma}) = 
\sum_{k=0}^{Q-1}\sum_{n} b_{k,n}^{(l)}c_k(t-nT)
,\quad\quad
Q=2^{L-1}$$

where pulses are given by

$$c_k(t) = \prod_{i=0}^{L-1} u(t+iT+LT\beta_{k,i})
,\quad\quad
0 \le k \le Q-1$$

and $u(t)$ is defined as

| $$u(t)$$ | Condition | 
| ---  | ----------|
| $$\sin(2\pi h q(t))/\sin(\pi h)$$ | $$0 \le t \lt LT$$ |
| $$\sin(\pi h - 2\pi h q(t-LT))/\sin(\pi h)$$ | $$LT \le t \lt 2LT$$ |
| $$0$$ | otherwise |


$$\alpha_i = \gamma_{1,i} + \gamma_{0,i}, \quad\quad
\gamma_{1,i},  \gamma_{0,i} \in \{\pm1\}$$

$$s(t;\boldsymbol{\alpha}) = 
s_{bin}(t;\boldsymbol{\gamma}_{1})
s_{bin}(t;\boldsymbol{\gamma}_{0})$$


$$s(t;\boldsymbol{\alpha}) =
\prod_{t=0}^{1}\sum_{k=0}^{Q-1}\sum_{n}
b_{k,n}^{(l)}c_k(t-nT)$$

$$s(t;\boldsymbol{\alpha}) = 
\sum_{k=0}^{N-1}\sum_{n}a_{k,n}g_k(t-nT)
, \quad\quad N=3\times 4^{L-1}
$$


$$b_{k,n}^{(l)} = \exp\left(j\pi h \left[ 
    \sum_{m=-\infty}^{n} \gamma_{l,m} + 
    \sum_{i=0}^{L-1} \gamma_{l,n-i} \beta_{k,i}
\right] \right)$$

$$v_{k,n} = \frac{1}{C_l}  \sum_{k:\bold{A}_k^{'}=\bold{A}_{B_l}^{'}}^{1} a_{k,n}$$

$$\rho_{l}(t) = C_l \prod_{c=0}^{2L-1}u(t+A_{B_l, c}^{'}T)$$

$$\tilde{s}(t;\alpha) = \sum_{k=0}^{1}\sum_{n}v_{k,n}\rho_{k}(t-nT)$$