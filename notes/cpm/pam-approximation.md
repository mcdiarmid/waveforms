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


## For SOQPSK

$$\rho_0(t) = C_0 \prod_{c=0}^{2L-1}u(t+A_{B_0,c}^{'}T)$$

$$\rho_1(t) = C_1 \prod_{c=0}^{2L-1}u(t+A_{B_1,c}^{'}T)$$

For SOQPSK-MIL

For the first term $\rho_0(t)$

$$\rho_0(t) = \prod_{c=0}^{1}u(t+A_{0,c}^{'}T)$$

$$\rho_0(t) = \left[
    u(t+0T) \times
    u(t+0T)
\right] = u^2(t)$$

And for the second term $\rho_1(t)$

$$\rho_1(t) = 2 \prod_{c=0}^{1}u(t+A_{1,c}^{'}T)$$

$$\rho_1(t) = 2 \left[
    u(t+0T) \times
    u(t+1T)
\right] = 2u(t)u(t+T)$$


| $k$ | $v_{k,n}$ | $\rho_k(t)$ |
| --- | ----------|-------------|
| $0$ | $a_{0,n}$ | $u^2(t)$    |
| $1$ | $\frac{1}{2}(a_{1,n} + a_{2,n})$ | $2u(t)u(t+T)$    |

For SOQPSK-TG

For the first term $\rho_0(t)$

$$\rho_0(t) = \prod_{c=0}^{15}u(t+A_{0,c}^{'}T)$$

$$\rho_1(t) = 2 \left[
    u^2(t+0T) \times
    u^2(t+1T) \times
    ... \times
    u^2(t+7T)
\right]$$

And for the second term $\rho_1(t)$

$$\rho_1(t) = 2 \prod_{c=0}^{15}u(t+A_{1,c}^{'}T)$$

$$\rho_1(t) = 2 \left[
    u(t+0T) \times
    u^2(t+1T) \times
    u^2(t+2T) \times
    ... \times
    u^2(t+7T) \times
    u(t+8T)
\right]$$
