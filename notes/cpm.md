# Continuous Phase Modulation

Continuous phase modulation (CPM) encodes all information as phase, with no variance in amplitude - maintaining a "constant envolope".
CPM waveforms provide excellent spectral and power efficiency, at the expense of receiver complexity.

## Describing CPM

The following equations used to describe to CPM has been taken from the excellent thesis of Erik Perrins.
I'll just be summarizing equations, some very brief observations, and a table of symbols.
For a detailed description and references to source material, please refer to section 1.3.
Erik's thesis presents approximations and simplifications for CPM receivers, reducing complexity allowing for realizable FPGA implementation [[1]][reduced-cpm].


$$s(t;\alpha) = \sqrt{\frac{E}{T}}\exp{\{j\psi(t; \boldsymbol{\alpha})\}}$$

Constant amplitude $\sqrt{\frac{E}{T}}$, all information data is encoded in phase.

$$
\psi(t; \boldsymbol{\alpha}) \triangleq 
2\pi \sum_{i=-\infty}^{n} \alpha_i h_{\underline{i}}q(t-iT)
,\quad\quad
nT \leq t \le (n+1)T
$$

$\boldsymbol{\alpha}=\{\alpha_i\}$ represents the $M$-ary alphabet (length $M$) of data symbols.

$$
\underline{i} \triangleq i \mod N_h
$$

Modulation index cycles through a set $\{h_{\underline{i}}\}$ of length $N_h$.
For the special case of $N_h=1$, known as "single-h", modulation index is constant.
CPM with $N_h \gt 1$ is known as "multi-h" CPM.

$$
h_{\underline{i}} \triangleq K_{\underline{i}}/p
$$

$\{h_{\underline{i}}\}$ is always described as a set of fractions with numerator $K_i$ and lowest common denominator $p$.

$$
q(t) \triangleq f(t) dt
, \quad\quad
q(LT) = \int_{0}^{LT} f(t) dt = \frac{1}{2}
$$

$L=1$ CPM are _full-response_ i.e. no inter-symbol-interference (ISI).
$L>1$ is said to be _partial-response_.
While the ISI of partial response increases detector complexity, the advantage of $L>1$ is increased spectral efficiency.


The phase signal can be split into three terms

$$
\psi(t; \boldsymbol{\alpha}) \triangleq 
\theta(t; \boldsymbol{\alpha_n}) + \theta_{n-L} + \phi_{n}
$$

are first, _correlative phase_ (a function of the $L$ most recent symbols $\alpha_n$)

$$
\theta(t; \boldsymbol{\alpha_n}) \triangleq 
2\pi\sum_{i=n-L+1}^{n} \alpha_{i} h_{\underline{i}} q(t-iT)
$$

second, _phase state_

$$
\theta_{n-L} \triangleq 2\pi\sum_{i=-\infty}^{n-L} U_i h_{\underline{i}}
$$

and third, _phase tilt_

$$
\phi_{n} \triangleq \phi_{n-1} - \pi h_{\underline{n}}(M-1)
$$

Wait what's $U_i$ and where did it come from?  After removing the _correlative phase_ term we're left with 

$$
2\pi \sum_{i=-\infty}^{n-L} \alpha_i h_{\underline{i}} q(t - iT)
$$

but, since since $q(t)=1/2$ for all symbols prior to the most recent $L$ symbols, this can be simlified to 

$$
\pi \sum_{i=-\infty}^{n-L} \alpha_i h_{\underline{i}}
$$

Alternative data symbols are defined as 

$$
U_i \triangleq (\alpha_i + M - 1) / 2
,\quad\quad
U_i \in {0, 1, \cdots, M-1}
$$

and substituded in for $\alpha_i$

$$
\pi \sum_{i=-\infty}^{n-L} \left(2U_i - (M-1)\right) h_{\underline{i}}
$$

This substitution of $U$ allows us to split the expression into two parts, a function of all data symbols prior to $L^{th}$ most recent, and an indefinitely increasing, data-independent offset that has been labelled as _phase tilt_.

$$
2\pi \left( \sum_{i=-\infty}^{n-L} U_i h_{\underline{i}} \right) - 
\pi \left( \sum_{i=-\infty}^{n-L}  h_{\underline{i}} \right)(M-1)
$$

As $n$ increments, the phase tilt $\phi_n$ decreases by $\pi h_{\underline{n}}(M-1)$, hence the recursive definition for _phase tilt_ mentioned previously.

Going back to the _phase state_ expression, despite this being a function of an infinite number of terms, it can be reduced to $p$ unique values when taking the modulo $-2\pi$ of the expression.
The modulo $-2\pi$ operation is often expressed in literature as $R_{2\pi}[.]$
A modulo $-p$ look-up table for the phase state can be defined as

$$
\theta[v] \triangleq \frac{2\pi}{p} \times [v  \mod p]
$$

and the _phase state index_ can be defined as

$$
I_{n-L} \triangleq
\left(\sum_{i=-\infty}^{n-L} U_i K_{\underline{i}}\right)  \mod p
$$

Breaking the original phase into the three aforementioned terms proves that CPM signals can be described as a trellis with finite states.
The input variable is the current symbol, $\alpha_n$.

| FSM Memory                  | Length | Description |
|  --                         | --     | --  |
| $\underline{n}$             | 1      | This can be tracked with a register, or by using a time-varying trellis definition. |
| $\boldsymbol{\alpha_{n-1}}$ | $L$    | $(I_{n-L}, \alpha_{n-L+1}, \cdots , \alpha_{n-2}, \alpha_{n-1})$ |

The number of finite states to fully describe the CPM signal with a trellis is

$$N_S = pM^{L-1}$$

as there are $M^{L-1}$ valid trellis paths from $p$ starting points.


There were a lot of symbols used here!
The table below summarizes all mathematical symbols mentioned in the description of general CPM.

| Symbol                  | Description       |
| :--:                    | :---------------- |
| $E$                     | Symbol energy     |
| $T$                     | Symbol duration   |
| $\psi(.)$               | Symbol phase      |
| $\alpha_i$              | $i^{th}$ symbol   |
| $M$                     | Symbol alphabet length |
| $N_h$                   | Number of modulation indexes |
| $\underline{i}$         | $i$ modulo $-N_h$  |
| $h_{\underline{i}}$     | $i^{th}$ modulation index |
| $p$                     | Lowest common denominator of $h_{\underline{i}}$ |
| $K_{\underline{i}}$     | Numerator of $h_{\underline{i}}$ |
| $q(t)$                  | Phase pulse       |
| $f(t)$                  | Frequency pulse   |
| $\theta(.)$             | Corrolative phase |
| $\phi_n$                | Phase tilt |
| $\boldsymbol{\alpha_n}$ | Corrolative state vector |
| $L$                     | Phase pulse length |
| $U_i$                   | Alternate $M$-ary digit data symbol |
| $I_{n-L}$               | Phase state index |
| $R_{2\pi}[.]$           | Modulo $-2\pi$ function |

To re-iterate, please check out Erik's thesis [[1]][reduced-cpm] for full explanations and references to source literature.
This summary is primarily for my own understanding.

## References

[[1]][reduced-cpm]
Reduced Complexity Detection Methods for Continuous Phase Modulation.

[reduced-cpm]: https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1619&context=etd
