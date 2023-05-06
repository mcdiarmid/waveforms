# Shaped-Offset Quadrature Phase-Shift Keying

## Frequency Pulses

SOQPSK-MIL's frequency pulse, $f_{MIL}(t)$, can be described with the following table

| $f_{MIL}(t)$ | Case |
| ---- | ------ |
| $\frac{1}{2T}$ | $0 \le t \lt T$ |
| 0 | otherwise|


The SOQPSK-TG frequency pulse, $f_{TG}(\tau)$, can be described as

$$
f_{TG}(\tau) = A\frac{cos(\pi \rho B \tau)}{1-4(\rho B \tau)^2}
\times
\frac{\sin({\pi B \tau})}{\pi B\tau}
\times w(\tau)
$$

where windowing function $w(\tau)$ is

| $w(\tau)$ | Case |
| ---- | ------ |
| 1 | $ 0 \le \|\tau\| \lt T_1$ |
| $\frac{1}{2} + \frac{1}{2} \cos \left( \frac{\pi}{T_2} (\tau - T_1 ) \right)$ | $T_1 \le \|\tau\| \le T_1 + T_2$ |
| 0 | $ T_1 + T_2 \lt \|\tau\|$ 

a substitution of $\tau = t/2T$ has been used to simplify the SOQPSK-TG expressions.