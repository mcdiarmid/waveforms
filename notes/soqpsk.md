# Shaped-Offset Quadrature Phase-Shift Keying

## SOQPSK-MIL

$$
f_{MIL}(t) = 

\begin{cases}
\frac{1}{2T}, & 0 \le t \lt T \\
0, & \text{otherwise}
\end{cases}
$$

## SOQPSK-TG

$$

f_{TG}(t) = 

\begin{cases}
1, &
0 \le |\frac{t}{2T}| \lt T_1 \\

\frac{1}{2} + \frac{1}{2} \cos \left( \frac{\pi}{T_2} (\frac{t}{2T} - T_1 ) \right), &
T_1 \le |\frac{t}{2T}| \le T_1 + T_2 \\

0, &
T_1 + T_2 \lt |\frac{t}{2T}| \\
\end{cases}

$$