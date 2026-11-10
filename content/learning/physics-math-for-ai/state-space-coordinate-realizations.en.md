## Internal memory and measured output

A state is information sufficient to predict future system behavior when future inputs are specified. An output is what a chosen observation rule reports. They need not have the same dimension, and state coordinates are not unique. A controller or learned world model can use different internal coordinates while predicting the same input–output behavior.

The lab deliberately uses a discrete-time system. Its matrices are already per-step transition matrices; they are not continuous-time derivatives that can be substituted into an ODE without a sampling model.

## Specify the timing convention

The recurrence and observation are

$$
x_{t+1}=Ax_t+Bu_t,\qquad y_t=Cx_t+Du_t,
$$

$$
A=\begin{pmatrix}0.8&0.2\\0&0.5\end{pmatrix},\quad
B=\begin{pmatrix}0\\1\end{pmatrix},\quad C=\begin{pmatrix}1&0\end{pmatrix}.
$$

Compute y_t using the current state, then update the state. D is direct feedthrough: the current input can influence the current output even before the next state is formed. An impulse has u0 equal to the selected amplitude and all later inputs zero. A constant or sinusoidal input follows the explicitly displayed rule. Rows include times zero through N.

## Change coordinates without changing the system

Use an invertible shear

$$
T=\begin{pmatrix}1&s\\0&1\end{pmatrix},\qquad z_t=Tx_t.
$$

Substitute x_t=T^{-1}z_t:

$$
z_{t+1}=(TAT^{-1})z_t+(TB)u_t,\qquad
 y_t=(CT^{-1})z_t+Du_t.
$$

The transformed matrices are A_new=[[0.8,0.2-0.3s],[0,0.5]], B_new=(s,1)ᵀ and C_new=(1,-s). The initial condition must also become z0=T x0. Transforming only A while leaving B,C or the initial state untouched generally creates a different realization, not an equivalent coordinate description.

## What the visualization actually computes

The block diagram distinguishes current state, next state and current output. It shows the original x coordinates; transformed matrices appear in the readouts. The implementation executes both recurrences independently rather than defining the orange output to equal the blue one. The table exposes x and z components and both computed outputs. Their small floating-point discrepancy is checked numerically.

Changing s often changes the state columns substantially while the output curves overlap. Overlap is the expected invariant, not an inactive control. This also explains why arbitrary distances or penalties in latent coordinates are not automatically physical quantities: a coordinate change can alter them unless the metric is transformed too.

## Worked exercise and solution

Choose s=2, zero initial state and unit impulse, with D=0. Compute the first state update and the next output.

**Solution.** x1=B=(0,1)ᵀ and z1=B_new=(2,1)ᵀ. The next output is C x1=0 and C_new z1=2-2=0. At t=2, x2=(0.2,0.5)ᵀ, so y2=0.2. The transformed state is (1.2,0.5)ᵀ, again yielding 1.2-2(0.5)=0.2. The two state vectors have different coordinates but represent the same behavior.

Now set D=0.5. At time zero, x0=0 and u0=1 give y0=0.5 immediately. Updating the state first and calling that result y0 would violate the chosen timing convention.

## Connection to AI system models

Latent-state models, recurrent networks and system identification all require careful state and observation definitions. Matching one short output trajectory does not identify a unique internal realization or prove that the state is minimal. Later controllability and observability tests ask which internal directions can be actuated or distinguished. Coordinate-invariant input–output equivalence is a concrete starting point for those questions.

## Primary reference

MIT, [Dynamic Systems and Control, 6.241J](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/), graduate course and open textbook. The shear example and paired recurrences here are original teaching constructions.
