## What is fixed before waiting?

Pointwise convergence fixes x before choosing how long to wait. Uniform convergence chooses one waiting time for every x:

$$
\text{pointwise: }\forall x\ \forall\varepsilon>0\ \exists N\ \forall n\ge N:\ |f_n(x)-f(x)|<\varepsilon,
$$

$$
\text{uniform: }\forall\varepsilon>0\ \exists N\ \forall x\ \forall n\ge N:\ |f_n(x)-f(x)|<\varepsilon.
$$

The issue is dependence of N on x. Successful approximations at finitely many probes do not erase this quantifier distinction.

## Pointwise powers on a closed interval

On [0,1), xⁿ tends to zero; at one it stays one:

$$
f(x)=\begin{cases}0&0\le x<1,\\1&x=1.\end{cases}
$$

At n=8,x=0.5 the error is only 1/256, yet points closer to one can have large error. For any finite n, x<1 can make xⁿ arbitrarily close to one. Thus the error supremum is one, but it is never attained: interior errors are below one and endpoint error is zero.

## Prove failure rather than sample it

Fix ε=1/2. For any proposed N, choose n=N and x=(3/4)^(1/n). The error is 3/4, refuting a common waiting time. A finite sample maximum below one does not contradict this supremum argument.

Restricting the domain changes the result. On [0,r] with r<1, the supremum rⁿ tends to zero, so convergence is uniform. State the domain as well as the function sequence.

## A uniform comparison

For fₙ(x)=x/n on [0,1],

$$
\sup_{x\in[0,1]}|x/n|=1/n\longrightarrow0.
$$

Choose N>1/ε to control every x and all n≥N. Here the supremum is attained at one, unlike the powers example.

## Preserved continuity and limits of the theorem

A uniform limit of continuous functions is continuous. Split |f(y)−f(x)| into two uniform approximation errors and the middle continuous-function error. The discontinuous limit of xⁿ offers another proof that convergence is not uniform.

Do not treat uniform convergence as permission to interchange every operation. In particular, uniform convergence of functions alone does not guarantee convergence of derivatives.

## AI connections and exercises

Accuracy on a finite evaluation set differs from uniform worst-case approximation. Check whether a research result controls pointwise, L² or supremum error.

1. On [0,0.9], xⁿ **does converge uniformly**, bounded by 0.9ⁿ.
2. On [0,1], one is the **supremum, not an attained maximum**, of the error.
3. Pointwise waiting time **may depend on x**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
