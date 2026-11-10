## 1. Eliminate a variable by summing its weights

Coarse graining starts by selecting degrees of freedom to retain. The marginal weight of a retained configuration is the sum of the weights of all compatible microscopic configurations. The negative logarithm of that weight defines an effective action, with the original partition function recovered by summing over the retained configurations. An additive constant matters for that partition function and cancels in normalized probabilities.

## 2. Decimate alternate spins on a ring

Take a zero-field ferromagnetic Ising ring of even length with dimensionless bond coupling $K$ and weights $e^{K\sum_i s_i s_{i+1}}$. Retain the even sites. Between two retained spins $s,t$, sum the eliminated spin $u$:

$$
\sum_{u=\pm1} e^{K(s+t)u} = A e^{K' st},\qquad
K'=\tfrac12\log\cosh(2K),\qquad A=2\sqrt{\cosh(2K)}.
$$

The two retained configurations with $st=1$ have local weight $2\cosh(2K)$. Those with $st=-1$ have weight $2$. Taking their ratio determines $K'$ and their geometric mean determines $A$. Each eliminated spin supplies one factor $A$. This identity retains the correct marginal even on a finite ring.

For an $N$-site ring, the transfer matrix has eigenvalues $2\cosh K$ and $2\sinh K$, so
$$
Z_N(K)=(2\cosh K)^N[1+(\tanh K)^N].
$$
After one step the recovered normalization is $A^{N/2}Z_{N/2}(K')$. In the experiment the recovered $\log Z$ is a finite-ring quantity, while $a\xi$ comes from the corresponding infinite chain at lattice spacing $a$.

## 3. Track a length and a fixed point

The identity $\tanh K'=(\tanh K)^2$ shows how a correlation measured two sites apart becomes a correlation measured one step apart on the coarse lattice. The infinite-chain correlation length in units of lattice spacing is $\xi=-1/\log(\tanh K)$ for $0<K<\infty$. One step halves this number and doubles the lattice spacing. Their product stays fixed.

For a chain with any finite positive $K$, iteration takes $K$ toward zero. Perfect order at $K=\infty$ is a separate fixed point. Writing $q=e^{-2K}$ gives $q'=2q/(1+q^2)$, so a small density parameter $q$ grows with linear multiplier 2 near perfect order. The change of length unit is also a factor 2. For a differentiable multivariate map, an eigenvalue of magnitude greater than one grows under repeated linearization, one below unity decays, and one of unit magnitude calls for higher-order analysis. [Tong, Statistical Field Theory §3.1](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S3)

## 4. Check whether the chosen coupling family closes

A central spin coupled to four retained neighbors gives local log weight $f(K(s_1+s_2+s_3+s_4))$, where $f(x)=\log(2\cosh x)$. Products of the four signs form an orthogonal basis under the uniform average on the sign cube. The coefficient of $s_1s_2s_3s_4$ is

$$
c_4=\frac{f(4K)-4f(2K)+3f(0)}8.
$$

At $K=0.4$ it is approximately $-0.027026$. This term appears even though every original bond is pairwise. Dropping it projects the effective interaction onto a smaller family and changes the marginal. The size of a discarded coefficient and the change in a target observable are both useful checks.

## 5. Compare exact and truncated weights

```python
from itertools import product
from math import cosh, exp, log, prod, sqrt, tanh

def ring(length, coupling):
    return sum(exp(coupling * sum(s[i] * s[(i+1) % length] for i in range(length)))
               for s in product((-1, 1), repeat=length))

K = 0.4
K_new = 0.5 * log(cosh(2*K))
A = 2 * sqrt(cosh(2*K))
assert abs(ring(8, K) - A**4 * ring(4, K_new)) < 1e-10
assert abs(tanh(K_new) - tanh(K)**2) < 1e-14

def log_weight(x):
    return log(2*cosh(x))

states = list(product((-1, 1), repeat=4))
c4 = sum(prod(s) * log_weight(K*sum(s)) for s in states) / 16
closed_form = (log_weight(4*K) - 4*log_weight(2*K) + 3*log_weight(0))/8
assert abs(c4 - closed_form) < 1e-14
exact = [exp(log_weight(K*sum(s))) for s in states]
truncated = [exp(log_weight(K*sum(s)) - c4*prod(s)) for s in states]
z, z_truncated = sum(exact), sum(truncated)
tv = 0.5 * sum(abs(p/z - q/z_truncated) for p, q in zip(exact, truncated))
assert tv > 0
print(K_new, c4, tv)
```

## 6. Blocking, rescaling and projection

Decimation keeps a subset of the original spins. A block-spin map instead assigns a new variable, such as the sign of the total magnetization, to an entire block. Specify the rule at ties before forming the marginal. For a field in Fourier coordinates one may keep low momenta and integrate out a shell of high momenta. Both choices need a length rescaling before successive effective models can be compared on the same grid. Field normalization may also be rescaled to keep a chosen coefficient fixed.

An exact flow acts on every coupling allowed by the model and its symmetries. In a numerical flow the four-spin example shows how truncation creates a measurable error. Check which observables or probability differences the chosen projection controls before iterating it. [Tong, Statistical Field Theory §§3.1–3.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S3)

## 7. What to inspect

- At weak coupling, expand $\log\cosh(2K)$ and predict how quickly $K$ shrinks after one step.
- Compare aligned and opposed retained spins across one bond and check how $A$ recovers the local weights.
- In the four-neighbor example remove the four-spin coefficient and compare the local log weight on all sixteen configurations.
- In a statistical or machine-learning model, ask which retained observables should survive marginalization before choosing an approximation to the effective interaction.
