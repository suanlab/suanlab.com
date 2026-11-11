## 1. Allocate a fixed number of measurements

Suppose two parameters $a,b$ have the same unit. Sensor A reads $a$ with independent zero-mean noise of standard deviation $\sigma_a$, and sensor B reads $b$ with standard deviation $\sigma_b$. Choose the numbers $n_a,n_b$ of observations before looking at the data, with $n_a+n_b=N$ and at least one reading per parameter. Sample means estimate the parameters with variances $\sigma_a^2/n_a$ and $\sigma_b^2/n_b$. [Boyd and Vandenberghe, Convex Optimization, §7.5](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

## 2. Minimize total variance

The A-optimal criterion minimizes the trace of the estimator covariance:

$$
J_A=\frac{\sigma_a^2}{n_a}+\frac{\sigma_b^2}{n_b}.
$$

Allowing continuous allocations and differentiating with $n_b=N-n_a$ gives $n_a/n_b=\sigma_a/\sigma_b$. The noisier parameter receives more readings. For an integer budget, evaluate the feasible allocations adjacent to the continuous optimum or enumerate the budget when it is small.

## 3. Minimize the uncertainty ellipse area

The D-optimal criterion minimizes the determinant $J_D=\sigma_a^2\sigma_b^2/(n_a n_b)$. In this diagonal example, fixed total count makes $n_a n_b$ largest at a balanced allocation. The square root of the determinant is proportional to the area of an uncertainty ellipse at fixed Gaussian confidence level. Trace and determinant express distinct priorities.

## 4. Work through a 12-reading budget

Let $\sigma_a=1$, $\sigma_b=3$, and $N=12$. A-optimal allocation gives $(3,9)$ with trace $4/3$ and determinant $1/3$. D-optimal allocation gives $(6,6)$ with trace $5/3$ and determinant $1/4$. Changing the objective changes which plan is optimal. A task that values one parameter more can explicitly weight the corresponding variance.

## 5. Enumerate the feasible allocations

```python
from math import isclose

def criteria(total, first, sigma_a, sigma_b):
    va = sigma_a**2 / first
    vb = sigma_b**2 / (total - first)
    return va + vb, va * vb

plans = [(i, *criteria(12, i, 1, 3)) for i in range(1, 12)]
by_trace = min(plans, key=lambda plan: plan[1])
by_det = min(plans, key=lambda plan: plan[2])
assert by_trace[0] == 3 and isclose(by_trace[1], 4/3)
assert by_det[0] == 6 and isclose(by_det[2], .25)
print("A-optimal", by_trace)
print("D-optimal", by_det)
```

## 6. Check the design assumptions before collecting data

Estimate sensor noise in a pilot run, and record the acquisition cost of each reading. A correlated observation or an unequal per-reading cost changes the optimization problem. Choose a criterion tied to the downstream quantity of interest, then use held-out measurements to check the noise assumptions and realized precision.
