## A linear system is an operation, not an inverse symbol

The lab first solves A x=(a,1) with A=[[2,1],[1,2]] using partial-pivot Gaussian elimination and back substitution. It does not construct an explicit inverse. The exact answer is ((2a-1)/3,(2-a)/3), which supplies an independent hand check. Although this small matrix is well behaved, checking pivots remains part of a general elimination algorithm.

For nonlinear equations the unknown does not appear through a fixed matrix. Consider F(z)=z²-a with a positive. There are two real roots; the experiment explicitly targets the positive branch. Domain and branch choices are part of the problem specification, not incidental display settings.

## Newton solves a local linear problem

At the current z, replace F by its tangent and solve for a correction. For the square-root example this gives

$$
F'(z)\Delta=-F(z),\qquad
z_{\mathrm{new}}=z+\Delta=\frac12\left(z+\frac az\right).
$$

Positive starting values keep the displayed iteration defined. Near a simple root, sufficiently accurate Newton steps can converge quadratically. That local statement does not guarantee convergence for an arbitrary function, singular derivative or starting point. The controlled positive-root example should not be generalized into such a promise.

For a=2 and z0=1 the first estimates are 1.5, 1.4166667 and 1.4142157. By the fourth update the error is tiny, but the iteration itself has not produced a bracket certificate.

## Bisection trades speed for an explicit enclosure

Initialize lo=0 and hi=max(1,a,start). The positive root lies in this interval throughout the allowed control range. Iteration zero displays the initial midpoint. Each sign test discards half the interval, and the next midpoint is reported:

$$
|z_{\mathrm{mid}}-\sqrt a|\le(hi-lo)/2.
$$

At a=2, initial upper endpoint two, four updates leave [1.375,1.5]. The midpoint 1.4375 has certified error at most 0.0625. Its actual error is smaller; a bound should not be presented as an equality. Changing the start control also changes the initial bracket reference, so comparisons should record that choice.

## Residual and root error are related but distinct

For any positive estimate, z²-a=(z-sqrt(a))(z+sqrt(a)). Thus the same residual can imply different root errors at different scales. In a multidimensional problem, the inverse Jacobian plays the corresponding conditioning role. Stopping on residual alone requires a reason that residual meaningfully controls the desired error.

## A global proof for this positive square-root iteration

For r=√a>0 and x>0, subtract r from the Newton update and use a=r²:

$$
x^+-r=\frac{(x-r)^2}{2x}\ge0.
$$

After one update every positive start is above the root. Above the root, x⁺≤x and x⁺≥r. The decreasing bounded sequence has a positive limit, and the update equation forces that limit to equal r. Moreover x≥r gives |x⁺−r|≤|x−r|²/(2r), an explicit quadratic-error bound. This global scalar argument is much stronger than the local Newton claim, but relies on this particular function and the positive branch.

An upper estimate x≥r also gives a lower estimate a/x≤r. Exact rational arithmetic can preserve the enclosure [a/x,x] without computing r:

| Start x₀, a=2 | First upper | First lower |
|---|---:|---:|
| 1/4 | 33/8 | 16/33 |
| 1 | 3/2 | 4/3 |
| 4 | 9/4 | 8/9 |

```python
from fractions import Fraction as F
for start in (F(1, 4), F(1), F(4)):
    a, x = F(2), start
    for k in range(4):
        x = (x+a/x)/2
        lower = a/x
        assert lower*lower <= a <= x*x
        if k == 0: print(start, x, lower)
```

The inequalities in the program are exact. In binary64, a rigorous enclosure would require directed rounding or a separately justified rounding margin. The lab's ordinary Newton mode therefore continues to report no bracket certificate. Its bisection enclosure has its own stated model. For linear elimination, pivoting controls small divisors; it does not remove data conditioning or make every finite computed residual an exact certificate.

## Exercise: compare stopping evidence

For a=1 and start one, choose zero iterations and Newton. The initial estimate is already exact. Now choose bisection: its iteration-zero midpoint is one half, because the initial bracket is [0,1]. Explain why both are legitimate initial states. For a=2 reproduce the four bisection intervals by hand and verify that every interval contains sqrt(2).

Further reading: [LAPACK accuracy and error measures](https://netlib.org/lapack/lug/node75.html). The elimination and positive-root experiments are constructed directly from their defining equations.
