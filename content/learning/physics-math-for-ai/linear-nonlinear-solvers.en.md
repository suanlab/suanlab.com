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

## Exercise: compare stopping evidence

For a=1 and start one, choose zero iterations and Newton. The initial estimate is already exact. Now choose bisection: its iteration-zero midpoint is one half, because the initial bracket is [0,1]. Explain why both are legitimate initial states. For a=2 reproduce the four bisection intervals by hand and verify that every interval contains sqrt(2).

Further reading: [LAPACK accuracy and error measures](https://netlib.org/lapack/lug/node75.html). The elimination and positive-root experiments are constructed directly from their defining equations.
