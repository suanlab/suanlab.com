## Why composition needs its own rule

A learning system rarely applies one operation. An input is rescaled, multiplied by weights, passed through an activation and compared with a target. To improve a parameter, we need the sensitivity of the final result to something several operations earlier.

The chain rule answers this question by combining **local sensitivities**. Think of a measuring instrument: the first stage converts a physical displacement into voltage; the second converts voltage into a display reading. The overall sensitivity is the product of their conversion rates, evaluated at the operating point. Rates may be negative, and nonlinear stages have rates that change with the current value.

After this unit, you should be able to identify an intermediate variable, evaluate each derivative at the right point, distinguish a derivative from a finite change and check a result numerically. The examples are synthetic scalar functions; no physical units are assumed in the lab.

## Notation and the operating point

| Symbol | Meaning in this unit |
| --- | --- |
| $x$ | Scalar input, the variable we perturb |
| $a,b$ | Fixed parameters while differentiating with respect to $x$ |
| $u=ax+b$ | Intermediate value |
| $y=u^2$ | Final scalar output |
| $\delta$ | A finite input displacement used to test the approximation |
| $\varepsilon>0$ | A separate small displacement used for central differences |

A derivative is evaluated **at a point**. For $a=2,b=1,x=1$, the intermediate value is $u=3$. The derivative of squaring must therefore be evaluated at 3, not at the original input 1. Keeping a table of forward values prevents this common mistake.

## Deriving the chain rule from small changes

Write $u=g(x)$ and $y=f(u)$. If $g$ is differentiable at $x$, a sufficiently small displacement produces

$$
\Delta u=g'(x)\delta+o(\delta).
$$

If $f$ is differentiable at $g(x)$, its first-order response is $\Delta y=f'(u)\Delta u+o(\Delta u)$. Combining these approximations gives

$$
\frac{dy}{dx}=f'(g(x))g'(x).
$$

Here $o(\delta)$ means a remainder whose ratio to $\delta$ tends to zero as $\delta\to0$. The rule concerns the limiting first-order response. It does not say that every finite displacement is exactly multiplied by the derivative.

The familiar notation $(dy/du)(du/dx)$ is helpful, but the proof is composition of linear approximations, not a license to cancel arbitrary derivative symbols like fractions. Differentiability of the component maps at the relevant points is a sufficient condition for this rule.

## A complete hand calculation

Keep $a=2,b=1$ fixed and use $x=1$.

| Stage | Forward value | Local derivative |
| --- | --- | --- |
| Input | $x=1$ | — |
| Affine stage | $u=2(1)+1=3$ | $du/dx=2$ |
| Square stage | $y=3^2=9$ | $dy/du=2u=6$ |

Multiply the local gains:

$$
\frac{dy}{dx}=6\times2=12.
$$

This means a sufficiently small positive change of $x$ produces approximately twelve times that change in $y$. It does not mean the output equals $12x$, nor that the slope remains twelve everywhere.

## Exact change versus tangent prediction

For this particular quadratic, expansion gives an exact finite-change identity:

$$
\Delta y=(a(x+\delta)+b)^2-(ax+b)^2
=2a(ax+b)\delta+a^2\delta^2.
$$

The first term is the tangent prediction; the second is the remainder. At the reference point with $\delta=0.1$:

$$
\Delta y=12(0.1)+4(0.1)^2=1.24.
$$

The new output is $10.24$, the old output is $9$, and their difference is $1.24$. Confusing the new value with the change gives the wrong interpretation even when the arithmetic is correct.

Reducing the displacement to $0.01$ makes the linear term $0.12$ and the remainder $0.0004$. The displacement shrinks tenfold, but the remainder shrinks one hundredfold. This quadratic scaling is special to the error order, not a claim that all functions have this exact remainder formula.

## Reading the interactive figures

The computation graph shows values flowing from left to right. Orange reverse labels trace how output sensitivity reaches earlier variables. Edge lengths do not encode value or gradient magnitude. The four stage buttons reveal the same full computation progressively; they do not alter the underlying function.

The chart plots horizontal input displacement against vertical **output change**, centered at the current input. Blue is the actual nonlinear change; the orange dashed line is its tangent. Both share coordinate axes. The axes automatically rescale and show positive display limits, so compare the numeric readouts before judging slopes between different settings.

Use the three guided experiments before freely changing $a,b,x$ and $\delta$. Central-difference $\varepsilon$ is a different control: changing it tests the derivative, while changing $\delta$ tests the quality of a finite-change prediction.

## Zero and negative local gains

At $x=-0.5$ with $a=2,b=1$, the intermediate value is zero. The derivative is therefore zero, but the function is not constant nearby:

$$
\Delta y=4\delta^2.
$$

For $\delta=0.1$, the change is still $0.04$. A stationary point removes the first-order term, not every higher-order effect.

At $x=-1$, the intermediate value is $-1$, so $dy/du=-2$ and $dy/dx=-4$. A positive input change can reduce the output. With $\delta=0.1$, the actual change is $-0.36$, versus the tangent prediction $-0.4$.

Finally, set $a=0$. Then $u=b$ for all inputs and $y=b^2$ really is constant with respect to $x$. Distinguish a zero gain at one operating point from a stage that blocks dependence everywhere.

## A numerical derivative check

Central differences estimate the derivative by two function evaluations:

$$
D_\varepsilon y(x)=\frac{y(x+\varepsilon)-y(x-\varepsilon)}{2\varepsilon}.
$$

For the lab’s quadratic, the quadratic terms cancel exactly in real arithmetic, so this expression equals the analytical derivative. Floating-point arithmetic can still introduce small differences. For a general sufficiently smooth function, central differences have truncation error of order $\varepsilon^2$; they are not universally exact.

Extremely small $\varepsilon$ subtracts nearly equal numbers and can amplify roundoff after division. A larger $\varepsilon$ can leave the local regime. Numerical checking supports a derivation but does not replace it, especially at nonsmooth points introduced in the backpropagation unit.

## From input derivatives to trainable parameters

The same graph answers different questions depending on the chosen variable. Holding $x,b$ fixed gives

$$
\frac{\partial y}{\partial a}=2u x,
\qquad \frac{\partial y}{\partial b}=2u.
$$

At the reference point these are 6 and 6, whereas $dy/dx=12$. State what changes and what is held fixed before writing a gradient. In a neural network, a parameter derivative measures how changing that parameter affects the loss for the fixed example. An input derivative can instead measure sensitivity to the example itself.

Multiple layers multiply local derivatives along a path. Repeated small factors can attenuate a signal; large factors can amplify it. With branches, contributions must also be added. The next unit develops that addition using Jacobians, rather than treating an entire network as one long scalar chain.

## Check your understanding

1. Set $x=0,a=3,b=-1$. Compute $u,y,dy/dx$, then the exact change for $\delta=0.1$. **Check:** $u=-1,y=1$, derivative $-6$, change $-0.51$.
2. At the reference point, reduce $\delta$ from $0.1$ to $0.05$. **Check:** the remainder becomes $0.01$, one quarter of the original.
3. Explain why the derivative with respect to $a$ differs from the derivative with respect to $x$, although both belong to the same forward calculation.
4. Sketch a case where the derivative is zero at a point but the function is not constant. Explain which higher-order term survives.

## Further study

The derivations and numeric examples above are worked examples for this book. For a broader treatment of partial derivatives, the chain rule and directional derivatives, see the [MIT OpenCourseWare Multivariable Calculus course](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/), especially Part B of Partial Derivatives. Continue to the Jacobian unit to replace scalar gains with local linear maps.
