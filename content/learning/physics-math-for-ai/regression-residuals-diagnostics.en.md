## Geometric fitting versus statistical inference

Least squares projects an observed response vector onto the design column space. The normal equations express orthogonality of residuals to design columns. This algebra does not itself require Gaussian noise. Statistical interpretations of coefficient intervals or variance estimates do require additional assumptions such as centered, homoscedastic and independent errors.

$$
\widehat y=X\widehat\beta,\qquad r=y-\widehat y,\quad X^\top r=0.
$$

## Work through three observations

The defaults are x=(−1,0,1), y=(0,1,3). With an intercept, the fitted intercept is 4/3 and slope 3/2. Fitted values are (−1/6,4/3,17/6), leaving residuals (1/6,−1/3,1/6). Their sum and inner product with x are zero. The plot uses observation index horizontally; actual x values appear in the table, allowing repeated-x designs to be compared.

$$
\operatorname{SSE}=\sum_i r_i^2=\frac16,\qquad\widehat\sigma^2=\frac{\operatorname{SSE}}{n-\operatorname{rank}(X)}.
$$

## Intercepts, degrees of freedom and leverage

Without an intercept the slope remains 3/2, but residuals sum to four. Zero residual sum depends on including a constant design column; it is not universal. Residual degrees of freedom use rank rather than merely the number of columns. Leverages are diagonal entries of the projection matrix and sum to design rank. High leverage alone does not establish an outlier or a causal effect.

$$
H=XX^+,\qquad h_{ii}=H_{ii},\qquad\operatorname{tr}(H)=\operatorname{rank}(X).
$$

## One fit can have multiple coefficients

When every x equals one, intercept and slope columns coincide. The column space has rank one and fitted values equal the response mean, but the two coefficients are not separately identified. The lab explicitly selects the minimum-norm solution, splitting the mean equally between coefficients. This does not establish two uniquely data-determined effects. Small training SSE likewise does not certify test prediction error or causal effects.

$$
x_i=1\implies\widehat\beta_0+\widehat\beta_1=\bar y,\qquad\widehat\beta_{\min\text{-norm}}=(\bar y/2,\bar y/2).
$$

## Exercises with explanations

1. What is the default residual degrees of freedom? **3−2=1.**
2. What about identical x? **3−1=2.**
3. Do residuals establish causal effects? **No: design and modeling assumptions matter.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
