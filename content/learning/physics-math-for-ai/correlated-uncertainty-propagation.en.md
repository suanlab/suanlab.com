## Specify the input law and the measurement function

Uncertainty propagation asks how uncertainty in input quantities affects a defined output quantity. The answer depends on the measurement function and on the joint law of the inputs, including their dependence. Knowing two marginal standard deviations is generally insufficient.

This lesson uses jointly Gaussian signed length coordinates X and Z, measured in centimeters. Their means are mu_X and mu_Z, their standard uncertainties are u_X and u_Z, and their covariance is c. Negative values are allowed by the declared model; the product is an algebraic signed quantity with area units, not automatically the positive area of a physical rectangle. The available measurement functions are a sum, a difference, and a product.

## Construct a covariance model that remains valid at its boundaries

Let U and V be independent standard Gaussian variables. A useful construction is

$$
X=\mu_X+u_XU,\qquad
Z=\mu_Z+u_Z\bigl(\rho U+\sqrt{1-\rho^2}V\bigr),
\qquad -1\le\rho\le1.
$$

The covariance is c equal to rho times u_X times u_Z. This construction yields a positive semidefinite covariance matrix, including singular cases. No division by a standard uncertainty is required to generate the geometric reference curve.

When both standard uncertainties are positive, rho is their Pearson correlation. When either standard uncertainty is zero, Pearson correlation is undefined. The latent coefficient control still specifies a construction, but the interface displays the observable correlation as unavailable instead of reporting a made-up zero. If both uncertainties vanish the covariance rank is zero; a single uncertain direction gives rank one; a nonsingular covariance has rank two.

## Derive the sum and difference formulas

Center the inputs and expand the square of their sum or difference. The cross term changes sign:

$$
\operatorname{Var}(X+Z)=u_X^2+u_Z^2+2c,\qquad
\operatorname{Var}(X-Z)=u_X^2+u_Z^2-2c.
$$

These identities require finite second moments; they are exact for the linear functions and do not need a Taylor approximation. The Gaussian assumption specifies the rest of the laboratory's input law, rather than being necessary for these two variance identities alone.

With means two and three, standard uncertainties one and one, and rho equal to one, the sum has mean five and variance four. The difference is always minus one and has variance zero. Each input can remain uncertain while their perfectly shared component cancels in the difference. Exact cancellation relies on exact equal amplitudes and perfect shared dependence; a nearly common error need not cancel completely.

## Locate the first-order approximation for a product

For the product function f(X,Z)=XZ, the gradient evaluated at the input means has components mu_Z and mu_X. The first-order variance is

$$
u_{\mathrm{lin}}^2=
\mu_X^2u_Z^2+\mu_Z^2u_X^2+2\mu_X\mu_Zc.
$$

This describes propagation through the linearized measurement function. It becomes exact for a truly linear function but need not be exact for a product. At zero means both sensitivities are zero, so the first-order result is zero even when both inputs have substantial uncertainty. A zero local gradient does not prove that a nonlinear output is constant.

## Derive the product mean correction

Write centered inputs as xi and zeta. Expanding the product gives

$$
XZ=\mu_X\mu_Z+\mu_X\zeta+\mu_Z\xi+\xi\zeta,
\qquad E[XZ]=\mu_X\mu_Z+c.
$$

Thus evaluating the function only at the mean inputs can miss the output mean as well as its variance. Positive covariance contributes a positive correction in this product example; negative covariance contributes a negative one. This correction follows from the definition of covariance and finite second moments, rather than a Gaussian-specific theorem.

The laboratory reports the exact-model output mean alongside the first-order mean. Changing rho can move the output mean while leaving both input means fixed. This is another reason that marginal input summaries alone do not determine the output behavior.

## Derive the omitted variance under joint Gaussianity

For jointly centered Gaussian inputs, third centered moments vanish. The linear part of the expansion is therefore uncorrelated with the centered product term. Expanding the latent construction using independent standard normals gives

$$
E[\xi^2\zeta^2]=u_X^2u_Z^2+2c^2.
$$

This follows using the normal moments E[U squared]=1, E[U to the fourth]=3, and zero odd moments. Subtracting the squared product mean c squared gives the variance of xi times zeta. Adding the linear contribution yields

$$
\operatorname{Var}(XZ)=u_{\mathrm{lin}}^2+u_X^2u_Z^2+c^2.
$$

The final two terms are the higher-order variance omitted by the first-order approximation. The Gaussian assumption matters here: other joint laws can have nonzero third moments and different fourth moments. A covariance matrix alone does not justify this exact nonlinear variance formula for arbitrary distributions.

## Work the zero-mean counterexample

Set both means to zero, both standard uncertainties to one centimeter, and rho to zero. The first-order variance is zero, but the exact-model product variance is one to the fourth-power centimeter. The output standard uncertainty is one square centimeter. Multiplying uncertain inputs has not eliminated their variability; the local linearization has discarded the relevant term.

Now change rho to one. The product mean becomes one square centimeter, and its variance becomes two to the fourth-power centimeter. The first-order variance remains zero. Both the mean correction and higher-order contribution are visible while the input means remain unchanged.

A Gaussian input pair generally has a non-Gaussian product. A standard uncertainty is a standard deviation under the stated law, not automatic permission to report a normal 95 percent interval for the output. Interval coverage would require an additional distributional calculation or another justified method.

## Read the contour without attaching a false coverage percentage

The second diagram maps a unit circle in the latent U,V coordinate plane into the X,Z plane. It is a geometric reference for dependence and relative input scales. At singular covariance the image becomes a segment or point. The axes retain centimeter units and an equal geometric scale.

This plotted unit-circle image is not labeled a 68 percent or 95 percent probability region. In two dimensions a radius-one Gaussian contour has a different coverage interpretation from a one-dimensional plus-or-minus-one-standard-deviation interval, and singular cases change that interpretation again. The laboratory deliberately makes no coverage claim for that outline.

The variance graph separately varies the latent coefficient from minus one to one. Blue shows the exact-model variance and orange the first-order result. For a sum or difference the curves coincide. For a product their separation displays the omitted term. This graph is not a sample cloud or a Monte Carlo experiment.

## Track units and exact supported comparisons

Sum and difference outputs use centimeters, so their variances use square centimeters. A product output uses square centimeters and its variance uses centimeters to the fourth power. Covariance between the two input coordinates uses square centimeters. Correlation, when defined, has no units.

The controls use integer means, uncertainty scales in tenths, and a latent coefficient in hundredths. Supported moment formulas can be represented with exact integer numerators. Variance numerators share denominator one hundred million, and mean numerators share denominator ten thousand. Those comparisons establish nonnegative variance and singularity without rounding a small negative value away. Displayed divisions, square roots, and curve coordinates are numerical approximations.

## Exercises with worked solutions

1. **Exercise:** Compute the sum variance for two unit uncertainties and covariance one. **Solution:** One plus one plus two equals four. Omitting covariance would incorrectly give two.
2. **Exercise:** Compute the corresponding difference variance. **Solution:** One plus one minus two equals zero. The centered input errors are identical and cancel exactly.
3. **Exercise:** Why can zero sensitivities miss product uncertainty? **Solution:** At zero means the linear part vanishes, while the quadratic fluctuation xi times zeta remains random.
4. **Exercise:** What is the mean of a product with zero input means and covariance minus one? **Solution:** Minus one in square-centimeter units. The output mean is the covariance correction, not the product of means.
5. **Exercise:** If u_X is zero, should Pearson correlation be shown as zero? **Solution:** No. Its definition divides by the product of standard deviations and is undefined. Covariance is zero, which is a different statement.
6. **Exercise:** Can the Gaussian product variance formula be used for any pair with the same covariance? **Solution:** No. Its nonlinear part uses Gaussian third and fourth moments; other joint laws require their own analysis.

## Source and AI research connection

The [JCGM discussion of combined uncertainty and correlated inputs](https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/05_e.html) explains the role of a measurement function and first-order propagation. The latent Gaussian construction, exact product moments, and numerical counterexamples above are directly derived for this lesson. For AI research, automatic differentiation can supply a Jacobian for first-order propagation, but derivatives alone do not establish an input law, identify measurement biases, or justify ignoring nonlinear uncertainty near a stationary point.
