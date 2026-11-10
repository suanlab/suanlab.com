## A steady equation asks a different question

A steady field has no time variable to initialize. Instead, boundary data help select a spatial configuration. We study minus the second spatial derivative of u equal to a constant source f on a rod of length L=1 meter. This simple setting is rich enough to show three outcomes: one solution, an entire family of solutions, or no solution satisfying the requested steady data.

Use temperature difference u in kelvins as the example field. The normalized source has units K/m squared. A physical steady heat model would relate that source to material properties and heat generation; those factors are not silently inferred here. The lesson studies the stated differential problem and its boundary traces, not a complete material experiment.

## Integrate the complete interior family

For a twice continuously differentiable function on the interval,

$$
-u''(x)=f\quad\Longrightarrow\quad
u(x)=-\frac{f}{2}x^2+ax+b.
$$

This is the entire C2 interior solution family, because integrating the constant second derivative twice introduces exactly two constants. Conversely, differentiating every member twice verifies the law. The coefficient a has units K/m and b has units K. These completeness and converse arguments allow an exhaustive boundary classification; a handful of plotted curves would not provide the same justification.

The interface draws one selected member. When the boundary problem is inconsistent, it still draws an explicitly labeled interior candidate so the failed right datum can be inspected. An interior candidate is not certified as a solution of the boundary-value problem merely because it satisfies the differential equation.

## Solve two value conditions

Dirichlet data prescribe u(0)=g0 and u(L)=g1. Keeping L explicit gives

$$
b=g_0,\qquad a=\frac{g_1-g_0}{L}+\frac{fL}{2}.
$$

Both coefficients are determined uniquely. In the numerical L=1 coordinates this becomes a=g1-g0+f/2. No choice of the otherwise visible mean or constant controls changes the solution in this mode; their labels restrict their meaning to the Neumann cases.

The first guide chooses f=2 and both endpoint values zero. It gives u=x-x squared, with a=1 and b=0. The endpoint coordinate derivatives are one and minus one K/m. The mean temperature difference is one sixth K. These derivative values arise from the solution; they were not additional independently imposed boundary data.

## Solve a mixed value and derivative condition

For mixed data, prescribe a left value u(0)=g0 and a right coordinate derivative u'(L)=q1. Since u'(x)=-f*x+a,

$$
b=g_0,\qquad a=q_1+fL.
$$

Again the coefficients are unique. The right input now has gradient units K/m, whereas the left input has temperature units K. The boundary selector changes the mathematical meaning and units of the data, not just the visual appearance of the endpoints.

For example, f=2, g0=0 and q1=0 produce u=-x squared+2x. Its right endpoint value is one K, not zero. Treating the right input as a value instead would describe a different problem and lead to the first guide's different solution.

## Derive the Neumann compatibility condition

Now prescribe coordinate derivatives u'(0)=q0 and u'(L)=q1. The left condition fixes a=q0. Integrating the differential equation across the interval requires

$$
q_1-q_0=-fL,\qquad q_1-q_0+fL=0.
$$

This is an existence condition. If it fails, changing b cannot help because derivatives do not depend on b. Adding a mean constraint cannot help either: it adjusts the vertical offset while leaving the derivative difference unchanged.

The second guide sets f=2 and q0=q1=0. Their derivative difference is zero, while the equation demands minus two. There is no C2 steady solution satisfying those data. The drawn orange candidate uses the correct interior equation and left derivative, but its right derivative is minus two rather than the requested zero. The residual readout reports two in the displayed numerical compatibility expression.

An incompatible steady balance does not mean a time-dependent heat problem with related data has no solution. With a sustained source and insulated ends, a transient temperature may keep changing rather than settling to a steady field. Steady and transient existence are different questions.

## Retain the whole additive-constant family

When compatibility holds, the two derivative conditions fix a but leave b unrestricted:

$$
u(x)=-\frac{f}{2}x^2+q_0x+C,\qquad C\in\mathbb R.
$$

Every real C gives the same derivative data. The constant control selects one member for the plot; the equation above retains the complete affine family. A bounded slider is not the mathematical boundary of that family.

This ambiguity is a simple nullspace: differentiation annihilates constants. An optimization algorithm might happen to choose C=0, but that is a selection rule rather than uniqueness supplied by the derivative observations. In a general connected domain, related Neumann problems also require attention to constants and compatibility; our complete proof here concerns the stated one-dimensional constant-source case.

## Use a mean constraint to choose a representative

Define the spatial mean with its length factor:

$$
\mu=\frac{1}{L}\int_0^L u(x)\,dx
=-\frac{fL^2}{6}+\frac{aL}{2}+b.
$$

For compatible derivative data, prescribing this mean determines

$$
b=\mu+\frac{fL^2}{6}-\frac{q_0L}{2}.
$$

The third guide uses f=2, q0=1, q1=-1 and mean mu=2. Compatibility holds, and b=2+1/3-1/2=11/6 K. The selected function is -x squared+x+11/6. Removing the mean constraint restores the arbitrary constant family; it does not change the derivative conditions.

The interface distinguishes the ungauged constant freedom from uniqueness after the extra constraint. It represents supported coefficients with integer numerators over six and means with integer numerators over twelve. Exact compatibility uses integers. Curves and decimal values are floating-point displays; their apparent agreement is not the reason existence or uniqueness is reported.

## Separate coordinate derivatives, normal derivatives and flux

Both derivative inputs are measured in the increasing-x direction. The outward normal points left at x=0 and right at x=L. Consequently the outward derivatives are minus q0 on the left and q1 on the right. In outward-normal form, their sum must equal minus fL.

A conductive heat-flux vector would add another constitutive minus sign and a conductivity factor. Gradient, outward derivative and outward heat flux must therefore be named with their units and orientation. The secondary figure draws the coordinate gradient and green prescribed derivative targets; it does not label those targets as watts or automatically reinterpret them as outward-normal data.

## Connect the constant ambiguity to learned inverse problems

A model trained only on derivative measurements cannot infer an additive temperature reference from those data alone. Many fields produce the same gradients. A mean, point value or other reference can remove that freedom, but it is additional information or a declared convention.

This example also separates an identifiability issue from inconsistent observations. Compatible derivative data with no reference give many solutions. Incompatible derivative data give no steady solution in the stated class. A learning loss can produce a compromise in either case, but a low objective value should not be interpreted as an existence or uniqueness theorem. Inspect the governing balance and the actual data constraints.

## Exercises with worked solutions

1. **Exercise:** Solve -u''=2 with u(0)=u(1)=0. **Solution:** Integration gives -x squared+ax+b; the left value sets b=0 and the right sets a=1.
2. **Exercise:** Replace the right value by u'(1)=0. **Solution:** Then a=2 and b=0, giving -x squared+2x. The problem changed.
3. **Exercise:** Can f=2 and q0=q1=0 be repaired with C=100? **Solution:** No. The derivative difference remains wrong; a constant has zero derivative.
4. **Exercise:** For f=2,q0=1, what right derivative is compatible? **Solution:** q1=q0-f=-1 in the L=1 numerical coordinates.
5. **Exercise:** Set the compatible solution's mean to two. **Solution:** The mean identity gives b=2+2/6-1/2=11/6.
6. **Exercise:** Does a mean-selected solution make the original derivative-only problem unique? **Solution:** No. It makes the augmented problem unique while the original problem retains its constant family.

## Sources and further study

The [MIT Poisson-equation notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/4336b9d0607ac33f430908fdf4cb78ba_MIT18_152F11_lec_06.pdf) discuss boundary conditions and uniqueness up to a constant for Neumann data. Their displayed Poisson sign convention differs from the minus-second-derivative convention used here. The explicit interval formulas, compatibility test and mean-selected family above are direct integrations for this lesson.
