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

## Research extension: Project incompatible source data and track the missing mean drift

### Generalize the balance before choosing an offset

Keep coordinate derivative data q_0 and q_1, but allow any continuous source f on a physical interval [0,L], with L positive. Direct integration of minus u second derivative equal to f gives a candidate with free constant C. Its right derivative matches exactly when the integrated defect c vanishes:

$$
u(x)=C+q_0x-\int_0^x(x-s)f(s)\,ds,\qquad
c=q_1-q_0+\int_0^Lf(s)\,ds.
$$

This proves both necessity and sufficiency for a C2 solution with these derivative data: c=0 selects the correct right derivative, and then every C is allowed. Prescribing spatial mean mu chooses C by integrating once more:

$$
C=\mu-{q_0L\over2}+{1\over2L}\int_0^L(L-s)^2f(s)\,ds\quad(c=0).
$$

The constant-source formula from the laboratory is the special case with that source independent of x. A mean fixes the nullspace after compatibility; it never changes c. Here f has units K/m², q has units K/m, and c has units K/m. The integral for C has temperature units.

### Define what a least-change repair actually changes

Suppose the derivative data must stay fixed and you deliberately allow a source correction delta f. Among square-integrable corrections satisfying the required integral, Cauchy–Schwarz gives the minimum L2 change. Equality holds only for a constant correction almost everywhere:

$$
\int_0^L\delta f=-c,\qquad
\|\delta f\|_2\ge {|c|\over\sqrt L},\qquad
\delta f_*=-{c\over L},\qquad
f_*=f-{c\over L}.
$$

This is the orthogonal projection of the source onto the affine compatibility set with fixed boundary gradients. It revises the source data. It is not a gauge choice and does not solve the original incompatible problem. Other choices of which data may change, or other error metrics, define different repairs and require separate justification.

For example, use L=2 m, f(x)=2 K/m²+(1 K/m³)x, q_0=1 K/m, and q_1=−2 K/m. The source integral is 6 K/m, hence c=3 K/m. The minimum correction is −1.5 K/m², giving f_*(x)=0.5+x in the same coordinates. With mean mu=3 K, the corrected solution has C=8/3 K. The code checks its endpoint derivatives and mean by exact polynomial integration; a source correction must be reported with that solution.

### Locate the discrepancy in a transient model

The original incompatible data can still define a time-dependent heat model with a mean that keeps changing. For constant diffusivity kappa and the same fixed coordinate derivative data, integration of the transient law gives

$$
u_t=\kappa(u_{xx}+f),\qquad
{d\bar u\over dt}={\kappa\over L}
\left(q_1-q_0+\int_0^Lf\right)={\kappa c\over L}.
$$

Thus c nonzero forbids a steady mean rather than every transient field. For constant f=2, zero endpoint derivatives, and a uniform initial field mu_0, the exact solution is u=mu_0+2 kappa t. Projecting that source to zero would remove this uniform rise. It would hide a physical mean evolution if presented as a harmless offset convention. This transient identity is conditional on a sufficiently regular solution; it is not an existence proof for every prescribed initial profile.

### Why a residual bound also needs a gauge

If a candidate and a reference have the same boundary derivatives and mean, their difference e has zero derivative at both ends and zero mean. For minus e second derivative equal to r, integration by parts and the mean-zero Neumann Poincaré inequality give

$$
\|e'\|_2^2=\langle e,r\rangle,\qquad
\|e\|_2\le {L\over\pi}\|e'\|_2,\qquad
\|e\|_2\le {L^2\over\pi^2}\|r\|_2.
$$

For smooth mean-zero functions, expanding in cosine modes explains the Poincaré constant: the zero mode is absent and the first remaining frequency is pi/L. The residual must have zero integral under the matched derivative conditions. Without the same mean, an arbitrary constant e has r=0 but unbounded field error. This distinguishes a compatibility condition from a nullspace convention and from a stability estimate. The [MIT Poisson notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/4336b9d0607ac33f430908fdf4cb78ba_MIT18_152F11_lec_06.pdf) provide boundary-problem background; their Poisson sign convention differs from ours.

| Intervention or premise | What it changes | What it cannot claim |
|---|---|---|
| Set the mean after c=0 | Selects an additive representative | Repair a nonzero compatibility defect |
| Replace f by f−c/L | Revises the source to the compatibility set | Solve the unchanged original source problem |
| Match mean and derivative traces | Removes the error's constant mode | Make sampled residuals a continuum norm bound |

### Worked research question

**Question:** Can a training pipeline silently subtract a source mean to make a Neumann solve succeed? **Answer:** It can define a modified problem that way, but must retain the original data, correction, and physical interpretation. A differentiable solver's numerical success does not validate the revised source. The code demonstrates the unique minimum source correction within a family of feasible perturbations, checks the exact corrected field, and verifies the inverse bound on a cosine eigenmode. These general-source calculations are standalone examples; the browser keeps its original constant-source controls.

```python
from fractions import Fraction as F
from math import pi, sqrt, isclose
L=F(2);q0=F(1);q1=F(-2);mu=F(3)
a,b=F(2),F(1)  # f(x)=a+b*x, in the declared physical coordinates.
integral=a*L+b*L**2/2
defect=q1-q0+integral
delta=-defect/L
ac=a+delta
assert defect==3 and ac==F(1,2)
assert q1-q0+ac*L+b*L**2/2==0
C=mu-q0*L/2+ac*L**2/6+b*L**3/24
assert C==F(8,3)
derivative_right=q0-ac*L-b*L**2/2
computed_mean=C+q0*L/2-ac*L**2/6-b*L**3/24
assert derivative_right==q1 and computed_mean==mu
minimum_norm2=delta**2*L
for beta in [F(-2),F(-1),F(0),F(1),F(2)]:
    # Feasible correction delta+beta*(x-L/2); its extra term integrates to zero.
    norm2=delta**2*L+beta**2*L**3/12
    assert norm2>=minimum_norm2 and (norm2==minimum_norm2)==(beta==0)
assert minimum_norm2==defect**2/L
print('defect',defect,'constant source correction',delta,'corrected C',C,'mean',computed_mean)
kappa=F(1,50)
print('uniform transient mean slope for original f=2, zero gradients',2*kappa)
epsilon=.1
residual_norm=epsilon*sqrt(float(L)/2)
field_norm=epsilon*float(L)**2/(pi*pi)*sqrt(float(L)/2)
assert isclose(field_norm/residual_norm,float(L)**2/(pi*pi),rel_tol=1e-14)
print('mean-zero cosine inverse norm ratio',field_norm/residual_norm)
```

## Exercises with worked solutions

1. **Exercise:** Solve -u''=2 with u(0)=u(1)=0. **Solution:** Integration gives -x squared+ax+b; the left value sets b=0 and the right sets a=1.
2. **Exercise:** Replace the right value by u'(1)=0. **Solution:** Then a=2 and b=0, giving -x squared+2x. The problem changed.
3. **Exercise:** Can f=2 and q0=q1=0 be repaired with C=100? **Solution:** No. The derivative difference remains wrong; a constant has zero derivative.
4. **Exercise:** For f=2,q0=1, what right derivative is compatible? **Solution:** q1=q0-f=-1 in the L=1 numerical coordinates.
5. **Exercise:** Set the compatible solution's mean to two. **Solution:** The mean identity gives b=2+2/6-1/2=11/6.
6. **Exercise:** Does a mean-selected solution make the original derivative-only problem unique? **Solution:** No. It makes the augmented problem unique while the original problem retains its constant family.

## Sources and further study

The [MIT Poisson-equation notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/4336b9d0607ac33f430908fdf4cb78ba_MIT18_152F11_lec_06.pdf) discuss boundary conditions and uniqueness up to a constant for Neumann data. Their displayed Poisson sign convention differs from the minus-second-derivative convention used here. The explicit interval formulas, compatibility test and mean-selected family above are direct integrations for this lesson.
