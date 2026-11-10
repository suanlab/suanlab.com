## Different potentials can describe the same fields

Adding the same constant to every height on a map changes the labels without changing the slope. Gauge freedom extends this intuition to time-dependent scalar and vector potentials. The extension needs care: electric field depends on both a spatial gradient and a time derivative. A large change in the potential picture therefore need not imply a large change in force. Our task is to inspect the two contributions separately, then compare the observable electric and magnetic fields. We will also construct changes that really do alter those fields.

## Prerequisites and the scope of this model

Review gradients, curls, divergence, commuting partial derivatives, Faraday law, and charge conservation. We work in a smooth local region with constant positive permittivity and permeability. The polynomial fields below are teaching families, not finite-energy solutions throughout infinite space. An actual device needs specified remote sources, boundaries, and initial data. Questions about multiply connected regions, singular potentials, or quantum phases require additional treatment. Keeping this scope explicit prevents a useful local calculation from becoming an unsupported claim about every electromagnetic system.

## Construct observables from potentials

The scalar potential V has units of volts; vector potential A has units of volt-seconds per metre. Both terms in E consequently have units of volts per metre, while B is measured in tesla. Writing B as a curl enforces zero magnetic divergence for smooth potentials. The corresponding expression for E enforces Faraday law. These are two important identities, but they do not by themselves establish Gauss and Ampere laws for an independently prescribed charge and current. We will keep that distinction when comparing examples.

$$
\mathbf E=-\nabla V-\partial_t\mathbf A,\qquad \mathbf B=\nabla\times\mathbf A.
$$

## A coordinated gauge transformation

Choose a smooth scalar function chi. If its spatial gradient is added to A, its time derivative must be subtracted from V. Chi has units of volt-seconds, making both changes dimensionally consistent. Practising only time-independent transformations can conceal why the scalar change is necessary. The two changes must use the same function. Arbitrarily editing V and A independently is not automatically a gauge transformation, even if the resulting pictures look related. Think of the coordinated rule as a constraint on how the representation may change.

$$
\mathbf A^{\prime}=\mathbf A+\nabla\chi,\qquad V^{\prime}=V-\partial_t\chi.
$$

## Derive the cancellation in the electric field

Commuting the space and time derivatives shows that the added electric contributions cancel in exact arithmetic. The curl of the added gradient also vanishes. This works because the two changes have matching magnitude and opposite sign, not because the potentials are small. It is useful to write the cancellation before entering numbers into a calculator. Subtracting large rounded contributions can lose a small background field. Such numerical loss is a property of the evaluation procedure; it does not invalidate the underlying gauge identity.

$$
\mathbf E^{\prime}=\mathbf E+\nabla\partial_t\chi-\partial_t\nabla\chi=\mathbf E,\qquad \mathbf B^{\prime}=\mathbf B+\nabla\times\nabla\chi=\mathbf B.
$$

## Connect normalized coordinates to SI units

Choose length scale L, potential scale U, and speed c determined by the medium. The coordinates xi, eta, zeta, and tau are dimensionless. Label normalized plots accordingly and convert physical potentials and fields with their own scales. Tau equal to one is not one second. For L equal to one metre and the vacuum teaching speed, it corresponds to approximately 3.336 nanoseconds. If U is zero, all physical potentials and fields vanish even though a normalized coefficient can remain nonzero. This is a change of scale, not a new physical effect.

$$
\boldsymbol\xi=\mathbf x/L,\quad \tau=ct/L,\quad V=Uv,\quad \mathbf A=(U/c)\mathbf a,\quad \chi=(UL/c)\widetilde\chi,\quad \mathbf E=(U/L)\mathbf e,\quad \mathbf B=(U/(cL))\mathbf b.
$$

## Build a reference induction field

From this point the formulas use normalized coordinates. The function b(tau) is the coefficient of a magnetic field in the z direction. Taking the curl of the vector potential gives b directly; taking its time derivative produces a circulating electric field. Since the magnetic field is spatially uniform, its curl is zero. The required normalized current is therefore minus the time derivative of e. A quadratic b makes that current explicit. This reference family is a driven local field, not an assertion that the configuration is a freely propagating source-free wave.

$$
b=b_0+b_1\tau+\tfrac12b_2\tau^2,\quad \mathbf a_0=(-b\eta/2,b\xi/2,0),\quad v_0=0,\quad \mathbf e_0=(b^{\prime}\eta/2,-b^{\prime}\xi/2,0),\quad \mathbf b_0=(0,0,b).
$$

## Choose a space- and time-dependent chi

Use a product of the three coordinates multiplied by a time polynomial. The spatial Laplacian is zero because the expression is linear in each spatial coordinate separately. Its second time derivative need not vanish. This gives a particularly transparent example in which the physical fields stay fixed while the Lorenz condition changes. A query on a coordinate plane may make the added term vanish at that point. That local zero does not prove that the transformation preserves the condition throughout the region and over time.

$$
\widetilde\chi=\alpha\xi\eta\zeta g(\tau),\quad g=g_0+g_1\tau+\tfrac12g_2\tau^2,\quad \nabla_{\xi}\widetilde\chi=\alpha g(\eta\zeta,\xi\zeta,\xi\eta),\quad \Delta_{\xi}\widetilde\chi=0.
$$

## Read the two changing contributions separately

Define the vector h containing the added mixed derivative. The scalar contribution becomes plus h, while the vector contribution becomes the reference field minus h. These are contributions to one electric field, not two independently measurable forces. Looking at just one arrow can therefore be misleading. Compare original potentials, transformed potentials, both contributions, and their sum at the same spatial point and time. The coordinated example should preserve the total even when the decomposition changes substantially. The partial changes below deliberately remove one side of that cancellation.

$$
\mathbf h=\alpha g^{\prime}(\eta\zeta,\xi\zeta,\xi\eta),\qquad -\nabla_{\xi}v^{\prime}=\mathbf h,\qquad -\partial_{\tau}\mathbf a^{\prime}=\mathbf e_0-\mathbf h.
$$

## A nonzero example to calculate by hand

Set the coefficients as shown and evaluate at the stated point and time. At this instant chi itself and the added A are zero, but their time derivatives are not. The transformed scalar potential is minus two. The two electric contributions are (2,1,2) and (0,-2,-2); their sum is (2,-1,0), the original field. This is an instructive check against reasoning only from a potential snapshot. A zero instantaneous value of a function says nothing by itself about whether that function is changing.

$$
b_0=1,\ b_1=2,\ b_2=0,\ \alpha=1,\ g_0=0,\ g_1=1,\ g_2=0,\quad (\xi,\eta,\zeta,\tau)=(1,2,1,0),\quad \mathbf e^{\prime}=(2,-1,0).
$$

## The Lorenz condition is an additional choice of representation

Define normalized G as the divergence of a plus the time derivative of v. It vanishes for the reference potentials. The coordinated transformation generally changes it when g2 is nonzero, while preserving E and B. A failed Lorenz-condition check is therefore not automatically a failed physical Maxwell solution. A residual transformation preserves the Lorenz representation when its gauge function satisfies the homogeneous wave equation. The next unit explains why the condition matters when using separated wave equations for the potentials. Notice the spelling Lorenz, distinct from Lorentz.

$$
\widetilde G=\nabla_{\xi}\cdot\mathbf a+\partial_{\tau}v,\quad \widetilde G^{\prime}=\widetilde G+(\Delta_{\xi}-\partial_{\tau}^2)\widetilde\chi=-\alpha\xi\eta\zeta g_2.
$$

## Controls that change only A or only V

Changing only A gives the reference electric field minus h; changing only V gives the reference field plus h. These generally change the physical electric field. The magnetic field can remain unchanged in both cases, so a B-only verification would miss the difference. Distinguish a point where h happens to vanish from a family where it vanishes everywhere. In this polynomial family, zero alpha or a time-independent g makes h identically zero. The name of a control is not a verdict: inspect its actual parameters and the scope of the comparison.

$$
\mathbf e_{A\,{\rm only}}=\mathbf e_0-\mathbf h,\qquad \mathbf e_{V\,{\rm only}}=\mathbf e_0+\mathbf h.
$$

## The same sources need not specify the same fields

Infer normalized sources from the actual fields rather than assuming the original current still works. The divergence of h is zero. When g2 is zero, h is also time independent, so the partial changes can preserve both charge and current while changing E. Different initial or boundary data can account for those different fields. When g2 is nonzero, the required current changes too. Keeping the original source request while omitting that change fails Ampere law. Source equality, field equality, and gauge-condition equality are three different questions.

$$
\widetilde\rho=\nabla_{\xi}\cdot\mathbf e,\quad \widetilde{\mathbf j}=\nabla_{\xi}\times\mathbf b-\partial_{\tau}\mathbf e,\quad \rho=(\epsilon U/L^2)\widetilde\rho,\quad \mathbf J=(\epsilon cU/L^2)\widetilde{\mathbf j}.
$$

## Check the curl through a closed loop

Traverse a fixed rectangle counterclockwise in the xy plane, so its normal is plus z. The added gradient contributes four edge integrals whose total is zero. The circulation of A equals the magnetic flux. Converting the normalized circulation to SI introduces the scale UL/c. The electric circulation instead has scale U and matches minus the time derivative of that flux. Reversing the traversal reverses the sign. A rectangle with exactly zero area is distinct from a positive width whose half-width cannot be represented numerically; the latter cannot support an invented geometric measurement.

$$
\oint\mathbf A\cdot d\boldsymbol\ell=(UL/c)bwh=\Phi_B,\qquad \oint\mathbf E\cdot d\boldsymbol\ell=-U b^{\prime}wh=-d\Phi_B/dt.
$$

## Separate numerical observations from identities

Compare the analytically preserved reference field with the sum of rounded contributions. Cancelling symbolically first is useful for this prescribed family, but it is not an independent certificate. Finite differences of the actual potentials and oriented edge quadrature provide separate numerical observations. Those observations still have discretization and rounding error; they are not rigorous global error bounds. If a nonzero product cannot be represented, report an unavailable observation rather than silently replacing it with zero. A symmetry zero, zero amplitude, zero area, and arithmetic loss have different meanings.

## A comparison protocol for an AI model

An AI model trained to output potentials can be penalized unfairly by pointwise potential error when another gauge represents the same fields. Evaluate E and B, Maxwell residuals with the requested sources, initial and boundary conditions, and the chosen gauge condition separately. The opposite shortcut is also unsafe: allowing arbitrary potential changes can hide partial changes that alter the physical field. Record physical equivalence and representation compliance as separate judgments. For background on coordinated transformations and the Lorenz representation, see [Feynman II §18–6](https://www.feynmanlectures.caltech.edu/II_18.html). The polynomial examples and numerical comparisons here are explicit local constructions.

## Exercises with explanations

### Exercise 1

What happens to V when chi is time independent?

**Answer:** Its time derivative is zero, so V stays fixed. The added gradient changes neither the curl nor the time derivative of A, preserving both fields.

### Exercise 2

In the worked example, what is e when only A changes?

**Answer:** Subtract h=(2,1,2) from e0=(2,-1,0), obtaining (0,-2,-2). Preserving B alone does not establish gauge invariance.

### Exercise 3

For g2=alpha=xi=eta=zeta=1, find the coordinated G and the change in E.

**Answer:** G is minus one while the electric-field change is zero. These potentials leave the Lorenz representation while describing the same physical fields.

### Exercise 4

Convert normalized e=(2,-1,0) with L=2m and U=3V.

**Answer:** Multiply by U/L=1.5V/m to obtain (3,-1.5,0)V/m. A normalized electric coefficient is not itself measured in volts.

### Exercise 5

For nonzero alpha and time-linear g, are sources and fields preserved by a partial change?

**Answer:** The extra charge and current are zero, but h is generally nonzero and E changes. Check initial and boundary data rather than inferring field equality from source equality.

### Exercise 6

With increasing b and a fixed loop normal plus z, what is the electric-circulation sign?

**Answer:** It is negative for positive area and U: minus U times b-prime times area. Reversing the traversal reverses the sign, consistent with Faraday law.
