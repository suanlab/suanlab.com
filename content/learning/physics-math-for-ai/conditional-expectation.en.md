## Condition on information rather than one event

Conditional expectation given G is a random variable expressible using the available information, not merely one number. Assume X is integrable and G is a sub-sigma-algebra. The defining requirements are G-measurability and preservation of integrals over every G-event. A prediction distinguishing points within one information atom would use information not present in G.

$$
Y=E[X\mid\mathcal G]:\quad Y\text{ is }\mathcal G\text{-measurable},\quad\int_A Y\,dP=\int_A X\,dP\ \forall A\in\mathcal G.
$$

## Construct it on finite atoms

With X=(0,2,1,3), weights (0.1,0.2,0.3,0.4) and atoms {0,1}, {2,3}, choose a constant on each atom. The weighted averages are 4/3 and 15/7. They preserve the integral on each atom, hence on every union of atoms and therefore every G-event.

$$
Y(\omega)=\frac{\sum_{i\in A}p_iX_i}{P(A)}\quad(\omega\in A,\ P(A)>0).
$$

## The tower property and information resolution

Taking the entire space as an event gives the common mean 1.9. Full information recovers X; trivial information gives the constant overall mean. For nested information H contained in G, averaging first under G and then under H agrees with averaging directly under H. General existence connects to the Radon–Nikodym theorem, while uniqueness is only almost sure.

$$
\mathcal H\subseteq\mathcal G\implies E[E[X\mid\mathcal G]\mid\mathcal H]=E[X\mid\mathcal H].
$$

## Null atoms and least-squares interpretation

A null atom has no defined averaging ratio, but any constant chosen there leaves integral preservation unchanged. The lab explicitly chooses zero as a version. This does not define elementary conditioning on a null event to equal zero. For square-integrable X, conditional expectation minimizes squared error among G-measurable predictors. Weighted residuals integrate to zero on every atom, exposing the underlying orthogonality.

$$
E[(X-Z)^2]=E[(X-Y)^2]+E[(Y-Z)^2]\quad(Y=E[X\mid\mathcal G],\ Z\in L^2(\mathcal G)).
$$

## Exercises with explanations

1. Is the overall mean preserved? **Yes: Ω belongs to G.**
2. Is the value on a null atom unique? **No: versions may differ on null sets.**
3. Must more information increase optimal squared error? **No: previous predictors remain admissible.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
