## Existence and uniqueness ask different questions

Existence asks whether the equation and data admit a solution. Uniqueness asks whether two solutions with those same data must coincide on their common interval. A solver returning one trajectory answers neither question by itself. In particular, visually separating curves are not evidence of nonuniqueness unless their initial conditions and governing equation match.

For an ODE y'=f(t,y), continuity of the field can provide local existence under the usual local-domain assumptions. A local Lipschitz bound in the state is a familiar sufficient hypothesis for uniqueness. These are local results: they do not automatically guarantee a solution remains bounded or exists for all future time.

## A linear field gives a controlled comparison

For y'=lambda y, the field is globally Lipschitz with constant |lambda|. The solution starting at zero is identically zero. The solution starting at epsilon is epsilon exp(lambda t). Positive lambda separates them, yet uniqueness is intact because their initial values differ.

$$
|y_\epsilon(t)-y_0(t)|=|\epsilon|e^{\lambda t}.
$$

The lab reports that initial-data difference explicitly. Sensitivity to initial data and nonuniqueness are distinct properties. A unique flow can amplify uncertainty, making prediction difficult without making the mathematical initial-value problem ambiguous.

## Construct several solutions with identical data

Now use the continuous field f(y)=2 sqrt(max(y,0)) and initial value zero. For every nonnegative waiting time c, define

$$
y_c(t)=(t-c)_+^2,\qquad t\ge0.
$$

Before c, both the state and derivative vanish. After c, the derivative is 2(t-c), equal to twice the square root of the state. At c the one-sided derivatives both equal zero, so the piecewise curve is continuously differentiable and genuinely satisfies the ODE, rather than merely fitting a plot.

All members start at zero. Taking c=0 and c=1/2 gives different solutions at t=1: one and one quarter. This explicit construction proves nonuniqueness for this problem. A solution that stays zero forever is another member of the broader family.

## Why the sufficient condition fails

Near zero, |f(h)-f(0)|/|h|=2/sqrt(h) for positive h. It grows without bound, so no finite local Lipschitz constant works. Failure of a sufficient theorem alone would not prove nonuniqueness for an arbitrary field; the constructed distinct solutions are the decisive evidence here.

## Exercise: diagnose the right phenomenon

Set the waiting time to one and query t=1/2. The waiting solution is zero while the immediate solution is one quarter, and both initial values are zero. Then switch to the linear model with epsilon 0.1. At t=1 the curves again differ, but now their initial values differ by 0.1. State which comparison disproves uniqueness and verify the ODE residual for the waiting solution exactly at its departure time.

Further reading: [MIT first-order differential equations](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The waiting family and matching-data comparison are constructed here.
