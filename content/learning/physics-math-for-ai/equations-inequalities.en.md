## A solution is a set of allowed inputs

Solving an equation means finding every input in the stated domain that makes a statement true. The result can be one number, several numbers, an interval, no values or every value. A numerical solver returning one candidate does not decide which of these situations applies.

This unit uses real inputs. Complex solutions are valid in another domain, but must not be silently mixed into a real inequality, where an order is required.

## Which transformations preserve the answer?

Adding the same expression to both sides preserves equality when all expressions remain defined. Multiplying by a nonzero number is reversible. Multiplying by zero loses information, and dividing by an unknown expression requires checking its nonzero condition.

For x²=x, factoring gives x(x−1)=0 and solutions {0,1}. Dividing by x before considering zero incorrectly removes one solution. Squaring both sides can introduce candidates: x=−1 becomes x²=1, whose positive solution does not satisfy the original equation. Always substitute candidates into the original conditions.

## Solve the linear family completely

For ax=b:

| Condition | Real solution set |
| --- | --- |
| a≠0 | {b/a} |
| a=0,b=0 | All real numbers |
| a=0,b≠0 | Empty set |

At a=2,b=4 the solution is 2, while probe x₀=1 has residual −2. At a=b=0 there is no unique inverse calculation: every real x works. The lab reports infinitely many solutions rather than calling zero the answer.

A residual measures how a particular candidate fails the equation. It is not an error to a unique true solution when the problem is nonidentifiable or has multiple solutions.

## Inequalities have direction

Adding the same quantity preserves order. Multiplying or dividing by a positive quantity preserves the inequality direction; a negative quantity reverses it. Thus −2x<4 gives x>−2. If the sign of a divisor is unknown, split into positive, negative and zero cases before dividing.

Strict and non-strict inequalities also differ at boundaries. The distinction remains even when a plotted point looks indistinguishable at screen resolution.

## Quadratic roots and sign intervals

For x²+ax+b, completing the square gives

$$
x^2+ax+b=(x+a/2)^2+b-a^2/4.
$$

The discriminant D=a²−4b determines real roots. If D<0, an upward monic parabola is positive everywhere. If D=0, it touches zero at one point. If D>0, its roots are (−a±√D)/2, and it is nonpositive on the closed interval between them.

For x²−3x+2=(x−1)(x−2), the real solutions of ≤0 are [1,2]. At 1.5 the value is −0.25. At the endpoints it is zero, so they are included. The strict inequality would instead use (1,2).

## Read the model without confusing its modes

Linear mode plots ax and the constant b. Quadratic mode plots x²+ax+b and zero. The actual selected statement appears above the plot. Parameters change the function; the probe changes only the readout, not the function itself.

The displayed x window is −4 to 4. A root outside this window can exist and is still reported numerically. A missing visible intersection is not proof that the equation has no solution. Empty real sets are displayed as ∅; no uniquely enumerable finite set is given for an all-real solution.

Computed probe values use floating-point arithmetic. Very near a root, distinguish an approximate decimal from the exact algebraic endpoint before interpreting a tiny residual or sign.

## AI connections and practice

Feasible sets in constrained optimization are defined by equations and inequalities. Incorrect sign handling can change the feasible region. Penalty minimization does not automatically enforce an exact constraint, and a small residual can coexist with ambiguous parameters.

1. Solve 0x=3. **Answer:** ∅.
2. Solve x²−4≤0. **Answer:** [−2,2].
3. Explain which solution is lost by dividing x(x−1)=0 by x.
4. For a fitted linear system, distinguish a candidate residual, uniqueness of a solution and sensitivity to perturbation.

See [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e) for equation and inequality methods. These laboratory cases are direct algebraic calculations, not numerical optimization guarantees.
