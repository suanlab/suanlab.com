## Choose the field before solving

The same symbolic polynomial can have different solutions over the reals, complex numbers or a finite field. The lab checks two constraints on every ordered pair in F5 or F7. The controls s and q are normalized in the chosen field. The table includes residuals at failed candidates as well as solutions. A plotted one means both constraints vanish; it is an indicator, not a probability.

$$
x+y=s,\qquad xy=q\qquad\text{over }\mathbb F_p.
$$

## Eliminate one variable by substitution

The first equation gives y=s−x. Substituting into the second produces a quadratic in x. For p=5,s=3,q=2, the polynomial factors as (x−1)(x−2), giving (1,2) and (2,1). These are distinct ordered pairs because the variables are distinguished. This elimination is elementary substitution, not an implementation of a general Gröbner-basis algorithm.

$$
x(s-x)=q\iff x^2-sx+q=0.
$$

## Discriminants in finite fields

Our moduli five and seven are odd primes, so two is invertible. A zero discriminant gives one repeated-root pair. A nonzero square discriminant gives two pairs; a nonsquare gives none. Do not apply the real positive-versus-negative test. Squares modulo five are zero, one and four, so the case s=0,q=2 requires x²=3 and has no solution.

$$
\Delta=s^2-4q,\qquad x=\frac{s\pm\sqrt\Delta}{2}\quad\text{when a square root exists}.
$$

## What exhaustive enumeration proves

Exhausting the finite grid proves completeness for these equations in the chosen field. With more variables, candidate count grows as p raised to the number of variables, making naive enumeration costly. Success on a small table does not establish an efficient general solver. For AI that proposes or verifies algebraic constraints, distinguish checking one satisfying answer from proving that every solution has been found.

$$
\#\text{candidates}=p^2\quad\text{for two variables}.
$$

## Exercises with explanations

1. What solves s=2,q=1? **Only (1,1).**
2. What is a square root of three modulo five? **None exists.**
3. Must real solutions match finite-field solutions? **No: the coefficient domains differ.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).
