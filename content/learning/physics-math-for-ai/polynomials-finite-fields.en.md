## A bit encoding for polynomials

Coefficients in F2 are zero and one, with one plus one equal to zero. Read the three bits b2b1b0 as b2t²+b1t+b0. Integer two encodes t, four encodes t² and five encodes t²+1. Integer ordering does not measure polynomial magnitude. Addition cancels matching terms and therefore uses bitwise XOR rather than integer addition modulo eight.

$$
(t+1)+(t+1)=0\quad\text{over }\mathbb F_2.
$$

## Multiply and reduce a polynomial remainder

Polynomial multiplication adds exponents and combines coefficients in F2. Reduce the result modulo the chosen polynomial. For m=t³+t+1, the relation t³=t+1 makes t times t² equal t+1, encoded three. Elements are equivalence classes represented by polynomials of degree below three, giving eight elements. Read the multiplication table for all eight bit encodings.

$$
\mathbb F_2[t]/(t^3+t+1),\qquad t\,t^2\equiv t+1.
$$

## Why irreducibility matters

A quotient by an irreducible polynomial over a field is a field. A cubic is irreducible when it has no linear factor. Neither zero nor one is a root of t³+t+1, so this cubic is irreducible over F2. In the reducible mode, t³+t=t(t+1)², and the nonzero elements t and t²+1 multiply to zero. Having eight elements alone does not guarantee a field.

$$
t(t^2+1)\equiv0\pmod{t^3+t}.
$$

## Connections and boundaries

In the default field, the inverse of t is t²+1, encoded five, because t³+t reduces to one. Check that all seven nonzero elements have inverses. Finite fields support exact calculations in coding, checks and algebraic reasoning. The bit encoding is an implementation convention; it does not provide an ordinary real embedding metric or derivative. This lab compares two explicit moduli rather than searching arbitrary irreducible polynomials or decoding a code.

## Exercises with explanations

1. What does bit encoding three mean? **t+1.**
2. Is F8 the same ring as Z8? **No: Z8 has nonzero zero divisors and F8 does not.**
3. Is the zero polynomial invertible? **No.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).
