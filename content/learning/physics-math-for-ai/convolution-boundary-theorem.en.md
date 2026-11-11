## A local rule needs a boundary rule

Convolution applies one translation-invariant linear rule repeatedly. A short impulse response h tells us how each input sample spreads to neighboring output positions. On an infinite sequence, with finite support here, outputs sum all shifted contributions. Circular convolution instead wraps indices around a fixed period N.

$$
(x*h)[n]=\sum_k x[k]h[n-k],\qquad
(x*_Nh)[n]=\sum_{k=0}^{N-1}x[k]h[(n-k)\bmod N].
$$

These are different operators. Correlation, often called convolution in machine-learning libraries, also differs by its reversal or conjugation convention; check the actual library definition before comparing kernels.

## Calculate a small example first

Take x=[1,0,0,2] and h=[1,r]. At r=0.5, linear convolution gives [1,0.5,0,2,1]. The final one comes from the last input sample times the delayed tap. Four-point circular convolution gives [2,0.5,0,2]: that last one wraps into index zero and adds to its existing one.

Blue shows the five linear outputs, orange the four circular outputs. An index outside the selected output length is undefined in that view; it is not an extra observed zero. Selecting index four in circular mode therefore leaves the selected-output readout empty.

## Why frequency multiplication works

The DFT diagonalizes circular shifts. Expanding the convolution sum and reindexing a finite periodic sum gives

$$
\operatorname{DFT}_N(x*_Nh)[k]=X[k]H[k].
$$

To compute a linear result using the same identity, pad both inputs with zeros to at least length 4+2−1=5. The experiment uses length eight for radix-two FFT convenience. Its inverse transform gives the five linear outputs followed by three zeros. Padding only to four forces the wrap described above. The reported reconstruction error compares the complex inverse result with the appropriate boundary-specific target.

## State the integrability needed to exchange sums or integrals

For f,h∈L¹(R), Tonelli gives ∫∫|f(u)h(t−u)|du dt=‖f‖₁‖h‖₁. Thus convolution exists almost everywhere, belongs to L¹, and Fubini justifies changing t=u+v in its Fourier transform. Under the lab's angular convention this proves F(f*h)=F(f)F(h), with no additional factor. Multiplication of time functions has a different convolution formula and its own assumptions.

For h∈L¹ and f∈L², weighted Cauchy–Schwarz gives |h*f(t)|²≤‖h‖₁∫|h(u)||f(t−u)|²du. Integrate once more to obtain Young's bound

$$
\|h*f\|_2\le\|h\|_1\|f\|_2.
$$

An impulse response of finite support satisfies these discrete counterparts, but arbitrary nonintegrable kernels require a different operator or distribution framework. A formal rearrangement alone is not a proof of the theorem there.

## Compare a cropped Toeplitz operator with a circulant operator

Let S be the cyclic one-step shift on four coordinates. For h=[1,r], the periodic operator C=I+rS commutes with S and is diagonalized by DFT modes. The zero-extended output cropped to four positions is a lower-bidiagonal Toeplitz matrix T. It loses the top-right wrap entry. These are finite operators with different boundary assumptions, not two algorithms for the same output.

| r | Cropped zero-extension output | Periodic output | ‖TS−ST‖F |
|---|---|---|---|
| 0 | (1,0,0,2) | (1,0,0,2) | 0 |
| 1/2 | (1,1/2,0,2) | (2,1/2,0,2) | .707107 |
| 1 | (1,1,0,2) | (3,1,0,2) | 1.414214 |

Here x=(1,0,0,2), ‖TS−ST‖F=√2|r|, and CS−SC=0. The code forms the matrices and their commutators exactly with fractions. This finite crop breaks cyclic shift equivariance. It does not say that full zero-extended convolution on all integers breaks ordinary translation equivariance.

**Exercise:** Why is zero-padding sufficient to use FFT for the full linear convolution? **Explanation:** For a transform period at least input length plus filter length minus one, no nonzero output wraps into another output index.

```python
from fractions import Fraction as F
from math import sqrt

N=4
S=[[F(int(i==(j+1)%N)) for j in range(N)] for i in range(N)]
x=list(map(F,[1,0,0,2]))
def matmul(A,B): return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]
for r in [F(0),F(1,2),F(1)]:
    T=[[F(i==j)+r*int(i==j+1) for j in range(N)] for i in range(N)]
    C=[[F(i==j)+r*int(i==(j+1)%N) for j in range(N)] for i in range(N)]
    def commutator(A):
        AS,SA=matmul(A,S),matmul(S,A)
        return sum((AS[i][j]-SA[i][j])**2 for i in range(N) for j in range(N))
    assert commutator(C)==0 and commutator(T)==2*r*r
    T_y=[sum(a*b for a,b in zip(row,x)) for row in T]
    C_y=[sum(a*b for a,b in zip(row,x)) for row in C]
    print('r, cropped, periodic, cropped commutator norm:',str(r),
          [str(v) for v in T_y],[str(v) for v in C_y],round(sqrt(float(2*r*r)),6))
```

## Exercise and research implication

Predict the two first outputs when r=−1. Linear index zero remains one; circular index zero becomes −1 because the wrapped final contribution is −2. Load each mode, inspect the table and verify your arithmetic. When r=0 both operators agree on their common indices, but their output lengths still differ.

In image deblurring or Fourier neural operators, a spectral multiplication implicitly encodes a boundary convention. Zero padding, periodic continuation and other extensions can produce different edge behavior even with identical interior kernels. A fast multiplication theorem is not evidence that the chosen boundary matches the data. Record the extension and crop rule as part of the mathematical model before judging reconstruction accuracy.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
