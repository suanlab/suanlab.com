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

## Exercise and research implication

Predict the two first outputs when r=−1. Linear index zero remains one; circular index zero becomes −1 because the wrapped final contribution is −2. Load each mode, inspect the table and verify your arithmetic. When r=0 both operators agree on their common indices, but their output lengths still differ.

In image deblurring or Fourier neural operators, a spectral multiplication implicitly encodes a boundary convention. Zero padding, periodic continuation and other extensions can produce different edge behavior even with identical interior kernels. A fast multiplication theorem is not evidence that the chosen boundary matches the data. Record the extension and crop rule as part of the mathematical model before judging reconstruction accuracy.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
