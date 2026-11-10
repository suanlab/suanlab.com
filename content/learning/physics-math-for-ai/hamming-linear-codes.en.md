## Parity lives in a finite field

GF(2) contains zero and one. Addition is XOR and multiplication is AND, so 1+1=0. Matrix multiplication can then test whether selected bits sum to even parity. This is different from real-valued least squares: use finite-field arithmetic throughout.

Hamming(7,4) adds three parity bits to four information bits. Columns of H are binary positions 1 through 7, with the low-order row first:

$$
H=\begin{pmatrix}1&0&1&0&1&0&1\\0&1&1&0&0&1&1\\0&0&0&1&1&1&1\end{pmatrix}.
$$

## Encode and locate one error

Place low-order payload bits at positions 3,5,6,7 and solve parity bits at 1,2,4 so Hc=0. Valid codewords form the four-dimensional null space of H. A generator matrix G equivalently satisfies c=Gm and HG=0.

Payload integer 9 has low-first bits (1,0,0,1), producing codeword 0011001 when read from position one. Flipping position three yields syndrome (1,1,0), interpreted low-first as integer three. Flip that position again.

$$
\begin{aligned}
r&=c+e,\\s&=Hr=He\pmod 2.
\end{aligned}
$$

## Derive the distance guarantee

Every column is nonzero and distinct. Thus no weight-one or weight-two vector lies in the null space. Columns one and two XOR to column three, so the vector supported at positions 1,2,3 is a valid weight-three codeword. Minimum distance is exactly three.

$$
d_{\min}=3,\qquad t=\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor=1.
$$

## Passing parity does not prove recovery

For payload 9, flip positions one and two instead. Their combined syndrome is again three. The single-error decoder flips position three and reaches another valid codeword, decoding payload 8. Post-correction syndrome is zero despite incorrect information.

The original code can detect up to two errors if used only as a detector. This particular decoder instead interprets every nonzero syndrome as one error and can miscorrect two. SECDED designs add an overall parity bit or another mechanism; that extension is not implemented here.

Blue and orange points compare sent and received bits. The table also shows corrected bits, making miscorrection visible.

## Exercises with explanations

1. Can a parity-position error be corrected? **Yes.** Every position has its own nonzero column.
2. Is the XOR of codewords valid? **Yes.** Linearity gives H(c₁+c₂)=0.
3. What is the rate? **4/7.** Count information bits over GF(2), not real-vector coordinates.

Further reading: [MIT MAS.160, Hamming Codes and Syndromes](https://ocw.mit.edu/courses/mas-160-signals-systems-and-information-for-media-technology-fall-2007/94144b6cc83df4bea4f9cef22c91baf4_1017_error.pdf).
