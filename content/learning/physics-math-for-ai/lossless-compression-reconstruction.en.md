## Compression includes a decoder

A short array is not automatically a lossless code. The receiver must uniquely reconstruct its source. This lab encodes a binary word of externally shared length N=8 or 16. If N were unknown, transmitting it would add cost.

The integer mask contributes its lowest N bits. Displayed strings use conventional high-to-low order. Length, bit order and mode headers are all protocol assumptions.

## Specify the grammar

Raw mode sends header 0 followed by N literal bits. Run-length mode sends header 1, the first source bit, and each run length minus one using w=log₂N bits. Subsequent run values alternate, so they need not be transmitted.

$$
\begin{aligned}
L_{\rm raw}&=1+N,\\
L_{\rm RLE}&=2+r\log_2N.
\end{aligned}
$$

Sixteen zeros encode as 1 · 0 · 1111. The last token is fifteen plus one, a run of sixteen. Total length is six bits. The word 0000000011111111 encodes as 1 · 0 · 0111 · 0111, costing ten bits.

## Actually reconstruct the source

Read the mode header. For RLE, repeat the current bit for the decoded length, then flip the bit between runs. Require lengths to sum exactly to N; reject incomplete tokens, invalid symbols and overflow. Raw mode requires exactly N remaining bits.

$$
D(E(x),N)=x,\qquad x\in\{0,1\}^{N}.
$$

The lab displays an actual encoded stream and its decoded result. It chooses raw when RLE is longer, including ties. The receiver still needs the mode header to know which representation was chosen.

## Why some inputs expand

The alternating sixteen-bit word has sixteen runs, giving 66 RLE bits. The encoder instead selects seventeen raw-protocol bits. This exceeds the original sixteen bits by one but remains invertible under the shared grammar.

There are 2ᴺ source words but only 2ᴺ−1 strings shorter than N. Pigeonhole counting therefore prevents an injective compressor from shortening every source word.

For AI checkpoints, include codebooks, scales, indices and format metadata in reported compression. Lossy quantization changes the exact-reconstruction requirement. Correcting transmission damage is a separate coding problem.

## A fixed external length permits prefix decoding

At the same N, raw streams have fixed length N+1 and a different first bit from RLE. Every w-bit RLE token adds a positive run length. Reaching total N terminates the codeword. Appending another token to a valid RLE stream would exceed N, so it cannot create another valid stream. Thus valid streams are prefix-free when the same external N is shared. This does not permit mixing N=8 and N=16 without communicating the distinction.

A source with r runs has two choices of first bit and r−1 boundaries among N−1 gaps: 2·binom(N−1,r−1) possibilities. The binomial theorem gives the Kraft sum of all canonical RLE streams:

$$
S_{\mathrm{RLE}}=\sum_{r=1}^N2{N-1\choose r-1}2^{-(2+wr)}
=2^{-1-w}(1+2^{-w})^{N-1}.
$$

Raw streams contribute 1/2. Selecting only the shorter representation uses a subset of the two valid code families and remains prefix-free. Decodability also makes the selected outputs distinct for different sources.

## Count how often uniform sources actually shrink

Each selected codeword of length at most N−s contributes at least 2⁻⁽ᴺ⁻ˢ⁾ to its Kraft sum. There can be at most 2ᴺ⁻ˢ such words. A uniformly distributed N-bit source therefore saves at least s bits with probability at most 2⁻ˢ:

$$
P_{\mathrm{uniform}}[L(E(X))\le N-s]\le2^{-s}.
$$

This is neither an assertion that real data are uniform nor a universal compression-rate limit for nonuniform data.

| N | Minimum length | Uniform mean length | Fraction shortened | Selected-code Kraft sum |
|---|---|---|---|---|
| 8 | 5 | 8.914062 | .007812 | .585938 |
| 16 | 6 | 16.986847 | .003693 | .571518 |

The rounded table comes from exhaustive evaluation of the current raw/RLE selection rule. The code reconstructs all 256 and 65,536 sources, checks prefixes in length order and computes exact-fraction Kraft sums. Constant sources can shrink substantially while uniform average length still includes protocol cost. Report the source distribution and structure when evaluating compression.

**Exercise:** Can uniform average length fall below N here? **Explanation:** No: a prefix lossless code has average length at least H(X)=N. Shortening a few individual inputs does not contradict the average bound.

```python
from collections import Counter
from fractions import Fraction as F
from itertools import groupby

for n in [8,16]:
    width=n.bit_length()-1
    codes={}
    for value in range(2**n):
        source=f'{value:0{n}b}'
        runs=[len(list(g)) for _,g in groupby(source)]
        raw='0'+source
        rle='1'+source[0]+''.join(f'{length-1:0{width}b}' for length in runs)
        encoded=rle if len(rle)<len(raw) else raw
        if encoded[0]=='0': restored=encoded[1:]
        else:
            bit=encoded[1];restored=''
            for i in range(2,len(encoded),width):
                restored+=bit*(int(encoded[i:i+width],2)+1)
                bit='1' if bit=='0' else '0'
        assert restored==source and encoded not in codes
        codes[encoded]=source
    # Increasing length makes any prefix conflict visible in the seen set.
    seen=set()
    for code in sorted(codes,key=len):
        assert not any(code[:i] in seen for i in range(len(code)))
        seen.add(code)
    histogram=Counter(map(len,codes))
    kraft=sum(F(count,2**length) for length,count in histogram.items())
    assert kraft<=1
    for saving in [1,2,3]:
        short=sum(count for length,count in histogram.items() if length<=n-saving)
        assert F(short,2**n)<=F(1,2**saving)
    average=sum(F(length*count,2**n) for length,count in histogram.items())
    short=sum(count for length,count in histogram.items() if length<n)
    print('N, shortest, longest, uniform mean length, shortened fraction, Kraft sum:',
          n,min(histogram),max(histogram),round(float(average),6),
          round(short/2**n,6),round(float(kraft),6))
```


## Exercises with explanations

1. How long is a uniform eight-bit word? **Five bits:** one mode bit, one value bit and three length bits.
2. May the header be omitted from a standalone comparison? **No.** If mode is shared externally, explicitly state that condition instead.
3. Does lossless mean a corrupted stream is repaired? **No.** A damaged token may decode to another valid source.

The grammar and decoder above define this original lab. Background: [MIT 6.441, Source Coding and Channel Coding](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/pages/lecture-notes/).
