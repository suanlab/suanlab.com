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

## Exercises with explanations

1. How long is a uniform eight-bit word? **Five bits:** one mode bit, one value bit and three length bits.
2. May the header be omitted from a standalone comparison? **No.** If mode is shared externally, explicitly state that condition instead.
3. Does lossless mean a corrupted stream is repaired? **No.** A damaged token may decode to another valid source.

The grammar and decoder above define this original lab. Background: [MIT 6.441, Source Coding and Channel Coding](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/pages/lecture-notes/).
