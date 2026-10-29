## Words and languages

An alphabet Σ is a set of symbols; Σ* contains all finite words. A language is a subset of Σ*. The empty word ε has length zero, whereas the symbol “0” forms a length-one word.

A deterministic finite automaton consists of a finite state set Q, alphabet Σ, transition function δ:Q×Σ→Q, initial state q₀ and accepting set F. Every state-symbol pair has exactly one next state. This lab uses Σ={0,1}.

## State retains only necessary memory

The first language contains words with an even number of ones. It needs only two states: q₀ for even parity so far, q₁ for odd. Zero preserves the state; one toggles it. The initial and accepting state is q₀.

Reading 101 follows q₀→q₁→q₁→q₀ and accepts. Visiting an accepting state midway is insufficient: acceptance is tested after all input symbols are consumed.

## Extend transitions to whole words

Define

$$
\delta^*(q,\varepsilon)=q,\qquad
\delta^*(q,wa)=\delta(\delta^*(q,w),a).
$$

A word w is accepted exactly when δ*(q₀,w)∈F. Prove correctness for every input length using an invariant: the current state equals the parity of the consumed prefix. The empty prefix is even; zero preserves parity and one toggles it. This induction, rather than testing only lengths up to eight, supports the general claim.

## A language ending in 01

The second machine tracks a relevant suffix. State q₀ means no useful suffix, q₁ means the final symbol is zero, and q₂ means the last two symbols are 01. Only q₂ accepts.

Zero from any state goes to q₁. One from q₁ goes to q₂; one from the other states goes to q₀. Thus 010 is rejected and 0101 accepted. The condition is ending in 01, not having encountered 01 somewhere earlier.

## Interpret integer controls as words

The lab takes the lowest N bits of integer w and pads leading zeros to length N. For w=5,N=3 the word is 101; for w=1,N=3 it is 001. N=0 gives ε regardless of w. Higher bits are outside the displayed word. The statement and trace explicitly show the actual consumed symbols.

A double circle marks acceptance; orange marks the final state. Edge labels are symbols, not probabilities or costs. The trace table includes the initial state and every subsequent transition.

## Count accepted words

For even parity and N≥1, toggling the last bit pairs even and odd words, giving 2^(N−1) accepted words. At N=0, the one empty word is accepted.

For words ending in 01 and N≥2, the final two symbols are fixed and the preceding positions are free, giving 2^(N−2). For N<2 the count is zero. A count becomes a probability only after specifying a word distribution.

DFA-recognizable languages are regular. Languages requiring unbounded nesting memory may need a richer model. Success on finite examples does not make a finite automaton a recognizer for every formal language.

## AI connections and exercises

Some output-format constraints, token filters and protocol checks admit finite-state representations. Tokens and characters are different alphabets. Enforcing a regular format does not guarantee factual correctness or semantic validity.

1. Is ε accepted by even parity? **Yes: it has zero ones.**
2. How many length-four words end in 01? **Four.**
3. Is 011 accepted by the ending-01 machine? **No; it finishes at q₀.**

Further reading: [MIT Theory of Computation](https://ocw.mit.edu/courses/18-404j-theory-of-computation-fall-2020/). The two machines and invariant arguments are independently constructed examples.
