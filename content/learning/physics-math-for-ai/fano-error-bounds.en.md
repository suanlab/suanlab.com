## Remaining uncertainty and classification error

Estimate one of M labels X from observation Y using decoder X̂=g(Y). Let E=1{X̂≠X} have probability Pₑ. Fano's inequality upper-bounds remaining conditional entropy using that decoder's error. Error depends on the decoder as well as the observation.

$$
H(X\mid Y)\leq h_2(P_e)+P_e\log_2(M-1).
$$

## First describe whether an error occurred

Knowing X and Y determines E. The conditional chain rule gives H(X|Y)=H(E|Y)+H(X|E,Y). The first term is at most h₂(Pₑ). With no error, the decoder identifies X. With an error, at most M−1 labels remain, giving the second term at most Pₑ log₂(M−1). Adding these bounds proves Fano.

## A symmetric example attaining equality

The lab uses uniform X with P(Y=X)=1−ε and probability ε/(M−1) for each other label. Its selected decoder returns Y unchanged. Symmetry makes the wrong alternatives conditionally uniform, attaining equality in the sharp bound.

At M=4, ε=.3, conditional entropy is h₂(.3)+.3 log₂3≈1.35678 bits and mutual information is 2−H(X|Y)≈.64322 bits. The table shows one conditional row for X=0; other rows follow by symmetry.

## A simple but loose error lower bound

Using h₂(Pₑ)≤1 and log₂(M−1)≤log₂M produces:

$$
P_e\geq\max\!\left(0,\frac{H(X\mid Y)-1}{\log_2M}\right).
$$

For uniform X, replace H(X|Y) by log₂M−I(X;Y). With four labels and independent output ε=.75, optimal error is .75 but this lower bound is only .5. A bound need not identify exact error. Blue plots this symmetric model's conditional entropy, equal to its sharp Fano expression; orange is the looser upper bound 1+ε log₂M.

## Error one does not imply information zero

At M=2, ε=1, Y always inverts X. Returning Y has error one; an inverting MAP decoder has error zero, with mutual information one bit. Distinguish selected and optimal decoders. Here MAP error is min(ε,1−ε/(M−1)). Mutual information alone does not specify the performance of every decoder on arbitrary data.

## Exercises with explanations

1. How many alternatives remain after an error? **At most M−1.**
2. Does equality here imply equality for all models? **No: the general relation is an inequality.**
3. What if the coarse lower-bound expression is negative? **Take its maximum with zero.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
