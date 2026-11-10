## What does the meter forget?

Imagine four labeled states, drawn as four boxes. A meter paints the first two boxes blue and the last two orange. It reports only a color. If the system is in state 0 or state 1, the reading is blue in either case. The observation is perfectly reliable about color while remaining unable to identify the original state.

Now make state 0 more likely than state 1. The blue label still joins two states, but they are no longer equally likely inside that label. Knowing how many states a reading collects is not the same as knowing their conditional probabilities. This is the step beyond the uniformly prepared binary shells: we keep the grouping operation and allow a general preparation.

The distinction matters for AI representations too. A histogram, cluster label or feature can retain some distinctions and discard others. A coarse representation alone does not specify how to reconstruct the missing details. We will separate the observation from one explicitly chosen reconstruction rule.

## Specify the preparation and the observation map

Let there be n labeled microstates. Nonnegative weights w specify the preparation; a second vector v specifies a comparison distribution. Each vector has a positive total. Normalize them separately:

$$
W=\sum_iw_i>0,\quad V=\sum_iv_i>0,\qquad p_i=\frac{w_i}{W},\quad q_i=\frac{v_i}{V}.
$$

Multiplying all preparation weights by the same positive factor leaves p unchanged. Multiplying only one weight generally changes it. The target q is a second probability law, not automatically a thermal equilibrium distribution. No energy, temperature or clock is supplied in this model.

Choose a deterministic map g from states to declared observation labels m. A group is the inverse image of a label. Its count and its probabilities under the two preparations are:

$$
C_m=\{i:g(i)=m\},\qquad \Omega_m=|C_m|,\qquad P_m=\sum_{i\in C_m}p_i,\quad Q_m=\sum_{i\in C_m}q_i.
$$

Every microstate belongs to exactly one group. Counts describe the map; P and Q describe a map together with a preparation. A declared label can have an empty inverse image. A nonempty group can also have probability zero if all of its preparation weights vanish. These cases need different descriptions even though both have zero observed probability.

## Condition on a reading before naming its hidden state

After a reading with positive probability, renormalize only the compatible states. The preparation and target have their own conditional laws:

$$
r_{i\mid m}=\frac{p_i}{P_m}\quad(P_m>0),\qquad s_{i\mid m}=\frac{q_i}{Q_m}\quad(Q_m>0),\qquad i\in C_m.
$$

There is no division by a count here unless the relevant weights are uniform inside that group. For a zero-probability reading, this formula has no defined conditional distribution. In an abstract probability construction one may choose a version on such events; the calculator instead reports that this conditional was not specified by the preparation.

For a positive-probability group, the two-stage description recovers each original probability:

$$
p_i=P_{g(i)}r_{i\mid g(i)}\qquad(P_{g(i)}>0).
$$

Think of choosing a color first and then a box within that color. Both choices are needed to recover the original random state. The first does not by itself make the second uniform.

## Follow ten units of probability mass through four boxes

Take weights w=(6,2,1,1) and groups {0,1} and {2,3}. Ten is a convenient common denominator; the weights need not represent ten actual repeated trials.

| State | Preparation probability | Reading | Conditional probability within that reading |
| --- | ---: | --- | ---: |
| 0 | 6/10 | blue | 6/8 = 3/4 |
| 1 | 2/10 | blue | 2/8 = 1/4 |
| 2 | 1/10 | orange | 1/2 |
| 3 | 1/10 | orange | 1/2 |

Picture arrows of widths 6 and 2 merging into a blue arrow of width 8. The orange arrows of widths 1 and 1 merge into width 2. The meter reports probabilities .8 and .2. Following the blue arrow backward does not determine whether its hidden split was 6:2, 4:4 or 8:0.

The group sizes are both two, yet the blue conditional entropy differs from the orange conditional entropy. Equal counts alone do not determine equal uncertainty. Keeping the four state probabilities beside the two group probabilities prevents that common visual misreading.

## Derive the information chain rule

We use natural logarithms and information measured in nats, with the continuous extension 0 ln 0=0. Define the state entropy and the observation entropy:

$$
H(X)=-\sum_ip_i\ln p_i,\qquad H(M)=-\sum_mP_m\ln P_m.
$$

Substitute p=Pr in every group with P>0. The logarithm becomes ln P plus ln r. Summing the first part over a group gives −P ln P because its conditional probabilities sum to one. The remaining part is that group's entropy weighted by its probability:

$$
H(X)=H(M)+H(X\mid M),\qquad H(X\mid M)=\sum_{m:P_m>0}P_mH(r_m).
$$

The worked preparation has H(X)≈1.088899975, H(M)≈.500402424 and H(X|M)≈.588497552 nats. Their unrounded values satisfy the identity. The conditional term is an average over possible readings, not the entropy of whichever group is currently selected on screen.

A deterministic observation cannot have more Shannon entropy than its input: the missing conditional information is nonnegative. This statement concerns the random reading M. It does not say that a reconstructed distribution over all original boxes must have smaller entropy.

## Counts bound hidden information but do not fix it

For any group with positive preparation probability, the maximum conditional entropy occurs at the uniform distribution on its compatible labeled states. Consequently:

$$
0\le H(X\mid M)\le\sum_{m:P_m>0}P_m\ln\Omega_m.
$$

For a one-state group, both bounds coincide at zero. For a larger group, a preparation concentrated on one state has conditional entropy zero even though its count exceeds one. The upper bound is reached exactly when every positive-probability group is internally uniform.

Thus a count can bound what a reading hides, but specifying a count does not assert that the preparation attains the bound. In the four-box example the orange group attains ln 2 and the blue group does not. A visualization should show the actual internal split before offering a uniform comparison.

## The same readings can hide different preparations

Replace the preparation by p'=(.8,0,.1,.1), keeping the map fixed. Its reading probabilities remain (.8,.2). Its full entropy is only .639031860 nats, and its hidden conditional entropy is .138629436 nats. The original preparation had the same observation entropy but more hidden uncertainty.

A second table makes the ambiguity explicit:

| Preparation | Blue mass | Orange mass | Hidden split inside blue |
| --- | ---: | ---: | --- |
| (.6,.2,.1,.1) | .8 | .2 | 3/4 versus 1/4 |
| (.8,0,.1,.1) | .8 | .2 | 1 versus 0 |
| (.4,.4,.1,.1) | .8 | .2 | 1/2 versus 1/2 |

All three look identical to this meter. Recovering their differences requires a finer observation or an assumption about internal preparation. In representation learning, matching an aggregate histogram similarly need not match the original distribution over labeled examples or configurations.

## Reconstructing uniformly is an additional rule

Suppose we retain only group probabilities and choose to spread each group's mass uniformly over its compatible states. Define this reconstructed law separately:

$$
\bar p_i=\frac{P_{g(i)}}{\Omega_{g(i)}}.
$$

Every actual state lies in a nonempty group, so the denominator is positive there. A zero-mass nonempty group assigns zero to all its states. An empty declared group has no state requiring a reconstructed coordinate.

For the worked preparation, the rule gives (.4,.4,.1,.1). Its group probabilities remain (.8,.2). Its entropy is:

$$
H(\bar p)=H(M)+\sum_{m:P_m>0}P_m\ln\Omega_m.
$$

This is the maximum entropy among finite laws that retain these group probabilities on this fixed labeled state space. Subtracting the original chain rule yields a more informative expression:

$$
H(\bar p)-H(p)=\sum_{m:P_m>0}P_mD(r_m\Vert u_m)\ge0,\qquad (u_m)_i=\Omega_m^{-1}\ (i\in C_m).
$$

The increase is approximately .104649629 nats for (.6,.2,.1,.1), and .554517744 for (.8,0,.1,.1). The same reconstruction fills in different amounts of missing internal structure. It should not be described as the actual preparation evolving toward a higher-entropy state. We have changed a description by applying an additional rule, without providing any dynamics.

## Compare two preparations before and after observation

For finite distributions, relative entropy compares their probabilities, with a support condition:

$$
D(p\Vert q)=\sum_{i:p_i>0}p_i\ln\frac{p_i}{q_i},\qquad
p_i>0,\ q_i=0\ \Longrightarrow\ D(p\Vert q)=+\infty.
$$

When q is positive wherever p is positive, factor both probabilities into their group masses and conditional laws. Splitting the logarithm and grouping terms gives:

$$
D(p\Vert q)=D(P\Vert Q)+\sum_{m:P_m>0}P_mD(r_m\Vert s_m),\qquad D(P\Vert Q)\le D(p\Vert q).
$$

The observed difference and the conditional difference are both nonnegative totals. Individual terms p ln(p/q) can be negative; they are not themselves nonnegative distances. The implementation therefore uses a different positive summation form when evaluating a very small total.

For p=(.6,.2,.1,.1), q=(.4,.4,.1,.1), the group KL is zero. The entire .104649629-nat difference is hidden inside blue. The orange conditionals agree. Observing the color removes the distinction between these two preparations completely, although they are different microstate laws.

The conditional KL chain rule and data-processing framework can be read further in [Polyanskiy and Wu, MIT 6.441, chapter 2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf). The specific grouping and reconstruction examples here follow directly from the stated finite model.

## When does the reading preserve a comparison?

For a finite comparison satisfying the support condition, the observation preserves KL exactly when it hides no conditional difference:

$$
D(p\Vert q)=D(P\Vert Q)\quad\Longleftrightarrow\quad r_m=s_m\ \text{for every }m\text{ with }P_m>0.
$$

This is a statement about the specified pair p,q. It is not automatically sufficiency for an entire statistical family. Nor is it the condition H(M)=H(X): that entropy equality requires the hidden state to be determined by the reading under p, whereas preserving this particular KL only requires two conditional laws to agree.

For example, p=(.4,.4,.1,.1) and q=(.1,.1,.4,.4) have uniform conditionals inside both groups. Their group probabilities differ, so their KL is positive, but the color meter preserves all of that comparison. Under p, it still hides ln 2 nats about which box was occupied. Preserving a comparison and identifying a state answer different questions.

## Refine the reading instead of guessing the hidden split

Let F be a finer deterministic observation and let M be obtained by merging some F labels. Then the coarse reading can be computed from the fine reading. Applying the same two-stage argument gives:

$$
H(F)=H(M)+H(F\mid M),\qquad H(X\mid M)=H(F\mid M)+H(X\mid F).
$$

Refining the observation moves information from the hidden term into the observed term. It does not change the preparation or increase H(X). Giving every state its own label reveals X completely. Merging all states into one label yields a constant reading and hides all of H(X).

Uniform reconstruction also depends on the chosen partition. In the four-state example, splitting blue into two individual labels prevents their original 6:2 imbalance from being erased by the reconstruction rule. This is a choice of resolution, not evidence that the underlying system physically changed.

## Separate zero mass, undefined conditionals and infinite KL

Take p=(1,0,0), labels (0,1,1), and three declared groups. Group 1 contains two states but has zero preparation mass. Group 2 contains no states. Both observed probabilities are zero, but only the first has compatible states. Neither has a preparation-determined conditional distribution; both contribute zero to the average conditional entropy.

Now take p=(1/2,1/2), q=(1,0). If both states share a label, observed KL is zero while microstate and hidden KL are infinite. If the states have different labels, observed KL is itself infinite. The second target group then has no defined conditional law. The calculator leaves that conditional decomposition undefined rather than subtracting two infinities.

A positive probability below the numerical display range is a different case. It remains in the support and carries a positive sign and logarithmic evidence. The word infinite is reserved for mathematical support mismatch, not an unavailable finite display value.

## Calculate small information differences without subtracting displays

Weight sums can exceed the largest binary64 number even when every normalized probability is ordinary. Conversely, the ratio of the smallest to the largest positive input can underflow. The calculator retains exact binary-rational sums and products until it projects an observation.

For finite KL it evaluates the nonnegative Bregman form, using a series near equality:

$$
D(p\Vert q)=\sum_{i:q_i>0}q_i\,\phi\!\left(\frac{p_i-q_i}{q_i}\right),\qquad
\phi(t)=(1+t)\ln(1+t)-t,\quad \phi(-1)=1,\quad \phi(t)=\tfrac12t^2+O(t^3).
$$

Here the finite-support condition is understood; zero target coordinates with zero preparation mass contribute nothing. One-ULP input biases can produce positive KL far smaller than the rounding error of two entropy displays. Preserve that positive measure directly instead of subtracting the displays or clipping their difference to zero.

There is an even smaller grouped example. Let a and b be positive with δ=a/b tending to zero, use weights (0,b,a) and target weights (a,b,a), and merge the first two coordinates. The two group laws differ only slightly:

$$
P=\left(\frac1{1+\delta},\frac\delta{1+\delta}\right),\quad
Q=\left(\frac{1+\delta}{1+2\delta},\frac\delta{1+2\delta}\right),\qquad
D(P\Vert Q)\sim\tfrac12\delta^3\quad(\delta\to0^+).
$$

At the smallest/largest binary64 weight ratio this is about 10⁻¹⁸⁹⁵. It is finite and positive despite having no linear display value. Independent references use 4096 decimal digits to resolve it. Exact normalization does not make transcendental logs exact; those remain approximations with separately tested accuracy.

## Worked exercises

### Exercise 1: Recover a conditional preparation

Use weights (6,2,1,1), with the first two and last two states grouped. Find the group probabilities and both conditional laws.

**Answer:** The total weight is 10, giving P=(.8,.2). The blue conditional is (6/8,2/8)=(.75,.25); orange is (.5,.5). The count two in each group does not make the blue conditional uniform.

### Exercise 2: Account for observed and hidden information

For the same preparation, combine H(M)≈.500402424 and H(X|M)≈.588497552. Compare them with a preparation (.8,0,.1,.1) having the same observations.

**Answer:** Using unrounded values, the original H(X)≈1.088899975 nats. The second law has H(X)≈.639031860 and hidden information ≈.138629436. Equal observed histograms leave the hidden information undetermined.

### Exercise 3: Reconstruct and identify the assumption

Uniformly reconstruct the first preparation from its two group probabilities. Give its entropy and the increase over the original preparation.

**Answer:** The reconstructed law is (.4,.4,.1,.1). Its entropy is approximately 1.193549604 nats and the increase is .104649629. Uniformity inside each group is an added reconstruction rule; the original probabilities inside blue were .75 and .25.

### Exercise 4: Distinguish two zero-probability groups

Use p=(1,0,0), labels (0,1,1), and group count 3. Which groups have undefined conditionals, and what are their weighted contributions?

**Answer:** Group 1 contains two zero-weight states; group 2 is empty. Both have probability zero and no specified conditional law. Their weighted conditional entropy contributions are exactly zero. A conditional of (.5,.5) for group 1 would be an extra choice, not conditioning this preparation.

### Exercise 5: Hide or reveal a support mismatch

Let p=(1/2,1/2), q=(1,0). Compare a single-label observation with separate labels for both states.

**Answer:** The microstate KL is infinite in both cases. One shared label gives observed KL zero and hidden KL infinite. Separate labels give observed KL infinite, and the second target conditional is undefined. An undefined decomposition must not be replaced by infinity minus infinity.

### Exercise 6: Preserve KL while still hiding states

Use p=(.4,.4,.1,.1), q=(.1,.1,.4,.4), grouped in pairs. Does the observation preserve KL? Does it identify the original state?

**Answer:** Both preparations have uniform pairwise conditionals, so KL is preserved. It equals .8 ln 4 + .2 ln(1/4)=.6 ln 4≈.831776617 nats. The reading still hides ln 2≈.693147181 nats under p, because each occupied pair contains two equally likely states. Preserving this comparison is not state identification.

## Explore the experiment

Start with “Hidden bias”. The preparation weights (6,2,1,1) and target weights (4,4,1,1) use map (0,0,1,1). Compare the four microstate bars with the two group probabilities before choosing a group in “Group to condition on”. The displayed selected conditional entropy is one group's value; H(X|M) in the information table is the probability-weighted average over groups.

| Example | Inspect | Interpretation |
| --- | --- | --- |
| Same reading, different preparation | H(M), H(X) and H(X|M) versus Hidden bias | The same (.8,.2) observation does not determine hidden uncertainty. |
| Preserved KL, hidden states | Weighted hidden KL and H(X|M) | Zero hidden KL preserves this comparison, while ln2 nats about the state remain hidden. |
| Hidden versus Visible support mismatch | Microstate KL, group KL and the selected target conditional | A support infinity can be hidden or revealed by the map; an undefined decomposition is not infinity minus infinity. |
| Zero-mass and empty groups | Group size and actual conditionals for M=1 and M=2 | A nonempty unused group and an empty label both lack a preparation-determined conditional. |
| Positive one-ULP gain | Reconstruction gain, its sign and logarithm | Equal rounded entropy displays need not establish a zero gain. |

The reader supports up to 32 states and 32 declared labels; these are computational limits of this experiment.

Use the weight controls for individual edits. “Extended settings” lets you change the mapping, state count and declared group count together as JSON. Preparation, target and mapping must have the same state count, and every mapping label must lie inside the declared range. Reducing the declared group count with its scalar control clips existing labels into the new range and can merge groups; it changes the observation while retaining the weights. The JSON editor instead requires an explicitly valid map for the supplied group count. An invalid proposal keeps the previous valid preparation.

The figures and comparison scan show separately specified preparations and maps. Uniform reconstruction is an added rule, not a simulated relaxation. The information quantities use nats; converting them to physical entropy requires a suitable physical model and kB. Continue with noisy observation to distinguish uncertainty introduced by a sensor from information hidden by deterministic grouping.
