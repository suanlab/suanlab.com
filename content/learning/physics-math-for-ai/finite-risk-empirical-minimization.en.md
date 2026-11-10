## Which average is being taken?

Training error averages already observed examples. Population risk averages over the joint distribution of future inputs and labels. They remain different quantities even with identical loss functions. Here x=0,1,2,3,4 are equally likely. Inputs at least two usually have label one, while smaller inputs usually have label zero. The label flips with probability epsilon. Because this teaching world is fully specified, population risks can be computed exactly rather than estimated by a test simulation.

The six hypotheses are h_t(x)=1{x≥t}, for t=0,...,5. Thresholds zero and five are constant-one and constant-zero classifiers. Blue shows actual sample risk, orange population risk, and markers show a separately selected query hypothesis rather than necessarily the ERM.

## Define risk and ERM

$$
R(h)=E_{(X,Y)\sim P}[1\{h(X)\ne Y\}],\qquad
\widehat R_S(h)=\frac1N\sum_{i=1}^N1\{h(X_i)\ne Y_i\}.
$$

ERM minimizes empirical risk over the declared class. Ties choose the smallest threshold. Changing the class or tie rule can select a different model from the same data. A concentration inequality for one hypothesis fixed independently of data cannot simply be transferred to a data-selected ERM. A finite-class union bound or another complexity argument is needed.

## Population oracle in this world

Since noise is below one half, threshold two predicts the more likely label at every input. Direct enumeration gives

$$
R(h_t)=\epsilon+(1-2\epsilon)\frac{|t-2|}{5},\qquad R(h_2)=\epsilon.
$$

Even this oracle cannot remove intrinsic label noise. A realized sample ERM can nevertheless have empirical error below or above epsilon. The inequality E[min Rhat]≤min E[Rhat] concerns expectations, not every dataset. Forcing every observed generalization gap to be positive would distort the data. The readout keeps its signed value.

## Worked example and exercise

Without noise, population risks from threshold zero through five are [0.4,0.2,0,0.2,0.4,0.6]. At N=1 and seed seven, the actual sample is (x=0,y=0). Thresholds one through five all have zero training error, so the tie rule selects one. Its population risk is 0.2. This is a minimal example of zero training error without zero generalization error.

Exercise: compute risks for t=0 and t=2 when noise is 0.1. Solution: threshold zero disagrees with the majority label on two inputs, giving 0.1+0.8·2/5=0.42. Threshold two has risk 0.1. A single sample's observed label proportions are not these exact expectations.

## Connection to AI evaluation

Real distributions are unknown, so evaluation requires held-out data, an explicit procedure and uncertainty assessment. This exact population curve is an oracle for a declared finite world. Repeatedly choosing classes or hyperparameters from test results creates further selection bias. Record the input law, fixed class, actual samples, selection rule and evaluation target before interpreting a learning curve or applying a generalization bound.

References: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) and [MIT statistical learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/). Text and finite examples are original.
