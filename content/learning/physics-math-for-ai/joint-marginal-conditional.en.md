## Begin with the joint table

A joint law for binary variables X and Y assigns probabilities to four pairs. The controls give counts for (0,0),(0,1),(1,0),(1,1), normalized by their total. Default counts (3,1,1,3) give masses (3/8,1/8,1/8,3/8). Matching values receive more probability. Numbers inside the cells give actual masses, so color intensity is not the sole carrier of information.

$$
p_{ij}=P(X=i,Y=j),\qquad\sum_{i,j}p_{ij}=1.
$$

## Summation differs from normalization

Marginalization sums out the variable we do not retain. Conditioning selects a row and divides by that row’s total. Here both marginals assign probability one half to one, while P(Y=1|X=1)=(3/8)/(4/8)=3/4. Merely deleting the other row leaves a total below one; normalization is essential.

$$
P(X=i)=\sum_jp_{ij},\qquad P(Y=j\mid X=i)=\frac{p_{ij}}{\sum_kp_{ik}}.
$$

## Covariance and correlation

Covariance is the expectation of the product of centered variables. Since E[XY]=P(X=1,Y=1)=3/8 here, covariance is 3/8−1/4=1/8. Both variances are one quarter, giving correlation one half. A constant variable has zero variance, making the correlation denominator vanish; the interface leaves that correlation undefined.

$$
\operatorname{Cov}(X,Y)=E[XY]-E[X]E[Y],\quad\operatorname{Corr}(X,Y)=\frac{\operatorname{Cov}(X,Y)}{\sqrt{\operatorname{Var}X\operatorname{Var}Y}}.
$$

## A special binary fact and a general counterexample

Independence factors every cell into a product of marginals. For two binary variables, zero covariance determines the remaining cells and is equivalent to independence. That special fact does not extend generally. If X is uniform on [−1,1] and Y=X², their covariance is zero by symmetry, yet Y is determined by X. Marginals alone cannot recover a joint law. Multivariate prediction therefore requires checking dependence beyond individual-variable accuracy.

$$
X\perp Y\iff p_{ij}=P(X=i)P(Y=j)\quad\text{for every }i,j.
$$

## Exercises with explanations

1. What if the X=1 row has zero mass? **Its conditional ratio is undefined.**
2. What if the total count is zero? **No probability law can be normalized.**
3. What is the default correlation? **0.5.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
