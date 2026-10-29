## Check more than the formula

An element of a quotient is an equivalence class, not a chosen integer representative. The formula f([x])=[kx] must give the same answer after replacing x by x+m. For a source Z/mZ and target Z/nZ, the output difference is km. Divisibility of km by n is necessary and sufficient. A plausible-looking list of the first m outputs cannot replace this check.

$$
f([x+m])-f([x])=[km],\qquad f\text{ well-defined}\iff n\mid km.
$$

## Which operation is preserved?

A group homomorphism preserves the group operation. Here that operation is addition. Once the formula is well-defined, distributivity establishes preservation. A ring homomorphism must also preserve multiplication, and one must specify the identity convention. Doubling modulo five is an additive-group isomorphism, but it sends one to two and is not an identity-preserving ring isomorphism.

$$
f(x+y)=f(x)+f(y).
$$

## Kernels and images measure collapse

The kernel contains inputs mapped to the identity. The image contains outputs actually reached. At m=6,n=3,k=1, the kernel is {0,3} and the image is {0,1,2}. The map forgets the distinction between zero and three while reaching every target element. For finite groups, kernel size times image size equals source size. A trivial kernel gives injectivity; full image gives surjectivity. A bijective homomorphism is an isomorphism.

$$
|G|=|\ker f|\,|\operatorname{im}f|.
$$

## Inspect the failure case

With m=3,n=6,k=1, zero and three represent the same source element but map to different target residues. The table then lists candidate values for selected representatives only. Kernel, image and isomorphism readouts remain undefined. An undefined map differs from a valid constant map. For a structure-preserving encoder in AI, specify the equivalence relation and operation that the representation must preserve before evaluating examples.

## Exercises with explanations

1. Is [x]↦[2x] from Z3 to Z6 well-defined? **Yes: six divides two times three.**
2. What does the default map forget? **Differences such as zero versus three.**
3. Is every bijection an isomorphism? **No: operation preservation is also required.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).
