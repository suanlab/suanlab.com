## Topology specifies which distinctions are observable

In Euclidean space we often define continuity using distances. Topology keeps the idea of neighborhoods without requiring a distance function. A topology on X is a collection of subsets called open sets: it includes the empty set and X, is closed under arbitrary unions, and under finite intersections. A finite example makes the entire definition inspectable rather than relying on a drawing.

Here X={0,1,2} and Y={0,1}. A bit mask encodes a subset: source mask five means {0,2}, while target mask one means {0}. The displayed positions of these elements are labels, not a metric or a neighborhood relation.

## Test every open preimage

A function is continuous when every target-open set has an open preimage:

$$
f\text{ continuous}\quad\Longleftrightarrow\quad
f^{-1}(V)=\{x\in X:f(x)\in V\}\text{ is open for every open }V\subseteq Y.
$$

The indiscrete source has only open masks zero and seven. Let function mask one mean f(0)=1, f(1)=f(2)=0. With a discrete target, both singletons are open. Their preimages are masks one and six, neither source-open. Two failed preimages prove discontinuity. This is not a judgment about whether the three plotted stems look smooth.

A constant function has only empty or full preimages and is continuous for any source topology. Conversely, a discrete source makes every subset open, so every function from it is continuous. An indiscrete target also makes every incoming map continuous. These examples show why the topology must be specified on both sides.

## Connectedness uses separation

A space is connected if it cannot be split into two disjoint nonempty open pieces. Equivalently, its only sets that are both open and closed are the empty set and the whole space. For these finite spaces we enumerate clopen sets and group points that no clopen set separates.

$$
X=A\sqcup B,\quad A,B\ne\varnothing,\quad A,B\text{ open}
\quad\Longrightarrow\quad X\text{ disconnected}.
$$

The indiscrete source has one connected component; the discrete three-point source has three. The chain topology with open masks {0,4,6,7} is connected despite having more open sets than the indiscrete space. Counting opens alone therefore does not determine component count.

## Exercise and explained answer

Can a continuous map from the indiscrete source reach both values of the discrete target? No. A continuous image of a connected space is connected: a separation of the image would pull back to a separation of X. Thus any such map must be constant. Enumerate all eight function masks to verify the finite claim, then give the preimage argument for a general connected source.

## Connection to data analysis

Classification into discrete labels need not be continuous on a connected input domain. A discontinuous decision boundary is not automatically a programming error. Conversely, a connected graph drawn from sampled points does not prove that an unknown population space is connected. Graph neighborhoods, metric neighborhoods and abstract topologies are related constructions with different assumptions. State the construction before interpreting a plot as evidence about data shape.

Further reading: [Allen Hatcher, Notes on Introductory Point-Set Topology](https://pi.math.cornell.edu/~hatcher/Top/TopNotes.pdf).
