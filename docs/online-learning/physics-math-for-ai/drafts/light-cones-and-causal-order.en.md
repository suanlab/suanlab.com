## 1. Ask whether a signal can arrive

In flat spacetime, an event $B$ can be reached from an earlier event $A$ by a signal traveling at or below light speed if
$$
\Delta ct \geq \|\Delta\mathbf{x}\|,\qquad B\ne A .
$$
Equality is the light cone. A strictly larger time separation allows a subluminal straight path. The condition expresses possible influence under a speed limit. The actual transmission also requires a sender, receiver and a physical channel. [Tong, The Einstein Equations §4.2](https://davidtong.org/teaching/general-relativity/grhtml/S4)

## 2. Separate clock order from causal order

In one spatial dimension a boost gives $\Delta ct'=\gamma(\Delta ct-\beta\Delta x)$. For a future causal separation, $\Delta ct>0$ and $|\Delta x|\leq\Delta ct$, so this remains positive for every $|\beta|<1$. For a spacelike separation, a boost at $\beta=\Delta ct/\Delta x$ makes the two events simultaneous. Frames on opposite sides of that speed can assign opposite time orderings. No permitted signal crosses the separation.

## 3. Combine causal legs

In $1+1$ dimensions a future causal displacement has nonnegative changes in both $u=ct+x$ and $w=ct-x$. A future boost multiplies the two changes by positive reciprocal factors. Adding two future displacements adds the two nonnegative pairs. Therefore reachability is transitive, and a nontrivial chain of future causal legs cannot return to its starting event.

## 4. Intersect a future cone with a past cone

Fix a sender $A=(0,0)$ and deadline $B=(6,b)$, using coordinates $(ct,x)$ with $c=1$. A relay at coordinate time $t$ must be reachable from $A$ and able to reach $B$. Its spatial coordinate belongs to
$$
[-t,t]\ \cap\ [b-(6-t),\,b+(6-t)] .
$$
The first interval comes from the future of $A$. The second comes from the past of $B$. If the intersection has two distinct endpoints, all positions between them can relay in time. Equal endpoints give a single position. An empty intersection means the relay cannot meet both constraints.

## 5. Compare three deadlines

At relay time $t=3$, the positions are $[-1,3]$ for $b=2$, $\{3\}$ for $b=6$, and empty for $b=7$. The first case leaves spatial room to choose the relay. The second consumes all six time units at light speed. The third puts the deadline outside the sender's future cone. Move the relay time earlier or later to see how the interval changes.

```python
def relay_positions(t, b):
    lo = max(-t, b-(6-t))
    hi = min(t, b+(6-t))
    return (lo,hi) if lo <= hi else None

assert relay_positions(3,2) == (-1,3)
assert relay_positions(3,6) == (3,3)
assert relay_positions(3,7) is None
for t in range(7):
    assert relay_positions(t,6) == (t,t)
    assert relay_positions(t,7) is None
print([relay_positions(3,b) for b in (2,6,7)])
```

## 6. Apply the same check to event data

- Draw the two cones before using coordinate timestamps to order candidate causes.
- Use the sign of $\Delta ct$ together with the interval. The squared interval alone cannot identify the future direction.
- For a proposed relay, check both legs. Being inside only one cone is insufficient.
- In a simulator, preserve the distinction between a causal constraint and an observed communication event.
