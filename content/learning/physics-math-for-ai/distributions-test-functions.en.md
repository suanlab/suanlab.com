## Ask how delta acts, not how tall it is

A distribution is a continuous linear functional on C_c^∞. Test-function convergence is stronger than pointwise convergence: supports stay in a common compact set and derivatives of every order converge uniformly. Dirac delta returns one test value.

$$
\delta_a[\varphi]=\varphi(a),\qquad T_f[\varphi]=\int f\varphi.
$$

Locally integrable functions define distributions through the second formula, but not every distribution is an ordinary function. Defining delta merely as an infinite point height obscures its calculus.

## Approximate with unit-area triangles

Use ηε(x)=ε⁻¹ max(1−|x−a|/ε,0). Support length is 2ε, height 1/ε and area one. For continuous tests,

$$
\int\eta_\varepsilon(x)\varphi(x)\,dx\longrightarrow\delta_a[\varphi].
$$

The test equals xᵈ on [−2,2] and has a smooth compactly supported extension outside. All triangle supports lie inside this interval, so the undrawn extension does not affect pairings. A polynomial alone is not compactly supported.

## Compute actions exactly

The centered triangular variance is ε²/6. For a quadratic test, the approximate action is a²+ε²/6. At a=.5, ε=.5, exact delta action is .25, triangle action 7/24 and error 1/24. Width .1 reduces error to 1/600.

Blue is a density and orange a local test function; they are not two approximations of the same quantity.

## Differentiate through the test

Transfer differentiation with a minus sign:

$$
T'[\varphi]=-T[\varphi'],\qquad \delta'_a[\varphi]=-\varphi'(a).
$$

For φ=x, the answer is −1. For x² it is −2a, also obtained by the triangle derivative. For x³, delta gives −3a² while the approximation gives −3a²−ε²/2. These signs matter in weak PDEs, point loads and Green functions.

## A jump and a corner have different derivatives

Heaviside H(x)=1{x>0} has classical derivative zero away from zero, but integration against a test gives H′[φ]=−∫₀∞φ′=φ(0). Thus H′=δ₀. In contrast, |x| is continuous at zero, so the two first-derivative boundary terms cancel and its derivative is sign(x). The sign function jumps by two:

$$
H'=\delta_0,\qquad (|x|)'=\operatorname{sign}(x),
\qquad (|x|)''=2\delta_0.
$$

Consequently |x| belongs to H¹ but not H² on a bounded interval containing zero in its interior. Using only the step's pointwise derivative zero misses its jump. Such terms encode point loads in weak PDEs. The function −|x|/2 is a distributional Green function for −d²/dx² on the real line; boundary-conditioned domains have different Green functions.

## Why delta is not an L¹_loc function

Choose smooth φ_h supported in [−h,h], with φ_h(0)=1 and 0≤φ_h≤1. If δ₀=T_f for locally integrable f, then |T_f[φ_h]|≤∫₋ₕʰ|f|→0, whereas δ₀[φ_h]=1. This contradiction rules out an exact ordinary-function representation.

The topology matters too. On a fixed compact support, |δ₀[φ]|≤sup|φ| and |δ₀′[φ]|≤sup|φ′|. Test-function derivative seminorms control continuity of these distributions.

## Check signs and approximation with a smooth bump

Set φ(x)=exp(−1/(1−x²)) for |x|<1 and zero outside. It is a genuine C_c^∞ test. The code numerically checks −∫Hφ′=φ(0) and ∫|x|φ″=2φ(0), and compares triangular actions with delta's exact value e⁻¹.

| ε | Triangle action | Absolute delta error |
|---|---|---|
| .5 | .351752949 | .016126492 |
| .25 | .363998934 | .003880507 |
| .125 | .366918420 | .000961022 |

The symmetric triangle has zero first centered moment and second moment ε²/6. Taylor's remainder bound yields |∫ηεφ−φ(0)|≤(sup|φ″|)ε²/12. The table is a numerical check; the moment estimate separately proves convergence for every smooth test.

**Exercise:** May the second almost-everywhere classical derivative of |x|, zero away from its corner, be used as its weak second derivative? **Explanation:** No. The sign jump contributes 2δ₀, invisible from values away from zero alone.

```python
from math import exp

def bump(x):
    if abs(x)>=1:return 0
    return exp(-1/(1-x*x))
def prime(x):
    if abs(x)>=1:return 0
    return -2*x/(1-x*x)**2*bump(x)
def second(x):
    if abs(x)>=1:return 0
    d=1-x*x
    return bump(x)*(4*x*x/d**4-2/d**2-8*x*x/d**3)
def simpson(f,lo,hi,n=2000):
    step=(hi-lo)/n
    return step/3*sum((1 if i in [0,n] else 4 if i%2 else 2)*f(lo+i*step)
                      for i in range(n+1))

assert abs(-simpson(prime,0,1)-bump(0))<1e-10
assert abs(2*simpson(lambda x:x*second(x),0,1)-2*bump(0))<1e-10
for width in [.5,.25,.125]:
    action=2*simpson(lambda x:(1-x/width)/width*bump(x),0,width)
    print('width, triangle action, error vs delta:',width,
          round(action,9),round(abs(action-bump(0)),9))
```


## Exercises with explanations

1. Halve the width: what happens to quadratic action error? **It becomes one quarter**, because variance scales as ε².
2. Does growing height imply growing mass? **No.** Width shrinks to preserve area one.
3. Do equal actions on one test prove equal distributions? **No.** Equality requires every test; finite experiments do not prove general distributional convergence.

Further reading: [MIT 18.155, Test Functions and Distributions](https://ocw.mit.edu/courses/18-155-differential-analysis-fall-2004/pages/lecture-notes/).
