"""Independent finite checks for the declared binary equilibrium thermometer.

Enumerate named microstates and all count outcomes, differentiate their
probabilities numerically, and compare with the derived score and Fisher bound.
These checks do not certify arbitrary probes or correlated observations.
"""
from math import comb, exp, log, isclose, tanh
import json


def named_probabilities(T, gap, g):
    weights = [1.0] + [exp(-gap / T)] * g
    return [w / sum(weights) for w in weights]


def close(a, b, tolerance=2e-8):
    assert isclose(a, b, rel_tol=tolerance, abs_tol=tolerance * 1e-3), (a, b)


cases = 0
for g in [1, 2, 5]:
    for gap in [0.4, 1.0, 3.0]:
        for T in [0.5, 1.0, 4.0]:
            q = named_probabilities(T, gap, g)
            energies = [0.0] + [gap] * g
            p = sum(q[1:])
            U = sum(a * b for a, b in zip(q, energies))
            variance = sum(a * (b - U)**2 for a, b in zip(q, energies))
            information = (gap / T)**2 * p * (1 - p)
            close(information, variance / T**2)
            h = 1e-5
            plus = named_probabilities(T * exp(h), gap, g)
            minus = named_probabilities(T * exp(-h), gap, g)
            scores = [(log(a) - log(b)) / (2*h) for a, b in zip(plus, minus)]
            # Central log differentiation has O(h²) truncation plus roundoff.
            assert abs(sum(a * s for a, s in zip(q, scores))) < 1e-9
            close(sum(a * s*s for a, s in zip(q, scores)), information)
            # Independently differentiate the mean, holding the spectrum fixed.
            up = sum(a*b for a, b in zip(plus, energies))
            um = sum(a*b for a, b in zip(minus, energies))
            close((up - um) / (2*h*T), information)
            # Full finite count law, no Monte Carlo or reused production solver.
            for N in [1, 3, 9]:
                count_law = [comb(N,m) * p**m * (1-p)**(N-m) for m in range(N+1)]
                close(sum(count_law), 1)
                count_scores = [(gap/T) * (m - N*p) for m in range(N+1)]
                close(sum(a*s*s for a,s in zip(count_law,count_scores)), N*information)
            for scale in [0.25, 2, 7]:
                for a,b in zip(q, named_probabilities(scale*T, scale*gap, g)):
                    close(a,b)
            # The two parameter scores form a singular Gram matrix.
            gap_plus = named_probabilities(T,gap*exp(h),g)
            gap_minus = named_probabilities(T,gap*exp(-h),g)
            gap_scores = [(log(a)-log(b))/(2*h) for a,b in zip(gap_plus,gap_minus)]
            for a,b in zip(scores,gap_scores): close(a,-b)
            # Invert the exact population occupancy, away from boundaries.
            close(gap / log(g*(1-p)/p),T)
            cases += 1

# Independently bracket the unique g=1 maximum; verify neighboring slopes.
lo,hi=2.0,3.0
for _ in range(80):
    mid=(lo+hi)/2
    if mid*tanh(mid/2)<2:lo=mid
    else:hi=mid
x=(lo+hi)/2
close(x,2.3993572805154677)
assert 2/(x-.01)-tanh((x-.01)/2)>0
assert 2/(x+.01)-tanh((x+.01)/2)<0

# Check boundary likelihood monotonicity by enumerating counts at g=1, N=4.
for m in [0,2,3,4]:
    values=[]
    for T in [.01,.1,1,10,100]:
        p=sum(named_probabilities(T,1,1)[1:])
        values.append(m*log(p)+(4-m)*log(1-p))
    assert all((b<a if m==0 else b>a) for a,b in zip(values,values[1:]))
close(1/log(3),0.9102392266268373)
print(json.dumps({'status':'passed','namedStateCases':cases,
                 'finiteCountLaws':cases*3,'boundaryCountCases':4,
                 'relativeInformationMaximumX':x}))
