"""Independent 130-decimal matrix-eigensystem references; no product Bloch formulas.
Decimal centers carry a conservative 1e-110 comparison allowance, not formal interval certificates.
"""
import json
from fractions import Fraction
from pathlib import Path
import mpmath as m
m.mp.dps = 130
states = [(0,0,0),(0,0,1),(0,0,-1),(1,0,0),(0,1,0),(0,0,'3/5'),('4/5',0,0),('1/5','2/5','1/5')]
states = [tuple(map(Fraction,r)) for r in states]
def mp(x): return m.mpf(x.numerator)/x.denominator

def matrix(r):
    x,y,z=map(mp,r)
    return m.matrix([[(1+z)/2,(x-m.j*y)/2],[(x+m.j*y)/2,(1-z)/2]])

def spectrum(a): return m.eighe(a)

def entropy(a):
    values,_=spectrum(a)
    return -sum(v*m.log(v,2) for v in values if v>m.mpf('1e-120'))

def h(x):
    return -sum(v*m.log(v,2) for v in [x,1-x] if v>m.mpf('1e-120'))

def scalar(x): return m.nstr(m.re(x),120)

ensembles=[]
relative=[]
for i,r in enumerate(states):
    for j,s in enumerate(states):
        a,b=matrix(r),matrix(s)
        for prior in map(Fraction,['1/5','1/2','4/5']):
            p=mp(prior);average=p*a+(1-p)*b
            chi=entropy(average)-p*entropy(a)-(1-p)*entropy(b)
            delta=p*a-(1-p)*b
            values,vectors=spectrum(delta)
            effect=m.zeros(2)
            if all(abs(v)<m.mpf('1e-120') for v in values): effect=m.eye(2)
            else:
                for k,v in enumerate(values):
                    if v>m.mpf('1e-120'):
                        column=vectors[:,k];effect+=column*column.H
            for measurement in ['helstrom','z']:
                chosen=effect if measurement=='helstrom' else m.matrix([[1,0],[0,0]])
                probs=[m.re(sum((chosen*x)[k,k] for k in range(2))) for x in [a,b]]
                py=p*probs[0]+(1-p)*probs[1]
                information=h(py)-p*h(probs[0])-(1-p)*h(probs[1])
                ensembles.append(dict(name=f'{i}/{j}/{prior}/{measurement}',bloch0=list(map(str,r)),bloch1=list(map(str,s)),prior=str(prior),measurement=measurement,
                    expected=dict(holevoInformation=scalar(chi),selectedMeasurementInformation=scalar(information),averageStateEntropy=scalar(entropy(average)),classicalLabelEntropy=scalar(h(p)))))
        infinite=sum(x*x for x in s)==1 and r!=s
        entry=dict(name=f'{i}/{j}',rho=list(map(str,r)),sigma=list(map(str,s)),status='infinite' if infinite else 'finite')
        if not infinite:
            if r==s: value=m.mpf(0)
            else:
                eigenvalues,eigenvectors=spectrum(b)
                logarithm=m.zeros(2)
                for k,v in enumerate(eigenvalues):
                    if v>m.mpf('1e-120'):
                        column=eigenvectors[:,k];logarithm+=m.log(v,2)*column*column.H
                value=-entropy(a)-m.re(sum((a*logarithm)[k,k] for k in range(2)))
            entry['value']=scalar(value)
        relative.append(entry)
output=Path('scripts/fixtures/quantum-information-entropy-independent.json')
output.write_text(json.dumps(dict(method='mpmath Hermitian matrix eigensystems at130 decimal digits',comparisonAllowance='1e-110',ensembles=ensembles,relative=relative),indent=2)+'\n')
print(json.dumps(dict(ensembles=len(ensembles),relative=len(relative),fixture=str(output))))
