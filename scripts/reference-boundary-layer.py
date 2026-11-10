"""Independent signed-coordinate integrating-factor references; no TS calculator.
mpmath decimals are high precision cross-checks, not formal interval proofs.
"""
import json, random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
import sympy as sp
mp.mp.dps=200
S=2**256
examples=json.loads(Path('scripts/fixtures/boundary-layer-examples.json').read_text())
rng=random.Random(20261110)
cases=list(examples)
for i in range(20):
    drift=F(rng.randint(-10,10),rng.randint(1,13))
    cases.append({'id':f'seeded-{i:02}', 'input':{'epsilon':str(F(rng.randint(1,10),rng.randint(1,100))), 'drift':str(drift), 'forcing':str(F(rng.randint(-12,12),rng.randint(1,9))), 'left':str(F(rng.randint(-8,8),rng.randint(1,7))), 'right':str(F(rng.randint(-8,8),rng.randint(1,7))), 'probe':str(F(rng.randint(0,100),100)), 'panels':rng.choice([4,16,32,256])}})
for sign in ['','-']:
    for epsilon in ['16','1/999999999983']:
        cases.append({'id':f'extreme-small-drift-{sign or "positive"}-{epsilon.replace("/","-")}', 'input':{'epsilon':epsilon,'drift':sign+'1/999999999989','forcing':'16','left':'-16','right':'16','probe':'1/999999999989','panels':256}})
def m(q):
    q=F(q);return mp.mpf(q.numerator)/q.denominator
def cell(z):return str(int(mp.floor(z*S)))
rows=[]
for c in cases:
    v=c['input']; ep,a,force,A,B=[F(v[k]) for k in ['epsilon','drift','forcing','left','right']]
    e,alpha,ff,aa,bb=map(m,[ep,a,force,A,B])
    def exact(x):
        xx=m(x)
        if not a:return aa+(bb-aa)*xx+ff/(2*e)*xx*(xx-1)
        if x==0:return aa
        if x==1:return bb
        return aa+ff/alpha*xx+(bb-aa-ff/alpha)*mp.expm1(-alpha*xx/e)/mp.expm1(-alpha/e)
    def derivatives(x):
        xx=m(x)
        if not a:return (bb-aa+ff/e*(xx-mp.mpf('0.5')),ff/e)
        weight=(bb-aa-ff/alpha)*mp.exp(-alpha*xx/e)/mp.expm1(-alpha/e)
        return(ff/alpha-alpha/e*weight,(alpha/e)**2*weight)
    def asymptotic(x):
        y=x if a>0 else 1-x; near,far=(A,B) if a>0 else (B,A)
        slope=force/abs(a); mismatch=far-near-slope
        outer=m(far+slope*(y-1));decay=mp.exp(-m(abs(a)*y/ep))
        return(outer,m(far-slope)-m(mismatch)*decay,outer-m(mismatch)*decay)
    xs=[F(i,64) for i in range(65)];layer=[]
    if a:
        ys=[min(F(1),8*ep/abs(a))*F(i,64) for i in range(65)]
        layer=[y if a>0 else 1-y for y in ys]
    def sample(x):
        out={'x':str(x),'exact':cell(exact(x))}
        if a:out.update(zip(['outer','inner','composite'],map(cell,asymptotic(x))))
        return out
    probe=F(v['probe']);d1,d2=derivatives(probe)
    row={**c,'physicalSamples':[sample(x) for x in xs],'layerSamples':[sample(x) for x in layer], 'probe':{**sample(probe),'derivative':cell(d1),'curvature':cell(d2)}}
    if a:
        slope=force/abs(a);near,far=(A,B) if a>0 else (B,A);D=far-near-slope;t=abs(a)/ep
        row['mismatch']=str(D);row['delta']=str(1/t);row['uniformError']=cell(abs(m(D))*mp.exp(-m(t)))
        row['derivativeUniformError']=cell(abs(m(D))*m(t)/mp.expm1(m(t)))
    rows.append(row)
x,eps,a,f,A,B=sp.symbols('x eps a f A B',nonzero=True)
u=A+f*x/a+(B-A-f/a)*(sp.exp(-a*x/eps)-1)/(sp.exp(-a/eps)-1)
assert sp.simplify(eps*sp.diff(u,x,2)+a*sp.diff(u,x)-f)==0
assert sp.simplify(u.subs(x,0)-A)==0 and sp.simplify(u.subs(x,1)-B)==0
comp=B+f/a*(x-1)-(B-A-f/a)*sp.exp(-a*x/eps)
assert sp.simplify(eps*sp.diff(comp,x,2)+a*sp.diff(comp,x)-f)==0
output={'precisionDecimalDigits':200,'referenceQuantumBits':256,'formula':'Original signed-coordinate integrating-factor solution using expm1, compared with reflected-coordinate directed TS formula. Cells are high precision numerical comparisons, not formal interval certificates.','symbolicExactResidual':'0','symbolicExactBoundaryResiduals':['0','0'],'symbolicCompositeInteriorResidual':'0','cases':rows}
Path('scripts/fixtures/boundary-layer-reference.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'referenceCases':len(rows),'physicalSamples':sum(len(r['physicalSamples']) for r in rows),'layerSamples':sum(len(r['layerSamples']) for r in rows),'symbolicResidualsVerified':True}))
