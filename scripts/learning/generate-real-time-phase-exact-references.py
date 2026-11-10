#!/usr/bin/env python3
"""Independent dense Fraction/Gauss-Jordan references for the finite core.

Input JSON is a list of {id,input}. Uses the earlier independent derivation
checker, never production code or a production result. Full matrix hashes
cover every exact cell without repeating huge denominators in fixture files.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True
sys.set_int_max_str_digits(131072)
BASE=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('phase_derivations',BASE/'scripts/learning/verify-real-time-phase-derivations.py')
x=importlib.util.module_from_spec(spec);sys.modules[spec.name]=x;spec.loader.exec_module(x)

def digest(a):
    return hashlib.sha256(json.dumps(x.encode(a),separators=(',',':')).encode()).hexdigest()
def map_at(n,time,mass,omega):
    delta=time/n;v=omega*delta*delta
    return x.power([[1-v/2,delta/mass],[-mass*omega*delta*(1-v/4),1-v/2]],n)
rows=[]
for case in json.loads(Path(sys.argv[1]).read_text()):
    p=case['input'];mode=p['mode']
    if mode=='fold':continue
    n=p['intervals'];d=n-1;time,mass,hbar,omega=[F(p[k])for k in ['time','mass','hbar','omegaSquared']]
    delta=time/n;s=mass/delta;v=omega*delta*delta
    h=[[s*((2-v)if i==j else -1 if abs(i-j)==1 else 0)for j in range(d)]for i in range(d)]
    total=map_at(n,time,mass,omega)
    expected={'map':total,'inertia':list(x.inertia(h)[:3]),'determinant':x.determinant(h),'hessianHash':digest(h),
              'refinements':[map_at(n*k,time,mass,omega)for k in [1,2,4,8]]}
    if mode=='chain':
        a,b,damping=[F(p[k])for k in ['start','end','damping']]
        forcing=[F(0)]*d;forcing[0]+=s*a;forcing[-1]+=s*b
        constant=s*(a*a+b*b)/2-delta*mass*omega*(a*a+b*b)/4
        compatible,rank,q=x.solve(h,forcing);inv=x.inverse(h)
        expected.update(forcing=forcing,constant=constant,compatible=compatible,stationary=q if inv is not None else None,
            stationaryAction=constant-sum(fi*qi for fi,qi in zip(forcing,q))/2 if compatible else None,
            inverseHash=digest(inv) if inv is not None else None)
        if damping:
            c=[[x.C(damping if i==j else 0,-h[i][j])for j in range(d)]for i in range(d)]
            invc=x.inverse(c);mean=[-x.I*y for y in x.mv(invc,forcing)]
            exponent=x.C(0,constant/hbar)-sum(fi*qi for fi,qi in zip(forcing,x.mv(invc,forcing)))/(2*hbar)
            expected['regulator']={'inverseHash':digest(invc),'mean':mean,'determinant':x.determinant(c),'exponent':exponent}
        else:expected['regulator']=None
    else:
        q0,p0,chi,width=[F(p[k])for k in ['position','momentum','chirp','width']]
        variance=width if p['widthMode']=='fixed-variance' else hbar*width
        covariance=[[variance,chi*variance],[chi*variance,chi*chi*variance+hbar*hbar/(4*variance)]]
        final=x.mm(x.mm(total,covariance),x.transpose(total));center=x.mv(total,[q0,p0])
        z0=x.C(chi,hbar/(2*variance));a,b=total[0];c,d0=total[1]
        expected.update(variance=variance,initialCovariance=covariance,finalCovariance=final,center=center,
                        centralAction=(center[0]*center[1]-q0*p0)/2,finalWidth=(c+d0*z0)/(a+b*z0))
    rows.append({'id':case['id'],'input':p,'expected':x.encode(expected)})
    print(case['id'],flush=True)
    (BASE/'.runtime/learning/real-time-phase-manuscripts/exact-reference-progress.json').write_text(json.dumps({'completed':len(rows),'lastId':case['id']})+'\n')
result={'source':'Independent dense Fraction elimination,Gauss-Jordan,1x1/2x2congruence andmatrix multiplication;all exact cells hashed','cases':rows}
out=BASE/'scripts/fixtures/real-time-phase-exact-references.json';out.write_text(json.dumps(result,indent=2)+'\n')
print('wrote',out,len(rows),'cases',flush=True)
