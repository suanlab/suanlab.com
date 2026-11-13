"""Independent finite-state, batch-count and exact geometric-series checks."""
from fractions import Fraction as F
from math import exp,log,factorial,cosh,tanh
from itertools import product
from collections import defaultdict
from decimal import Decimal,localcontext
import json

def near(a,b): assert abs(float(a)-float(b))<2e-10*max(1,abs(float(a)),abs(float(b))),(a,b)
finite=0
for E,bs,bt,c in product([[0,1,2],[-1,.5,3],[0,0,2]],[.25,1,2],[.125,.5,3],[-3,0,5]):
    Zs=sum(exp(-bs*e) for e in E);Zt=sum(exp(-bt*e) for e in E)
    q=[exp(-bs*e)/Zs for e in E];p=[exp(-bt*e)/Zt for e in E]
    W=[exp(-(bt-bs)*e) for e in E];R=sum(qi*w for qi,w in zip(q,W))
    near(R,Zt/Zs)
    variance=sum(qi*(w-R)**2 for qi,w in zip(q,W))
    chi=sum((pi-qi)**2/qi for pi,qi in zip(p,q))
    near(variance/R**2,chi)
    near(chi+1,Zs*sum(exp(-(2*bt-bs)*e) for e in E)/Zt**2)
    shifted=[exp(-(bt-bs)*(e+c)) for e in E];shiftR=sum(qi*w for qi,w in zip(q,shifted))
    near(shiftR/R,exp(-(bt-bs)*c))
    near(sum(qi*(w-shiftR)**2 for qi,w in zip(q,shifted))/shiftR**2,chi)
    finite+=1
batches=0
q=[F(4,7),F(2,7),F(1,7)]
for power,M in product(range(4),[1,2,4,8]):
    W=[F(2)**((1-power)*e) for e in range(3)];R=sum(qi*w for qi,w in zip(q,W));v=sum(qi*(w-R)**2 for qi,w in zip(q,W))
    mean=second=F(0);meanlog=0.;mass=F(0)
    for n0 in range(M+1):
        for n1 in range(M-n0+1):
            ns=[n0,n1,M-n0-n1]
            probability=F(factorial(M),factorial(ns[0])*factorial(ns[1])*factorial(ns[2]))
            for qi,n in zip(q,ns):probability*=qi**n
            estimate=sum(n*w for n,w in zip(ns,W))/M
            mass+=probability;mean+=probability*estimate;second+=probability*estimate**2
            meanlog+=float(probability)*log(float(estimate))
    assert mass==1 and mean==R and second-mean**2==v/M
    gap=log(float(R))-meanlog
    assert gap>=-1e-12
    if power!=1:assert gap>0
    else:near(gap,0)
    batches+=1
spins=0
for b,J in product([0,.25,.5,1,2],[.5,1,2]):
    beta=b/J;states=list(product([-1,1],repeat=3));E={s:-J*(s[0]*s[1]+s[1]*s[2]) for s in states};weight={s:exp(-beta*E[s]) for s in states};Z=sum(weight.values());p={s:w/Z for s,w in weight.items()}
    group=defaultdict(list)
    for s in states:group[s[0],s[2]].append(s)
    endpoint={g:sum(p[s] for s in sub) for g,sub in group.items()}
    cond={g:sum(p[s]*E[s] for s in sub)/endpoint[g] for g,sub in group.items()}
    hidden=sum(sum(p[s]*(E[s]-cond[g])**2 for s in sub) for g,sub in group.items())
    U=sum(p[s]*E[s] for s in states);total=sum(p[s]*(E[s]-U)**2 for s in states)
    between=sum(endpoint[g]*(cond[g]-U)**2 for g in group)
    k=.5*log(cosh(2*b));a=log(2)+k
    for g,sub in group.items():
        near(sum(weight[s] for s in sub),exp(a+k*g[0]*g[1]))
        near(cond[g],-J*sum(g)*tanh(b*sum(g)))
    near(Z,8*cosh(b)**2);near(sum(v*g[0]*g[1] for g,v in endpoint.items()),tanh(b)**2)
    same=sum(v for g,v in endpoint.items() if g[0]==g[1]);opposite=1-same
    near(between,(2*J*tanh(2*b))**2*same*opposite)
    near(hidden,4*J*J*same/cosh(2*b)**2)
    near(total,2*J*J/cosh(b)**2);near(total,between+hidden)
    assert hidden>0 and between>=0
    spins+=1
geometric=0
for a,b in product([F(1,4),F(1,2),F(3,4)],repeat=2):
    ratio=b*b/a;prefactor=(1-b)**2/(1-a)
    last=F(0)
    for L in [0,1,5,20,80]:
        first=second=F(0)
        for n in range(L+1):
            qn=(1-a)*a**n;pn=(1-b)*b**n;w=pn/qn
            first+=qn*w;second+=qn*w*w
        assert first==1-b**(L+1)
        formula=prefactor*(L+1) if ratio==1 else prefactor*(1-ratio**(L+1))/(1-ratio)
        assert second==formula and second>=last;last=second
        if ratio<1:assert second<prefactor/(1-ratio)
        else:assert second>=prefactor*(L+1)
    kl=log(float((1-b)/(1-a)))+float(b/(1-b))*log(float(b/a))
    assert kl>=-1e-12
    if ratio<1:assert prefactor/(1-ratio)>=1
    geometric+=1
# Independent 60-digit evaluation of the four published inverse-temperature cases.
with localcontext() as ctx:
    ctx.prec=60
    one=Decimal(1)
    for t in [Decimal('.75'),Decimal('.5'),Decimal('.4'),Decimal('.25')]:
        s=one;pref=(1-(-t).exp())**2/(1-(-s).exp());r=(s-2*t).exp()
        for L in [5,20]:
            direct=sum(pref*r**n for n in range(L+1))
            fast=pref*(L+1) if r==1 else pref*(1-r**(L+1))/(1-r)
            near(direct,fast)
print(json.dumps({'status':'passed','finiteReweightingPreparations':finite,'exactIndependentBatchLaws':batches,'hiddenSpinPreparations':spins,'exactGeometricTailPairs':geometric,'decimalPublishedTailCases':4}))
