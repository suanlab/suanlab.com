"""Independent standard-library checks of the declared probe and four-cycle models.
No product imports. Exact fractions parameterize equilibria and graph constraints;
finite differences check potential derivatives, and row elimination checks rank.
"""
import argparse,json,math
from fractions import Fraction as F

B=[[-1,0,0,1],[1,-1,0,0],[0,1,-1,0],[0,0,1,-1]]

def root(eta,stable=True):
    if not 0 < eta < 4/27: raise ValueError('Two interior branches require 0<eta<4/27')
    lo,hi=(0.,1/3) if stable else (1/3,1.)
    for _ in range(100):
        mid=(lo+hi)/2;value=mid*(1-mid)**2
        if (value<eta)==stable:lo=mid
        else:hi=mid
    return (lo+hi)/2

def table_probe():
    return [[eta,root(eta),1/(1-root(eta))**2,(1-3*root(eta))/(1-root(eta))] for eta in [.005,.025,.075,.14]]

def rank(matrix):
    a=[[F(x) for x in row] for row in matrix];r=0
    for col in range(len(a[0])):
        pivot=next((i for i in range(r,len(a)) if a[i][col]),None)
        if pivot is None:continue
        a[r],a[pivot]=a[pivot],a[r];v=a[r][col];a[r]=[x/v for x in a[r]]
        for i in range(len(a)):
            if i!=r:
                v=a[i][col];a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==len(a):break
    return r

def mv(a,x):return [sum(v*w for v,w in zip(row,x)) for row in a]
def near(a,b,atol=2e-8,rtol=2e-6):
    assert math.isfinite(a) and abs(a-b)<=atol+rtol*abs(b),(a,b)

def verify():
    counts=dict(equilibriumBranches=0,potentialDerivativeChecks=0,localExtremumChecks=0,fieldToleranceChecks=0,smallProbeSequences=0,softTrapSequences=0,graphFamilies=0,allNodeSubsetBalances=0,augmentedRanks=0,noiseGains=0,regularizationIdentities=0)
    for stable in [True,False]:
        for i in range(1,31):
            # Independent exact rational locations, not the manuscript root algorithm.
            sf=F(i,96) if stable else F(1,3)+F(i,48)
            eta=float(sf*(1-sf)**2);s=float(sf)
            near(root(eta,stable),s,1e-13,1e-12)
            W=lambda t:t*t/2-eta/(1-t)
            h=2e-5*min(s,1-s);d=(W(s-2*h)-8*W(s-h)+8*W(s+h)-W(s+2*h))/(12*h)
            near(d,0,2e-9,0);counts['potentialDerivativeChecks']+=1
            curvature=(1-3*s)/(1-s)
            d2=(W(s+h)-2*W(s)+W(s-h))/h**2
            near(d2,curvature,3e-4,3e-4);counts['potentialDerivativeChecks']+=1
            delta=1e-4*min(s,1-s,abs(s-1/3))
            assert all((W(s+sign*delta)>W(s))==stable for sign in [-1,1]);counts['localExtremumChecks']+=1
            assert (curvature>0)==stable;counts['equilibriumBranches']+=1
    for tolerance in [.001,.01,.05,.2,.8,1.2]:
        st=1-1/math.sqrt(1+tolerance);threshold=st*(1-st)**2
        for factor in [.1,.5,1,1.01]:
            eta=threshold*factor
            if eta>=4/27:continue
            bias=1/(1-root(eta))**2-1
            assert (bias<=tolerance+1e-12)==(factor<=1);counts['fieldToleranceChecks']+=1
    errors=[]
    for eta in [1e-2,1e-3,1e-4,1e-5]:
        s=root(eta)
        errors.append((abs(s/eta-1),abs((1/(1-s)**2-1)/eta-2)))
    assert all(errors[i+1][j]<errors[i][j] for i in range(3) for j in range(2))
    assert errors[-1][0]<3e-5 and errors[-1][1]<1e-4
    counts['smallProbeSequences']=1
    for eta in [.02,.08,.14]:
        ratios=[]
        for q in [1e-9,1e-12,1e-15]:
            k,Q,R=9e9,1e-9,.1;spring=k*Q*q/(eta*R**3)
            actual_eta=k*Q*q/(spring*R**3);ratios.append(1/(1-root(actual_eta))**2)
        assert max(ratios)-min(ratios)<1e-12 and ratios[0]>1;counts['softTrapSequences']+=1
    assert rank(B)==3 and mv(B,[1]*4)==[0]*4
    b=[-1,0,0,1];jp=[1,1,1,0]
    for alpha in [F(i,4) for i in range(-8,9)]:
        j=[F(x)+alpha for x in jp];assert mv(B,j)==b;counts['graphFamilies']+=1
        norm=sum(x*x for x in j);assert norm==F(3,4)+4*(alpha+F(3,4))**2;counts['regularizationIdentities']+=1
        for mask in range(16):
            subset=[(mask>>i)&1 for i in range(4)]
            cut=[sum(subset[i]*B[i][e] for i in range(4)) for e in range(4)]
            assert sum(x*y for x,y in zip(cut,j))==sum(x*y for x,y in zip(subset,b));counts['allNodeSubsetBalances']+=1
    for h in [[1,0,0,0],[0,-1,0,0],[1,-1,0,0],[1,1,0,0],[1,1,1,1],[0,0,0,0]]:
        total=sum(h);assert rank(B+[h])==3+(total!=0);counts['augmentedRanks']+=1
        if total:
            for error in [F(-1,10),F(1,100),F(1,2)]:
                delta=[error/F(total)]*4;assert mv(B,delta)==[0]*4 and sum(a*d for a,d in zip(h,delta))==error
                assert sum(d*d for d in delta)==4*error**2/F(total)**2;counts['noiseGains']+=1
    return counts

def references():
    probe=[]
    for s in [F(1,100),F(1,20),F(1,10),F(1,5),F(3,10)]:
        eta=float(s*(1-s)**2)
        for R in [.05,.1,.2]:
            k,Q,spring=9e9,1e-9,1e-4;q=eta*spring*R**3/(k*Q)
            probe.append(dict(s=float(s),eta=eta,R=R,Q=Q,q=q,spring=spring,field=k*Q/(R*(1-float(s)))**2))
    return dict(probe=probe,graph=[dict(alpha=a,edges=[1+a,1+a,1+a,a],derivative=[-1,0,0,1]) for a in [-.75,0,.5,1]])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');p.add_argument('--references',action='store_true');a=p.parse_args()
    print(json.dumps(references() if a.references else verify(),sort_keys=True))
