"""Independent labeled-state enumeration for the three finite research examples."""
from fractions import Fraction as F
from itertools import product
from collections import defaultdict
from math import comb, log
import json

def entropy(p): return -sum(float(x)*log(float(x)) for x in p if x)
def close(a,b): assert abs(float(a)-float(b))<1e-12,(a,b)
shells=0
for N in range(2,8):
    for NA in range(1,N):
        NB=N-NA
        for K in range(N+1):
            states=[x for x in product([0,1],repeat=N) if sum(x)==K]
            support=range(max(0,K-NB),min(K,NA)+1)
            pi={a:F(comb(NA,a)*comb(NB,K-a),comb(N,K)) for a in support}
            incoming=defaultdict(F)
            for x in states:
                row=defaultdict(F); row[x]+=F(1,2)
                for i in range(NA):
                    for j in range(NA,N):
                        y=list(x); y[i],y[j]=y[j],y[i]; row[tuple(y)]+=F(1,2*NA*NB)
                assert sum(row.values())==1
                a=sum(x[:NA]); macro=defaultdict(F)
                for y,p in row.items(): macro[sum(y[:NA])]+=p; incoming[y]+=p/len(states)
                birth=F((NA-a)*(K-a),2*NA*NB); death=F(a*(NB-K+a),2*NA*NB)
                for b in support:
                    assert macro[b]==(birth if b==a+1 else death if b==a-1 else 1-birth-death if b==a else 0)
                assert sum((b-a)*p for b,p in macro.items())==F(NA*K-N*a,2*NA*NB)
                if a+1 in pi: assert pi[a]*birth==pi[a+1]*F((a+1)*(NB-K+a+1),2*NA*NB)
            assert all(incoming[x]==F(1,len(states)) for x in states)
            shells+=1
# Independent microscopic propagation checks conditional uniformity and both entropy trends.
states=[x for x in product([0,1],repeat=8) if sum(x)==4]
p={x:F(1,5*comb(4,sum(x[:4]))**2) for x in states}
lastD=float('inf'); lastH=-float('inf'); lastCount=float('inf')
for t in range(11):
    counts=[sum(v for x,v in p.items() if sum(x[:4])==a) for a in range(5)]
    for x in states: assert p[x]==counts[sum(x[:4])]/comb(4,sum(x[:4]))**2
    H=entropy(p.values()); hc=entropy(counts); D=log(70)-H
    assert D<=lastD+1e-12 and H>=lastH-1e-12 and hc<=lastCount+1e-12
    assert sum(a*v for a,v in enumerate(counts))==2
    close(H,hc+sum(float(v)*log(comb(4,a)**2) for a,v in enumerate(counts)))
    lastD,lastH,lastCount=D,H,hc
    nxt=defaultdict(F)
    for x,v in p.items():
        nxt[x]+=v/2
        for i in range(4):
            for j in range(4,8):
                y=list(x); y[i],y[j]=y[j],y[i]; nxt[tuple(y)]+=v/32
    p=nxt
assert lastD>0
# One-step period-two counterexample and its lazy repair.
flip=[[F(0),F(1)],[F(1),F(0)]]
assert flip[0][1]*flip[1][0]==1
assert [(flip[0][j]+int(j==0))/2 for j in range(2)]==[F(1,2)]*2
memories=0
for r in [F(0),F(1,10),F(3,10),F(3,5),F(1)]:
    T=[[r/3+(1-r)*int(j==(i+1)%3) for j in range(3)] for i in range(3)]
    assert all(sum(T[i][j] for i in range(3))==1 for j in range(3))
    if r>0: assert all(p>0 for row in T for p in row)
    triple=defaultdict(F); group=[0,0,1]
    for i,j,k in product(range(3),repeat=3): triple[group[i],group[j],group[k]]+=T[i][j]*T[j][k]/3
    pair=defaultdict(F); middle=defaultdict(F)
    for (a,b,c),v in triple.items(): pair[a,b]+=v; middle[b]+=v
    q=pair[0,1]/middle[0]; after=triple[1,0,1]/pair[1,0]; actual=triple[1,0,1]/middle[1]
    assert q==F(1,2)-r/6
    assert after==2*(r/3)*(1-2*r/3)/(1-r/3)
    assert q-after==(1-r)**2/(2*(1-r/3))
    assert actual==2*(r/3)*(1-2*r/3)
    assert (pair[1,0]/middle[1])*q-actual==(1-r)**2/2
    cmi=sum(float(v)*log(float(v*middle[b]/(pair[a,b]*pair[b,c]))) for (a,b,c),v in triple.items() if v)
    if r<1: assert cmi>0 and T[0][2]!=T[1][2]
    else: close(cmi,0)
    memories+=1
sensors=0
for e in [F(1,10),F(1,4)]:
    previous=0
    for n in range(1,7):
        results={}
        for shared in [False,True]:
            joint=defaultdict(F)
            for x in [0,1]:
                for errors in product([0,1],repeat=n):
                    if shared:
                        mass=(e if errors[0] else 1-e) if len(set(errors))==1 else F(0)
                    else:
                        mass=F(1)
                        for bit in errors: mass*=e if bit else 1-e
                    joint[x,tuple(x^bit for bit in errors)]+=mass/2
            counts=defaultdict(F); outputs=defaultdict(F)
            for (x,y),v in joint.items(): counts[x,sum(y)]+=v; outputs[y]+=v
            I=0
            for (x,y),v in joint.items():
                if v: I+=float(v)*log(float(2*v/outputs[y]))
            Ic=0
            for (x,k),v in counts.items():
                if v: Ic+=float(v)*log(float(2*v/(counts[0,k]+counts[1,k])))
            close(I,Ic)
            err=sum(min(joint[0,y],joint[1,y]) for y in outputs)
            posterior=joint[1,(1,)*n]/outputs[(1,)*n]
            if shared:
                close(I,log(2)-entropy([e,1-e])); assert err==e and posterior==1-e
            else:
                assert posterior==(1-e)**n/((1-e)**n+e**n)
                for x,k in product([0,1],range(n+1)):
                    expected=F(comb(n,k),2)*(e if x==0 else 1-e)**k*(1-e if x==0 else e)**(n-k)
                    assert counts[x,k]==expected
                assert I>previous and I<=log(2); previous=I
                if n<=2: assert err==e
            for x in [0,1]:
                assert sum(v for (xx,y),v in joint.items() if xx==x and y[0]!=x)==e/2
            results[shared]=I
        if n>1: assert results[False]>results[True]
        sensors+=1
print(json.dumps({'status':'passed','enumeratedShells':shells,'microscopicEntropySteps':11,'coarseMemoryKernels':memories,'repeatedSensorPairs':sensors}))
