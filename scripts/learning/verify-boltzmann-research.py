"""Independent full observation laws, rational Gibbs fluxes and Lyapunov solves."""
from math import exp,log
from fractions import Fraction as F
from itertools import product
import json

def near(a,b):assert abs(float(a)-float(b))<3e-10*max(1,abs(float(a)),abs(float(b))),(a,b)
calibrations=0
for eta,gap,pair,repeats in product([-.7,0,.7],[.5,1,2],[(.5,.5),(.5,1),(.5,2),(.5,4),(1,3)],[(1,1),(2,1)]):
    probs=[1/(1+exp(b*gap-eta)) for b in pair]
    obsbeta=[b for b,m in zip(pair,repeats) for _ in range(m)]
    obsprob=[p for p,m in zip(probs,repeats) for _ in range(m)]
    fisher=[[0.,0.],[0.,0.]];means=[0.,0.]
    for observations in product([0,1],repeat=sum(repeats)):
        mass=1.
        for y,p in zip(observations,obsprob):mass*=p if y else 1-p
        score=[sum(y-p for y,p in zip(observations,obsprob)),sum(-b*(y-p) for b,y,p in zip(obsbeta,observations,obsprob))]
        for i in range(2):
            means[i]+=mass*score[i]
            for j in range(2):fisher[i][j]+=mass*score[i]*score[j]
    near(means[0],0);near(means[1],0)
    determinant=fisher[0][0]*fisher[1][1]-fisher[0][1]**2
    expected=repeats[0]*repeats[1]*probs[0]*(1-probs[0])*probs[1]*(1-probs[1])*(pair[0]-pair[1])**2
    near(determinant,expected)
    if pair[0]!=pair[1]:
        logits=[log(p/(1-p)) for p in probs]
        recovered_gap=(logits[0]-logits[1])/(pair[1]-pair[0]);recovered_eta=logits[0]+pair[0]*recovered_gap
        near(recovered_gap,gap);near(recovered_eta,eta)
    # A single calibrated preparation cannot separate intercept and energy gap.
    alternative_gap=gap+.4;alternative_eta=eta+pair[0]*.4
    near(1/(1+exp(pair[0]*alternative_gap-alternative_eta)),probs[0])
    # Common energy and reference normalizations remain gauges at both temperatures.
    for b,p in zip(pair,probs):
        for c,scale in [(-2,.2),(0,1),(3,5)]:
            w0=scale*exp(-b*c);w1=scale*exp(eta-b*(gap+c));near(w1/(w0+w1),p)
    calibrations+=1
states=list(product([0,1],repeat=2));n=len(states)
def stationary(p,T):return [sum(p[i]*T[i][j] for i in range(n)) for j in range(n)]
kernels=0;biased=0
for raw,a,mode in product([[1,1,1,1],[1,2,3,4],[2,5,1,2]],[F(1,2),F(7,10),F(9,10),F(99,100)],[0,1]):
    p=[F(w,sum(raw)) for w in raw]
    selection=[[a,1-a] if (x[0]==0 if mode==0 else x[0]==x[1]) else [1-a,a] for x in states]
    coordinate=[]
    for coord in range(2):
        T=[[F(0) for _ in states] for _ in states]
        for i,x in enumerate(states):
            compatible=[j for j,y in enumerate(states) if y[1-coord]==x[1-coord]]
            mass=sum(p[j] for j in compatible)
            for j in compatible:T[i][j]=p[j]/mass
        assert stationary(p,T)==p
        coordinate.append(T)
    naive=[[F(0) for _ in states] for _ in states];corrected=[[F(0) for _ in states] for _ in states]
    for i,x in enumerate(states):
        for coord in range(2):
            for j,y in enumerate(states):
                proposal=selection[i][coord]*coordinate[coord][i][j]
                naive[i][j]+=proposal
                accept=min(F(1),selection[j][coord]/selection[i][coord])
                corrected[i][j]+=proposal*accept;corrected[i][i]+=proposal*(1-accept)
    assert all(sum(row)==1 for row in naive+corrected)
    assert stationary(p,corrected)==p
    assert all(p[i]*corrected[i][j]==p[j]*corrected[j][i] for i in range(n) for j in range(n))
    if stationary(p,naive)!=p:biased+=1
    if raw==[1,1,1,1] and mode==0:
        biasedlaw=[(1-a)/2,(1-a)/2,a/2,a/2];assert stationary(biasedlaw,naive)==biasedlaw
        initial=[F(1),F(0),F(0),F(0)];u=initial[:];v=initial[:];rate=min(a,1-a)
        for k in range(21):
            assert u[2]+u[3]==a*(1-F(1,2)**k)
            assert v[2]+v[3]==F(1,2)*(1-(1-rate)**k)
            u=stationary(u,naive);v=stationary(v,corrected)
    kernels+=1
# Solve a 3x3 linear system for a symmetric 2x2 covariance using rational elimination.
def multiply(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def transpose(A):return [list(row) for row in zip(*A)]
def inverse(A):
    d=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    return [[A[1][1]/d,-A[0][1]/d],[-A[1][0]/d,A[0][0]/d]]
def solve(A,b):
    a=[row[:]+[v] for row,v in zip(A,b)]
    for k in range(3):
        pivot=next(i for i in range(k,3) if a[i][k]);a[k],a[pivot]=a[pivot],a[k];v=a[k][k];a[k]=[x/v for x in a[k]]
        for i in range(3):
            if i!=k:
                v=a[i][k];a[i]=[x-v*y for x,y in zip(a[i],a[k])]
    return [row[-1] for row in a]
gaussians=0
for raw,step in product([[[1,0],[0,4]],[[F(5,2),F(-3,2)],[F(-3,2),F(5,2)]],[[2,1],[1,2]],[[3,1],[1,1]]],[F(1,100),F(1,20),F(1,10),F(1,5),F(9,20)]):
    P=[[F(x) for x in row] for row in raw];B=[[int(i==j)-step*P[i][j] for j in range(2)] for i in range(2)]
    basis=[[[F(1),F(0)],[F(0),F(0)]],[[F(0),F(1)],[F(1),F(0)]],[[F(0),F(0)],[F(0),F(1)]]]
    M=[]
    for i,j in [(0,0),(0,1),(1,1)]:
        M.append([A[i][j]-multiply(multiply(B,A),transpose(B))[i][j] for A in basis])
    entries=solve(M,[2*step,F(0),2*step]);S=[[entries[0],entries[1]],[entries[1],entries[2]]]
    PP=multiply(P,P);candidate=inverse([[P[i][j]-step*PP[i][j]/2 for j in range(2)] for i in range(2)])
    assert S==candidate
    residual=multiply(multiply(B,S),transpose(B));assert S==[[residual[i][j]+2*step*int(i==j) for j in range(2)] for i in range(2)]
    assert S[0][0]>0 and S[0][0]*S[1][1]>S[0][1]**2
    assert S!=inverse(P)
    gaussians+=1
# The stiff eigenmode at step=1/2 has multiplier -1 and unit added variance.
v=F(0)
for k in range(21):assert v==k;v=(-F(1))**2*v+1
exactsteps=0
for lam,step in product([1,4],[.05,.2,.45,.5,2,10]):
    decay=exp(-step*lam);noise=(1-decay**2)/lam
    near(decay**2/lam+noise,1/lam);assert noise>0;exactsteps+=1
print(json.dumps({'status':'passed','calibratedObservationPreparations':calibrations,'rationalSelectionKernels':kernels,'biasedUncorrectedKernels':biased,'exactLyapunovPreparations':gaussians,'boundaryVarianceSteps':21,'exactGaussianTransitions':exactsteps}))
