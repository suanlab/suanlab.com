"""Independent dense TT-SVD and circuit derivative checks; no runtime calculator imports."""
import json,itertools
import numpy as np
from scipy.linalg import expm
import sympy as sp
rng=np.random.default_rng(241113)
residuals=[]
def near(a,b,tol=3e-11):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    residuals.append(err)
    assert err<tol,(err,a,b)
def sweep(psi,dims,budgets):
    n=len(dims);q=np.ones((1,1),complex);res=psi.reshape(1,-1);xs=[psi];losses=[];qs=[]
    for k in range(n-1):
        m=res.reshape(q.shape[1]*dims[k],-1)
        u,s,vh=np.linalg.svd(m,full_matrices=False);r=min(budgets[k],len(s))
        losses.append(float(np.sum(s[r:]**2)))
        q=np.kron(q,np.eye(dims[k]))@u[:,:r];res=s[:r,None]*vh[:r]
        near(q.conj().T@q,np.eye(r))
        near(res,(q.conj().T@psi.reshape(q.shape[0],-1)))
        xs.append((q@res).ravel());qs.append(q)
    return xs,losses,qs
states=orthogonal_pairs=rankchecks=localrotations=0
for n in range(2,7):
    for trial in range(12):
        dims=[2]*n;psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n);psi/=np.linalg.norm(psi)
        budgets=[1+(trial+k)%3 for k in range(n-1)]
        xs,losses,qs=sweep(psi,dims,budgets);b=xs[-1];e=float(np.sum(losses));err=psi-b
        near(np.vdot(err,err).real,e);near(np.vdot(b,b).real,1-e);near(np.vdot(psi,b),1-e)
        discarded=[xs[k]-xs[k+1]for k in range(n-1)]
        for j,k in itertools.combinations(range(n-1),2):near(np.vdot(discarded[j],discarded[k]),0);orthogonal_pairs+=1
        for k in range(n-1):
            near(np.vdot(discarded[k],b),0)
            original=np.linalg.svd(psi.reshape(2**(k+1),-1),compute_uv=False)
            tail=float(np.sum(original[budgets[k]:]**2));assert losses[k]<=tail+3e-12
            s=np.linalg.svd(b.reshape(2**(k+1),-1),compute_uv=False)
            assert np.linalg.norm(s[budgets[k]:])<1e-12;rankchecks+=1
            # The realized nested physical projectors, not only their coefficient norms.
            if k:
                pk=qs[k]@qs[k].conj().T
                pp=np.kron(qs[k-1]@qs[k-1].conj().T,np.eye(2))
                near(pk@pp,pk)
        bhat=b/np.linalg.norm(b);rho=np.outer(psi,psi.conj());sigma=np.outer(bhat,bhat.conj())
        traceD=np.sum(np.abs(np.linalg.eigvalsh(rho-sigma)))/2
        near(traceD**2,e);near(np.linalg.norm(psi-bhat)**2,2*(1-np.sqrt(1-e)))
        op=rng.normal(size=(2**n,2**n))+1j*rng.normal(size=(2**n,2**n));op=(op+op.conj().T)/2
        delta=abs(np.vdot(psi,op@psi)-np.vdot(bhat,op@bhat))
        assert delta<=2*np.linalg.norm(op,2)*np.sqrt(e)+1e-11
        # Separate local unitary transformations preserve original-cut Schmidt spectra.
        local=np.ones((1,1),complex)
        for _ in range(n):
            z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2));u,_=np.linalg.qr(z);local=np.kron(local,u)
        transformed=local@psi
        for k in range(1,n):near(np.linalg.svd(transformed.reshape(2**k,-1),compute_uv=False),np.linalg.svd(psi.reshape(2**k,-1),compute_uv=False));localrotations+=1
        states+=1
examples=[]
for a,b,c,d in [(3,2,1,1),(4,2,1,1),(3,2,2,1),(2,1,1,0)]:
    psi=np.zeros(8,complex);psi[[0,3,5,6]]=[a,b,c,d];psi/=np.linalg.norm(psi)
    xs,loss,qs=sweep(psi,[2]*3,[1,1]);N=a*a+b*b+c*c+d*d
    near(xs[-1],np.eye(8)[0]*a/np.sqrt(N));near(loss,[(c*c+d*d)/N,b*b/N]);examples.append(loss)
# A feasible product competitor strictly improves the rank-one sweep in one example.
v=np.zeros(8);v[[0,3,5,6]]=[3,2,2,1];v/=np.linalg.norm(v)
product=np.kron(np.kron(np.array([2,1])/np.sqrt(5),np.array([2,1])/np.sqrt(5)),np.array([2,1])/np.sqrt(5))
fidelity=float(np.vdot(product,v)**2);near(fidelity,578/1125)
competitor=np.vdot(product,v)*product;near(np.linalg.norm(v-competitor)**2,547/1125)
assert sp.Rational(547,1125)<sp.Rational(1,2)
# Degenerate choices may differ, but any SVD output still has orthogonal discarded pieces.
degenerate=0
for psi,n in [(np.array([1,0,0,1])/np.sqrt(2),2),(np.eye(8)[0]/np.sqrt(2)+np.eye(8)[7]/np.sqrt(2),3)]:
    xs,loss,_=sweep(psi,[2]*n,[1]*(n-1));near(np.linalg.norm(psi-xs[-1])**2,sum(loss));degenerate+=1
I=np.eye(2);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);CX=np.eye(4)[[0,1,3,2]]
ket00=np.eye(4)[0];G=np.kron(Y/2,I);Gphi=np.kron(I,Y/2);O=np.kron(I,Z)
def rotation(t):return expm(-1j*t*Y/2)
def circuit(theta,phi):return np.kron(I,rotation(phi))@CX@np.kron(rotation(theta),I)@ket00
def value(theta,phi):
    v=circuit(theta,phi);return float(np.vdot(v,O@v).real)
circuits=0
for theta in np.linspace(-2.2,3.1,13):
    for phi in [-1.2,0,.7,2.0]:
        v=circuit(theta,phi)
        dt=np.kron(I,rotation(phi))@CX@(-1j*G)@np.kron(rotation(theta),I)@ket00
        dp=-1j*Gphi@v
        gradients=[2*np.vdot(dt,O@v).real,2*np.vdot(dp,O@v).real]
        exact=[-np.sin(theta)*np.cos(phi),-np.cos(theta)*np.sin(phi)]
        near(value(theta,phi),np.cos(theta)*np.cos(phi));near(gradients,exact)
        near((value(theta+np.pi/2,phi)-value(theta-np.pi/2,phi))/2,exact[0])
        near((value(theta,phi+np.pi/2)-value(theta,phi-np.pi/2))/2,exact[1]);circuits+=1
# Verify the generator rule for random preparations and readouts, not only the chosen observable.
general_two=controlled=0
Gc=np.kron(np.diag([0,1]),Y/2);near(np.linalg.eigvalsh(Gc),[-.5,0,0,.5])
for trial in range(24):
    v=rng.normal(size=4)+1j*rng.normal(size=4);v/=np.linalg.norm(v)
    a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));a=(a+a.conj().T)/2
    for gen in [G,Gc]:
        t=-2+trial/6
        def f(t):
            w=expm(-1j*t*gen)@v;return float(np.vdot(w,a@w).real)
        w=expm(-1j*t*gen)@v
        derivative=float((1j*np.vdot(w,(gen@a-a@gen)@w)).real)
        if gen is G:
            near(derivative,(f(t+np.pi/2)-f(t-np.pi/2))/2);general_two+=1
        else:
            near(derivative,(f(t+np.pi/2)-f(t-np.pi/2))/2+(1-np.sqrt(2))/4*(f(t+np.pi)-f(t-np.pi)));controlled+=1
control=np.kron(np.array([1,1])/np.sqrt(2),np.array([1,0]));Oc=np.kron(X,I)
for t in np.linspace(-2*np.pi,2*np.pi,17):
    v=expm(-1j*t*Gc)@control;near(np.vdot(v,Oc@v),np.cos(t/2))
# Enumerate independent binomial outcomes exactly at small shot counts.
from fractions import Fraction as F
from math import comb
variancecases=0
for pp,pm in [(F(1,2),F(1,2)),(F(1,5),F(3,4)),(F(1),F(0)),(F(2,3),F(1,3))]:
    mp,mm=3,4;mean=F(0);second=F(0)
    for kp in range(mp+1):
        for km in range(mm+1):
            prob=comb(mp,kp)*pp**kp*(1-pp)**(mp-kp)*comb(mm,km)*pm**km*(1-pm)**(mm-km)
            g=(F(2*kp,mp)-F(2*km,mm))/2
            mean+=prob*g;second+=prob*g*g
    fp,fm=2*pp-1,2*pm-1
    assert mean==(fp-fm)/2
    assert second-mean*mean==(1-fp*fp)/(4*mp)+(1-fm*fm)/(4*mm);variancecases+=1
# Symbolic identities and the two cautionary examples.
t=sp.symbols('t',real=True);f=sp.cos(t/2)
assert sp.simplify(sp.diff(f,t)-(f.subs(t,t+sp.pi/2)-f.subs(t,t-sp.pi/2))/2-(1-sp.sqrt(2))/4*(f.subs(t,t+sp.pi)-f.subs(t,t-sp.pi)))==0
assert sp.diff(sp.cos(t)**2,t).subs(t,sp.pi/4)==-1
assert sp.simplify((sp.cos(t+sp.pi/2)**2-sp.cos(t-sp.pi/2)**2)/2)==0
print(json.dumps(dict(status='passed',sequentialStates=states,orthogonalDiscardPairs=orthogonal_pairs,bondRankChecks=rankchecks,localUnitaryCutComparisons=localrotations,analyticSweepExamples=len(examples),degenerateSweeps=degenerate,circuitPreparations=circuits,generalTwoEigenvalueCases=general_two,controlledFourShiftCases=controlled,controlledInterferenceCases=17,exactBinomialVarianceCases=variancecases,strictlyBetterProductCompetitor=True,symbolicSharedParameterCounterexample=True,maxResidual=max(residuals))))
