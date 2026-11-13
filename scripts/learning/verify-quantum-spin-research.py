"""Independent rotation-score and finite-spin preparation checks."""
from fractions import Fraction as Q
from math import comb
import itertools,json,math
import numpy as np
from scipy.linalg import expm
pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])];I=np.eye(2)
raw=[(1,0),(0,1),(1,1),(1,1j),(1,-1j),(2+1j,3-2j),(1-2j,2+3j)]
states=[np.array(v,dtype=complex)/np.linalg.norm(v)for v in raw]
axes=[np.array(v,dtype=float)/np.linalg.norm(v)for v in [(1,0,0),(0,1,0),(0,0,1),(1,2,3),(0,3,4)]]
rotation_cases=regular_readouts=boundary_readouts=optimal_tests=0;max_score_residual=0.
for psi,a,angle in itertools.product(states,axes,[0.,.2,.8,1.6]):
    generator=sum((x*p/2 for x,p in zip(a,pauli)),np.zeros((2,2),complex));U=expm(-1j*angle*generator);v=U@psi;dv=-1j*generator@v;rho=np.outer(v,v.conj());drho=np.outer(dv,v.conj())+np.outer(v,dv.conj());sld=2*drho
    qfi=float(np.trace(rho@sld@sld).real);r=np.array([np.vdot(v,p@v).real for p in pauli]);expected_qfi=1-float(a@r)**2;assert abs(qfi-expected_qfi)<4e-14
    for n in axes:
        detector=(I+sum((x*p for x,p in zip(n,pauli)),np.zeros((2,2),complex)))/2;p=float(np.trace(rho@detector).real);dp=float(np.trace(drho@detector).real)
        geometric_p=(1+float(n@r))/2;geometric_dp=float(n@np.cross(a,r))/2;residual=max(abs(p-geometric_p),abs(dp-geometric_dp));max_score_residual=max(max_score_residual,residual);assert residual<2e-14
        if min(p,1-p)>1e-12:
            information=dp*dp/(p*(1-p));assert -2e-14<=information<=qfi+5e-13;regular_readouts+=1
        else:boundary_readouts+=1
    tangent=np.cross(a,r)
    if np.linalg.norm(tangent)>1e-10:
        n=tangent/np.linalg.norm(tangent);P=(I+sum((x*p for x,p in zip(n,pauli)),np.zeros((2,2),complex)))/2;p=float(np.trace(rho@P).real);dp=float(np.trace(drho@P).real)
        assert abs(p-.5)<2e-14 and abs(dp*dp/(p*(1-p))-qfi)<4e-14;optimal_tests+=1
    rotation_cases+=1
# Exact regular Bernoulli probabilities, local information and ambiguous pairs.
rational_cases=0
for t in [Q(i,j)for i in range(-4,5)for j in [1,2,3]]:
    c,s=(1-t*t)/(1+t*t),2*t/(1+t*t);p=(1+Q(3,5)*s)/2;dp=Q(3,10)*c;information=dp*dp/(p*(1-p));assert information==9*c*c/(25-9*s*s);assert Q(1,5)<=p<=Q(4,5);assert 0<=information<=1
    if t:
        other=1/t;other_s=2*other/(1+other*other);assert other_s==s
    rational_cases+=1
# Independently build normalized angular-momentum matrices from square-root ladders.
spin_cases=moment_cases=fringe_cases=0;max_fringe_residual=0.
for n in range(9):
    j=n/2;d=n+1;m=np.array([j-k for k in range(d)]);Jz=np.diag(m);plus=np.zeros((d,d),complex)
    for k in range(1,d):plus[k-1,k]=math.sqrt(k*(n-k+1))
    minus=plus.conj().T;Jx=(plus+minus)/2;Jy=(plus-minus)/(2j);coherent=np.array([math.sqrt(comb(n,k)/2**n)for k in range(d)],complex)
    # Raw polynomial coefficients use the inverse-binomial metric.
    raw=np.array([comb(n,k)for k in range(d)],float);physical=raw/np.sqrt(raw);physical/=np.linalg.norm(physical);assert np.linalg.norm(physical-coherent)<1e-14
    assert np.linalg.norm(Jx@coherent-j*coherent)<2e-14
    mean=sum((Q(n,2)-k)*Q(comb(n,k),2**n)for k in range(d))
    variance=sum((Q(n,2)-k)**2*Q(comb(n,k),2**n)for k in range(d))-mean*mean;assert mean==0 and variance==Q(n,4)
    vectors=[('coherent',coherent,float(n))]
    if n:
        cat=np.zeros(d,complex);cat[0]=cat[-1]=1/math.sqrt(2);vectors.append(('cat',cat,float(n*n)))
        if n>=2:assert max(abs(np.vdot(cat,J@cat))for J in [Jx,Jy,Jz])<1e-14
    for label,state,expected_qfi in vectors:
        dv=-1j*Jz@state;rho=np.outer(state,state.conj());drho=np.outer(dv,state.conj())+np.outer(state,dv.conj());qfi=np.trace(rho@(2*drho)@(2*drho)).real;assert abs(qfi-expected_qfi)<2e-13;moment_cases+=1
        for angle in [0.,.17,.43,.91,1.7]:
            U=expm(-1j*angle*Jz);out=U@state;overlap=np.vdot(state,out);expected=math.cos(angle/2)**n if label=='coherent' else math.cos(j*angle);residual=abs(overlap-expected);max_fringe_residual=max(max_fringe_residual,float(residual));assert residual<3e-14
            if label=='cat':
                alias=expm(-1j*(angle+math.pi/j)*Jz)@state;assert np.linalg.norm(alias+out)<3e-14
                p=abs(overlap)**2;dp=2*np.real(overlap.conjugate()*np.vdot(state,-1j*Jz@out))
                if 1e-10<p<1-1e-10:assert abs(dp*dp/(p*(1-p))-expected_qfi)<2e-11
            fringe_cases+=1
    U=expm(-2j*math.pi*Jz);assert np.linalg.norm(U-(-1)**n*np.eye(d))<2e-14
    spin_cases+=1
print(json.dumps({'status':'passed','rotationPreparations':rotation_cases,'regularReadouts':regular_readouts,'boundaryReadouts':boundary_readouts,'optimalTangentReadouts':optimal_tests,'rationalReadouts':rational_cases,'spinRepresentations':spin_cases,'stateMomentPreparations':moment_cases,'fringePreparations':fringe_cases,'maxScoreResidual':max_score_residual,'maxFringeResidual':max_fringe_residual},indent=2))
