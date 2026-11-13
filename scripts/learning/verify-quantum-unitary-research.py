"""Independent matrix-exponential checks of ray geometry, projections and splitting."""
import itertools,json,math
import numpy as np
from scipy.linalg import expm
X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.]);I=np.eye(2)
norm=lambda M:float(np.linalg.norm(M,2))
geometry=0;max_speed_residual=0.
for w,b,center,hbar in itertools.product([0.,.1,.25,.5,.8,1.],[-2.,-.3,.7,1.5],[-3.,0.,4.],[.5,1.,2.]):
    psi=np.array([math.sqrt(w),1j*math.sqrt(1-w)]);H=center*I+b*Z;mean=np.vdot(psi,H@psi).real;variance=np.vdot(psi,H@H@psi).real-mean*mean
    variance=max(0.,variance);sigma=2*abs(b)*math.sqrt(w*(1-w));assert abs(variance-sigma*sigma)<4e-14
    derivative=-1j*H@psi/hbar;v=(I-np.outer(psi,psi.conj()))@derivative;residual=abs(np.vdot(v,v).real-sigma*sigma/hbar**2);max_speed_residual=max(max_speed_residual,residual);assert residual<5e-14
    for phase in [.0,.2,.7,1.4,2.5]:
        t=phase*hbar/abs(b);out=expm(-1j*t*H/hbar)@psi;fidelity=abs(np.vdot(psi,out))**2
        assert abs(fidelity-(1-4*w*(1-w)*math.sin(b*t/hbar)**2))<2e-14
        distance=math.acos(min(1.,max(0.,math.sqrt(fidelity))));assert distance<=sigma*t/hbar+4e-8
        if sigma*t/hbar<=math.pi/2:assert fidelity+3e-14>=math.cos(sigma*t/hbar)**2
    geometry+=1
# Repeated measurements as branch products, not repeated display sampling.
zeno=0;psi=np.array([1.,1.])/math.sqrt(2);P=np.outer(psi,psi)
for total,b,hbar,n in itertools.product([.3,1.,2.],[-1.,.4,2.],[.5,1.],[1,2,4,16]):
    U=expm(-1j*b*total*Z/(n*hbar));branch=psi.copy()
    for _ in range(n):branch=P@U@branch
    probability=float(np.vdot(branch,branch).real);expected=math.cos(b*total/(n*hbar))**(2*n)
    assert abs(probability-expected)<3e-14;zeno+=1
# General dimensions and complex Hermitian actions; operator errors use SVD.
splitting=0;max_unitary_residual=0.;max_lie_excess=max_symmetric_excess=0.
for d,seed,scale in itertools.product([2,3],range(5),[.1,.5,1.]):
    raw=np.array([[complex((i+2*j+seed)%5-2,(3*i-j+seed)%5-2)for j in range(d)]for i in range(d)]);A=scale*(raw+raw.conj().T)/2
    raw=np.array([[complex((2*i-j+seed)%7-3,(i+3*j+seed)%5-2)for j in range(d)]for i in range(d)]);B=scale*(raw+raw.conj().T)/2
    target=expm(-1j*(A+B));C=norm(A@B-B@A);L=norm(A)+norm(B)
    for n in [1,2,4,8]:
        a=expm(-1j*A/n);b=expm(-1j*B/n);half=expm(-1j*A/(2*n));lie=np.linalg.matrix_power(b@a,n);symmetric=np.linalg.matrix_power(half@b@half,n)
        errors=[norm(lie-target),norm(symmetric-target)];bounds=[min(2.,C/(2*n)),min(2.,L**3*math.exp(L/n)/(3*n*n))]
        assert all(e<=bound+2e-13 for e,bound in zip(errors,bounds));max_lie_excess=max(max_lie_excess,errors[0]-bounds[0]);max_symmetric_excess=max(max_symmetric_excess,errors[1]-bounds[1])
        v=np.arange(1,d+1,dtype=complex);v/=np.linalg.norm(v);r=np.arange(d,0,-1,dtype=complex)+1j*np.arange(d);r/=np.linalg.norm(r);R=np.outer(r,r.conj())
        for U,error in zip([lie,symmetric],errors):
            residual=norm(U.conj().T@U-np.eye(d));max_unitary_residual=max(max_unitary_residual,residual);assert residual<2e-13
            pu=float(np.vdot(R@U@v,R@U@v).real);pv=float(np.vdot(R@target@v,R@target@v).real);assert abs(pu-pv)<=min(1.,2*error)+2e-13
        splitting+=1
# Commuting cases have exact endpoint identities, up to numerical evaluation.
for n in [1,2,4,16]:
    A=.4*X+.2*Y;B=2*A;assert norm(np.linalg.matrix_power(expm(-1j*B/n)@expm(-1j*A/n),n)-expm(-1j*(A+B)))<1e-13
# Independent Pauli table checks avoid the manuscript's SU(2) trace shortcut.
pauli=[];target=expm(-1j*(X+Z))
for n in [1,2,4,8]:
    a=expm(-1j*X/n);b=expm(-1j*Z/n);half=expm(-1j*X/(2*n));pauli.append({'n':n,'lie':norm(np.linalg.matrix_power(b@a,n)-target),'symmetric':norm(np.linalg.matrix_power(half@b@half,n)-target)})
print(json.dumps({'status':'passed','geometryPreparations':geometry,'zenoBranchPreparations':zeno,'splittingPreparations':splitting,'commutingSchedules':4,'pauliTable':pauli,'maxSpeedResidual':max_speed_residual,'maxUnitaryResidual':max_unitary_residual,'maxLieBoundExcess':max_lie_excess,'maxSymmetricBoundExcess':max_symmetric_excess},indent=2))
