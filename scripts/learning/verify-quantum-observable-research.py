"""Independent matrix calibration, duality, nine-probe and repeatability checks."""
import itertools,json
import numpy as np
import sympy as S
# General Hermitian calibration uses direct centered-vector variance and eigensolvers.
calibration=0;max_bound_excess=0.
for d,a,e,k in itertools.product([2,3],range(5),range(4),range(6)):
    raw=np.array([[complex((i+2*j+a)%5-2,(3*i-j+a)%7-3)for j in range(d)]for i in range(d)])
    A=(raw+raw.conj().T)/2
    eraw=np.array([[complex((2*i+j+e)%5-2,(i-3*j+e)%5-2)for j in range(d)]for i in range(d)])
    E=(eraw+eraw.conj().T)/(2*(e+1));B=A+E;eta=max(abs(np.linalg.eigvalsh(E)))
    psi=np.array([complex((i+k)%3+1,(2*i+k)%5-2)for i in range(d)]);psi/=np.linalg.norm(psi)
    means=[np.vdot(psi,M@psi).real for M in [A,B]]
    spreads=[np.linalg.norm(M@psi-m*psi)for M,m in zip([A,B],means)]
    errors=[abs(means[1]-means[0])-eta,abs(spreads[1]-spreads[0])-eta,max(abs(np.linalg.eigvalsh(A)-np.linalg.eigvalsh(B)))-eta]
    assert max(errors)<1e-12;max_bound_excess=max(max_bound_excess,*errors)
    assert abs(spreads[1]**2-spreads[0]**2)<=eta*sum(spreads)+1e-12
    calibration+=1
# Exact symbolic C3 projectors, independent primal and dual update paths.
I=S.eye(3);basis=[I[:,i]for i in range(3)];hermitians=[]
for i in range(3):
    b=S.zeros(3);b[i,i]=1;hermitians.append(b)
for i,j in itertools.combinations(range(3),2):
    for value in [1,S.I]:
        b=S.zeros(3);b[i,j]=value;b[j,i]=S.conjugate(value);hermitians.append(b)
unit=lambda v:v/S.sqrt((v.conjugate().T*v)[0])
canonical=lambda M:M.applyfunc(S.expand)
project=lambda v:canonical(v*v.conjugate().T)
trace=lambda M:S.expand(S.trace(M))
zero=lambda M:all(S.expand(x)==0 for x in M)
probes=basis+[unit(basis[i]+c*basis[j])for i,j in itertools.combinations(range(3),2)for c in [1,S.I]]
frames=[S.Matrix([1,0,0]),S.Matrix([1,1,0]),S.Matrix([1,S.I,0]),S.Matrix([1,1,1]),S.Matrix([1,2,S.I]),S.Matrix([1,S.I,1+S.I])]
operator_cases=probe_cases=commuting_cases=0;max_witness_error=0.
for raw in frames:
    U=canonical(I-2*raw*raw.conjugate().T/(raw.conjugate().T*raw)[0]);P=project(U[:,0]);Qp=I-P;blocks=[P,Qp]
    dephase=lambda M:canonical(sum((p*M*p for p in blocks),S.zeros(3)))
    for rawB in hermitians:
        for B in [rawB,dephase(rawB)]:
            H=dephase(B)-B;allzero=True
            measured=[]
            for v in probes:
                rho=project(v);before=trace(B*rho);after=trace(B*dephase(rho));delta=S.expand(after-before)
                assert delta==trace(rho*H);allzero=allzero and delta==0;measured.append(delta);probe_cases+=1
            assert allzero==zero(H)==zero(P*B-B*P)
            if allzero:commuting_cases+=1
            recovered=S.zeros(3)
            for i in range(3):recovered[i,i]=measured[i]
            for index,(i,j) in enumerate(itertools.combinations(range(3),2)):
                avg=(measured[i]+measured[j])/2;real=measured[3+2*index]-avg;imag=avg-measured[4+2*index]
                recovered[i,j]=real+S.I*imag;recovered[j,i]=real-S.I*imag
            assert zero(recovered-H)
            h=np.array(H.evalf().tolist(),dtype=complex);values,vectors=np.linalg.eigh(h);idx=int(np.argmax(abs(values)));w=vectors[:,idx]
            residual=abs(abs(np.vdot(w,h@w))-max(abs(values)));max_witness_error=max(max_witness_error,float(residual));assert residual<1e-12
            operator_cases+=1
# Exact branch vectors, repeatability and the downstream phase law.
P=project(basis[0]);Qp=I-P;A=P+2*Qp;psi=unit(basis[1]+basis[2]);B=project(psi);rotations=0
for t in [S.Rational(i,j)for i in range(-4,5)for j in [1,2,3]]:
    c=(1-t*t)/(1+t*t);s=2*t/(1+t*t);U=S.diag(1,1,c+S.I*s);K=[P,U*Qp]
    assert zero(sum((k.conjugate().T*k for k in K),S.zeros(3))-I)
    for k,label in zip(K,[1,2]):assert zero(A*k-label*k)
    output=sum((k*project(psi)*k.conjugate().T for k in K),S.zeros(3));assert trace(output**2)==1
    assert trace(B*output)==1/(1+t*t);assert zero(A*B-B*A)
    for v in [*basis,psi,unit(basis[0]+S.I*basis[2])]:
        rho=project(v)
        assert trace(K[0]*rho*K[0].conjugate().T)==trace(P*rho)
        assert trace(K[1]*rho*K[1].conjugate().T)==trace(Qp*rho)
    rotations+=1
# An exact phase flip outside the finite half-angle chart.
U=S.diag(1,1,-1);assert trace(B*U*B*U.conjugate().T)==0
# Direct matrix check of the instrument change at exact degeneracy.
rho=S.Matrix([[S.Rational(1,2)]*2]*2);test=rho;Z=S.diag(1,-1);gap_cases=0
for eps in [S.Rational(-1,4),S.Rational(1,16),S.Rational(1,256),S.Rational(0)]:
    obs=eps*Z;assert trace(obs*rho)==0 and trace(obs**2*rho)==eps**2
    output=S.diag(*rho.diagonal())if eps else rho
    assert trace(test*output)==(S.Rational(1,2)if eps else 1)
    gap_cases+=1
print(json.dumps({'status':'passed','calibrationPreparations':calibration,'operatorCases':operator_cases,'probeComparisons':probe_cases,'commutingCases':commuting_cases,'repeatablePhasePreparations':rotations,'closingGapPreparations':gap_cases,'maxCalibrationBoundExcess':max_bound_excess,'maxWitnessResidual':max_witness_error},indent=2))
