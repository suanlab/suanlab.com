"""Independent SymPy differentiation, constraint inversion, Jacobi and reduction checks."""
import json,random
from pathlib import Path
import sympy as s
rng=random.Random(20261105)
r,vr,w,ell,beta,m,k,a,Om=s.symbols('r vr w ell beta m k a Om',nonzero=True)
U=k*r*r/2+a*r**4/4;L=m*(vr*vr+r*r*w*w)/2+beta*r*r*w/2-U
canonical=s.diff(L,w);omega=s.solve(canonical-ell,w)[0];Ls=s.simplify(L.subs(w,omega));R=s.simplify((L-ell*w).subs(w,omega));Ffull=s.simplify(s.diff(L,r).subs(w,omega));Fr=s.simplify(s.diff(R,r));Fn=s.simplify(s.diff(Ls,r));effective=s.simplify(m*vr*vr/2-R);E=s.simplify((m*(vr*vr+r*r*w*w)/2+U).subs(w,omega));alpha=s.diff(omega,r)*vr
Fd=s.diff(L,r).subs(w,Om);Pd=canonical.subs(w,Om);torque=s.diff(Pd,r)*vr;Ed=m*(vr*vr+r*r*Om*Om)/2+U;Jd=s.simplify(Ed-Om*Pd)
assert s.simplify(Fr-Ffull)==0 and s.simplify(E-effective-m*vr*vr/2)==0
assert s.simplify(Ffull-Fn-2*ell**2/(m*r**3))==0
closedD=lambda z:s.simplify(s.diff(z,r)*vr+s.diff(z,vr)*Fr/m)
drivenD=lambda z:s.simplify(s.diff(z,r)*vr+s.diff(z,vr)*Fd/m)
assert closedD(E)==0 and drivenD(Jd)==0 and s.simplify(drivenD(Ed)-torque*Om)==0
expr={'angularVelocity':omega,'angularAcceleration':alpha,'angularVelocityHessian':s.diff(L,w,2),'reducedVelocityHessian':s.diff(R,vr,2),'potential':U,'potentialDerivative':s.diff(U,r),'radialKinetic':m*vr*vr/2,'angularKinetic':ell*0+m*r*r*omega*omega/2,'mechanicalAngularMomentum':s.simplify(m*r*r*omega),'mechanicalAngularMomentumRate':closedD(m*r*r*omega),'canonicalAngularMomentumRate':closedD(canonical.subs(w,omega)),'reconstructedLagrangian':Ls,'routhian':R,'effectivePotential':effective,'reducedEnergy':E,'trueRadialForce':Fr,'naiveRadialForce':Fn,'radialForceGap':Fr-Fn,'routhForceSlope':s.diff(Fr,r),'minusRouthMomentumDerivative':-s.diff(R,ell),'drivenCanonicalMomentum':Pd,'drivenRadialForce':Fd,'drivenForceSlope':s.diff(Fd,r),'motorTorque':torque,'motorPower':torque*Om,'drivenEnergy':Ed,'drivenJacobi':Jd,'drivenEnergyRate':drivenD(Ed),'drivenJacobiRate':drivenD(Jd),'physicalEnergyRate':closedD(E)}
routh=[]
for i in range(220):
 p=[rng.randint(1,4),rng.randint(0,4),rng.randint(0,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(1,40),rng.randint(-4,4),rng.randint(-4,4)]
 if i==0:p=[1,1,0,2,0,10,2,2]
 if i==1:p=[1,0,0,0,2,10,2,0]
 if i<24:p[3]=0 if i%2 else p[3]
 vals={m:p[0],k:p[1],a:p[2],ell:s.Rational(p[3],2),beta:s.Rational(p[4],2),r:s.Rational(p[5],10),vr:s.Rational(p[6],2),Om:s.Rational(p[7],2)}
 # Angular Hessian derivative above is evaluated on the reconstructed angular rate as needed.
 out={name:str(s.cancel(e.subs(w,omega).subs(vals)))for name,e in expr.items()}
 routh.append(dict(p=p,query=out))
x,y,px,py,Fx,Fy,Rtarget,Mass=s.symbols('x y px py Fx Fy Rtarget Mass',real=True);z=s.Matrix([x,y,px,py]);J=s.Matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
pb=lambda f,g:(s.Matrix([s.diff(f,c)for c in z]).T*J*s.Matrix([s.diff(g,c)for c in z]))[0]
phi=s.Matrix([(x*x+y*y-Rtarget**2)/2,x*px+y*py]);G=phi.jacobian(z);C=G*J*G.T;Ci=s.simplify(C.inv());D=s.simplify(J-J*G.T*Ci*G*J);projector=s.simplify(s.eye(4)-J*G.T*Ci*G)
assert s.simplify(G*D)==s.zeros(2,4) and s.simplify(projector**2-projector)==s.zeros(4,4)
assert s.simplify(D-projector*J*projector.T)==s.zeros(4,4)
# Verify all independent coordinate triples before inserting any constraint values.
jacobi=[]
for i in range(4):
 for j in range(i+1,4):
  for h in range(j+1,4):
   v=s.factor(sum(D[i,l]*s.diff(D[j,h],z[l])+D[j,l]*s.diff(D[h,i],z[l])+D[h,l]*s.diff(D[i,j],z[l])for l in range(4)));assert v==0;jacobi.append(str(v))
H=(px*px+py*py)/(2*Mass)-Fx*x-Fy*y;gradH=s.Matrix([s.diff(H,c)for c in z]);flow=s.simplify(D*gradH);can=J*gradH;angular=x*py-y*px;gradAngular=s.Matrix([s.diff(angular,c)for c in z]);angRate=s.simplify((gradAngular.T*flow)[0]);assert s.simplify(angRate-(x*Fy-y*Fx))==0;assert s.simplify((gradH.T*flow)[0])==0
angleGrad=s.Matrix([-y/(x*x+y*y),x/(x*x+y*y),0,0]);assert s.simplify((angleGrad.T*D*gradAngular)[0])==1
matrix_strings=lambda A,vals:[[str(s.cancel(A[i,j].subs(vals)))for j in range(A.cols)]for i in range(A.rows)]
dirac=[]
for i in range(240):
 p=[rng.randint(1,4),rng.randint(1,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4)]
 if i<40:
  # Include exact positive-radius/tangent-momentum physical states in every cardinal direction.
  rr=1+(i//4)%2;sign=-1 if i%4>=2 else 1;p[1]=2*rr;p[2:6]=[2*rr*sign,0,0,2]if i%2==0 else[0,2*rr*sign,2,0]
 if p[2]==p[3]==0:p[2]=1
 vals={Mass:p[0],Rtarget:s.Rational(p[1],2),x:s.Rational(p[2],2),y:s.Rational(p[3],2),px:s.Rational(p[4],2),py:s.Rational(p[5],2),Fx:s.Rational(p[6],2),Fy:s.Rational(p[7],2)}
 phis=[str(v.subs(vals))for v in phi];flowvals=[str(s.cancel(v.subs(vals)))for v in flow];reaction=flow[2:4,0]-s.Matrix([Fx,Fy]);on=all(s.sympify(v)==0 for v in phis)
 dirac.append(dict(p=p,constraints=phis,constraintMatrix=matrix_strings(C,vals),constraintInverse=matrix_strings(Ci,vals),diracTensor=matrix_strings(D,vals),phaseProjector=matrix_strings(projector,vals),configurationProjector=matrix_strings(D[:2,2:],vals),flow=flowvals,canonicalFlow=[str(s.cancel(v.subs(vals)))for v in can],canonicalConstraintRates=[str(s.cancel(v.subs(vals)))for v in G*can],angularMomentum=str(angular.subs(vals)),angularMomentumRate=str(angRate.subs(vals)),energy=str(H.subs(vals)),onPhysicalConstraintSurface=on,physicalReactionForce=[str(s.cancel(v.subs(vals)))for v in reaction]if on else None))
Path('scripts/fixtures/constraints-reduction-independent.json').write_text(json.dumps(dict(method='Independent SymPy velocity Legendre solve and partial derivatives; arbitrary-state constraint inverse,Dirac construction and all four Jacobi triples; no product imports',routh=routh,dirac=dirac,jacobiResiduals=jacobi),indent=2)+'\n')
print('Generated',len(routh),'Routh and',len(dirac),'Dirac cases;four symbolic coordinate Jacobi triples passed.')
