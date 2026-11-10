"""Independent symbolic geometry derivatives, NumPy solves and mpmath checks.
Run at repository root with sympy, numpy and mpmath installed.
No physical dynamics is integrated by these instantaneous-reference checks.
"""
from pathlib import Path
from fractions import Fraction as Q
import json,random,math
import sympy as s
import numpy as np
import mpmath as mp
rng=random.Random(20261104422);mp.mp.dps=50
r,theta=s.symbols('r theta',real=True);X=s.Matrix([r*s.cos(theta),r*s.sin(theta)]);q=s.Matrix([r,theta]);J=X.jacobian(q);H=[s.hessian(X[i],q)for i in range(2)]
polar_J=s.lambdify((r,theta),J,'numpy');polar_H=[s.lambdify((r,theta),h,'numpy')for h in H]
a,b,l1,l2=s.symbols('a b l1 l2',real=True);P=s.Matrix([l1*s.sin(a),-l1*s.cos(a),l1*s.sin(a)+l2*s.sin(b),-l1*s.cos(a)-l2*s.cos(b)]);JP=P.jacobian([a,b]);P_fn=s.lambdify((l1,l2,a,b),P,'numpy');JP_fn=s.lambdify((l1,l2,a,b),JP,'numpy');z=s.Matrix(s.symbols('x1 y1 x2 y2'));h=s.Matrix([z[0]**2+z[1]**2-l1*l1,(z[2]-z[0])**2+(z[3]-z[1])**2-l2*l2]);C_fn=s.lambdify((l1,l2,*z),h.jacobian(z),'numpy')
A=np.array([[1.,0.],[1.,1.]])
def polar(values):
 m,rc,ac,vr,w,fx,fy,dr,dt,ec=values;R=rc/2;th=ac*math.pi/12;rate=np.array([vr,w])/10;F=np.array([fx,fy])/10;delta=np.array([dr,dt])/10;eps=ec/100
 jac=np.asarray(polar_J(R,th),dtype=float);vel=jac@rate;G=jac.T@(m*jac);Qforce=jac.T@F;curv=np.array([rate@fn(R,th)@rate for fn in polar_H]);physical=F/m
 acc=np.linalg.solve(jac,physical-curv)if rc>0 else None;naive=jac@acc if acc is not None else None;virtual=jac@delta
 rp=Q(rc,2)+Q(ec,100)*Q(dr,10);valid=rp>=0;point=np.array([R*math.cos(th),R*math.sin(th)]);linear=point+eps*virtual;exact=np.array([float(rp)*math.cos(th+eps*delta[1]),float(rp)*math.sin(th+eps*delta[1])])if valid else None
 return dict(values=values,position=point.tolist(),jacobian=jac.tolist(),velocity=vel.tolist(),curvature=curv.tolist(),generalizedAcceleration=None if acc is None else acc.tolist(),physicalAcceleration=physical.tolist(),naiveAcceleration=None if naive is None else naive.tolist(),metric=G.tolist(),generalizedForce=Qforce.tolist(),kinetic=float(m*(vel@vel)/2),power=float(F@vel),virtualDisplacement=virtual.tolist(),virtualWork=float(F@virtual),probeValid=valid,probeRegular=rp>0,probeExact=None if exact is None else exact.tolist(),probeLinear=linear.tolist(),probeError=None if exact is None else float(np.linalg.norm(exact-linear)),probeWork=None if exact is None else float(F@(exact-point)))
def double(values):
 m1,m2,lc1,lc2,ac,bc,w1,w2,fx,fy,da,db,ec=values;L1=lc1/2;L2=lc2/2;aa=ac*math.pi/12;bb=bc*math.pi/12;rates=np.array([w1,w2])/10;delta=np.array([da,db])/10;eps=ec/100
 point=np.asarray(P_fn(L1,L2,aa,bb),dtype=float).ravel();jac=np.asarray(JP_fn(L1,L2,aa,bb),dtype=float);C=np.asarray(C_fn(L1,L2,*point),dtype=float);vel=jac@rates;W=np.diag([m1,m1,m2,m2]);M=jac.T@W@jac;Mrel=A.T@M@A;rel=np.linalg.solve(A,rates);F=np.array([0,-10*m1,fx/10,fy/10-10*m2]);Qabs=jac.T@F;Qrel=A.T@Qabs;virt=jac@delta
 exact=np.asarray(P_fn(L1,L2,aa+eps*delta[0],bb+eps*delta[1]),dtype=float).ravel();linear=point+eps*virt
 def constraints(p):return [float(p[0]**2+p[1]**2-L1*L1),float((p[2]-p[0])**2+(p[3]-p[1])**2-L2*L2)]
 return dict(values=values,position=point.tolist(),jacobian=jac.tolist(),velocities=vel.tolist(),metric=M.tolist(),metricRelative=Mrel.tolist(),eigenvalues=np.sort(np.linalg.svd(np.sqrt(W)@jac,compute_uv=False)**2).tolist(),eigenvaluesRelative=np.sort(np.linalg.svd(np.sqrt(W)@jac@A,compute_uv=False)**2).tolist(),determinant=float(np.linalg.det(M)),kinetic=float(vel@W@vel/2),potential=float(10*m1*point[1]+10*m2*point[3]),force=Qabs.tolist(),forceRelative=Qrel.tolist(),power=float(F@vel),relativeRates=rel.tolist(),constraintJacobian=C.tolist(),constraintVelocity=(C@vel).tolist(),virtualDisplacement=virt.tolist(),virtualWork=float(F@virt),probeExact=exact.tolist(),probeLinear=linear.tolist(),probeExactConstraints=constraints(exact),probeLinearConstraints=constraints(linear),probeError=float(np.linalg.norm(exact-linear)),probeWork=float(F@(exact-point)))
polar_inputs=[[m,r,a,*case,-10,7,75]for m in range(1,5)for r in range(7)for a in [0,3,6,12,24]for case in [[0,0,0,0],[10,20,0,0],[-7,3,-12,23]]];assert len(polar_inputs)==420
polar_inputs += [[rng.randint(1,4),rng.randint(0,6),rng.randint(-24,24),*[rng.randint(-20,20)for _ in range(2)],*[rng.randint(-40,40)for _ in range(2)],*[rng.randint(-10,10)for _ in range(2)],rng.randint(0,100)]for _ in range(480)]
double_inputs=[[m1,m2,l1,l2,a,b,*rates,30,0,10,-10,50]for m1 in [1,4]for m2 in [1,4]for l1,l2 in [(2,2),(1,6),(6,1)]for a,b in [(0,0),(0,6),(6,6),(-12,12)]for rates in [(0,0),(10,-10),(10,10)]];assert len(double_inputs)==144
double_inputs += [[rng.randint(1,4),rng.randint(1,4),rng.randint(1,6),rng.randint(1,6),rng.randint(-12,12),rng.randint(-12,12),*[rng.randint(-20,20)for _ in range(2)],*[rng.randint(-30,30)for _ in range(2)],*[rng.randint(-10,10)for _ in range(2)],rng.randint(0,100)]for _ in range(756)]
polars=[polar(p)for p in polar_inputs];doubles=[double(p)for p in double_inputs]
polar_diff=[]
for ref in [v for v in polars if v['values'][1]>0][::4][:180]:
 v=ref['values'];m,rc,ac,vr,w,fx,fy,dr,dt,ec=v;R=mp.mpf(rc)/2;th=mp.mpf(ac)*mp.pi/12;vr=mp.mpf(vr)/10;w=mp.mpf(w)/10;ddr,alpha=[mp.mpf(str(x))for x in ref['generalizedAcceleration']]
 def path(t):
  rr=R+vr*t+ddr*t*t/2;aa=th+w*t+alpha*t*t/2
  return [rr*mp.cos(aa),rr*mp.sin(aa)]
 vel=[float(mp.diff(lambda t:path(t)[j],0))for j in range(2)];acc=[float(mp.diff(lambda t:path(t)[j],0,2))for j in range(2)];polar_diff.append(dict(values=v,velocity=vel,acceleration=acc))
double_diff=[]
for ref in doubles[::5][:180]:
 v=ref['values'];m1,m2,lc1,lc2,ac,bc,w1,w2,fx,fy,da,db,ec=v;L1=mp.mpf(lc1)/2;L2=mp.mpf(lc2)/2;aa=mp.mpf(ac)*mp.pi/12;bb=mp.mpf(bc)*mp.pi/12
 def point(t,rate1,rate2):
  a=aa+mp.mpf(rate1)/10*t;b=bb+mp.mpf(rate2)/10*t;x=L1*mp.sin(a);y=-L1*mp.cos(a)
  return [x,y,x+L2*mp.sin(b),y-L2*mp.cos(b)]
 vel=[float(mp.diff(lambda t:point(t,w1,w2)[j],0))for j in range(4)];virt=[float(mp.diff(lambda e:point(e,da,db)[j],0))for j in range(4)];double_diff.append(dict(values=v,velocity=vel,variation=virt))
assert len(polar_diff)==len(double_diff)==180
out=dict(polar=polars,double=doubles,polarDifferential=polar_diff,doubleDifferential=double_diff)
Path('scripts/fixtures/generalized-coordinates-independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('Generated900polar symbolic/linear-solve references;900double symbolic/weighted-Gram/SVD references;180polar and180double independent50-digit differential checks. No time integration or reaction-force solve is claimed.')
