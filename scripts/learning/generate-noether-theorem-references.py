"""Independent SymPy jet prolongations and polynomial path integrals. No product imports."""
import json,random
from pathlib import Path
import sympy as s
rng=random.Random(20261104)
t,x,y,v,w,acc,bcc,eps=s.symbols('t x y v w acc bcc eps')
def D(e):return s.expand(s.diff(e,t)+v*s.diff(e,x)+w*s.diff(e,y)+acc*s.diff(e,v)+bcc*s.diff(e,w))
def coeffs(e):
 p=s.Poly(s.expand(e),t);return [str(p.nth(j)) for j in range(max(0,p.degree())+1)] if e!=0 else ['0']
def zero(e):return s.expand(e)==0
def at(e,T):return str(s.cancel(e.subs(t,T)))
point=[]
for i in range(240):
 p=[rng.randint(1,4),rng.randint(0,4),rng.randint(0,4),rng.randint(-4,4),rng.randint(-4,4),i%6,(i//6)%3,rng.randint(-4,4),rng.randint(0,40)]
 if i<72:
  p[1:5]=[0,0,0,0] if i<36 else [1,1,0,0]
 if i in [72,73,74]:p=[1,0,0,2,0,1,0,[2,0,-2][i-72],20]
 if i==75:p=[1,0,1,0,0,4,2,0,20]
 if i==76:p=[1,1,1,0,0,1,2,0,20]
 if i==77:p=[1,2,1,0,0,2,2,0,0]
 m,kx,ky,fc,dc,path,g,gc,tc=p;F=s.Rational(fc,2);H=s.Rational(dc,2);gamma=s.Rational(gc,2);T=s.Rational(tc,20)
 L0=m*(v*v+w*w)/2-kx*x*x/2-ky*y*y/2+F*x+H*y;G=gamma*t*x;L=L0+D(G)
 eta=[(s.Integer(1),s.Integer(0)),(s.Integer(0),s.Integer(1)),(-y,x)][g]
 # Differentiate an independently prolonged finite first-order point variation.
 variation=s.diff(L.subs({x:x+eps*eta[0],y:y+eps*eta[1],v:v+eps*D(eta[0]),w:w+eps*D(eta[1])},simultaneous=True),eps).subs(eps,0)
 B0=[F*t,H*t,s.Integer(0)][g];B=B0+s.diff(G,x)*eta[0]+s.diff(G,y)*eta[1]
 defect=s.expand(variation-D(B));momentum=[s.diff(L,v),s.diff(L,w)];Qcan=sum(momentum[j]*eta[j]for j in range(2));Q=s.expand(Qcan-B)
 paths=[(1+t,1-t),(1+t*t/2,1+t),(s.Integer(1),s.Integer(1)),(1+t,2+2*t),(1+t,s.Integer(0)),(s.Integer(0),s.Integer(0))];qx,qy=paths[path]
 subs={x:qx,y:qy,v:s.diff(qx,t),w:s.diff(qy,t),acc:s.diff(qx,t,2),bcc:s.diff(qy,t,2)}
 pathExpr=lambda e:s.expand(e.subs(subs,simultaneous=True))
 rr=[pathExpr(D(momentum[0])-s.diff(L,x)),pathExpr(D(momentum[1])-s.diff(L,y))];q=pathExpr(Q);d=pathExpr(defect);rterm=sum(rr[j]*pathExpr(eta[j])for j in range(2));qrate=s.diff(q,t);identity=s.expand(qrate-d-rterm)
 queries={'position':[at(qx,T),at(qy,T)],'velocity':[at(s.diff(qx,t),T),at(s.diff(qy,t),T)],'newtonResidual':[at(r,T)for r in rr]}
 for k,e in [('charge',q),('canonicalCharge',pathExpr(Qcan)),('boundary',pathExpr(B)),('baseBoundary',pathExpr(B0)),('gaugeBoundary',pathExpr(B-B0)),('defect',d),('chargeRate',qrate),('residualContraction',rterm),('identityResidual',identity),('lagrangian',pathExpr(L)),('lagrangianVariation',pathExpr(variation)),('boundaryRate',pathExpr(D(B)))]:queries[k]=at(e,T)
 point.append(dict(p=p,query=queries,flags=dict(globalQuasiSymmetry=zero(defect),strictBaseLagrangianSymmetry=zero(s.expand(variation-D(s.diff(G,x)*eta[0]+s.diff(G,y)*eta[1]))),strictGaugeLagrangianSymmetry=zero(variation),wholePathSatisfiesDynamics=all(zero(r)for r in rr),wholePathDefectZero=zero(d),wholePathChargeConstant=zero(qrate),wholePathCanonicalChargeConstant=zero(s.diff(pathExpr(Qcan),t))),polynomials=dict(charge=coeffs(q),canonicalCharge=coeffs(pathExpr(Qcan)),boundary=coeffs(pathExpr(B)),defect=coeffs(d),chargeRate=coeffs(qrate),residualContraction=coeffs(rterm),newtonResidual=[coeffs(r)for r in rr]),chargeChange=at(q-q.subs(t,0),s.Integer(2)),integratedDefect=str(s.integrate(d,(t,0,2))),integratedResidualContraction=str(s.integrate(rterm,(t,0,2)))))
 assert zero(identity)
boost=[]
for i in range(216):
 p=[rng.randint(1,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),rng.randint(-4,4),i%3,rng.randint(0,40)]
 if i<18:p[1:3]=[0,0]
 if i<6:p[7]=0
 if i==18:p=[2,0,0,2,4,0,2,0,40]
 if i==19:p=[1,0,2,0,0,2,2,0,40]
 if i==20:p=[1,0,0,0,0,0,2,2,0]
 m,f0c,f1c,x0c,v0c,gc,ec,trial,tc=p;f0,f1,x0,v0,gamma,e=[s.Rational(c,2)for c in [f0c,f1c,x0c,v0c,gc,ec]];T=s.Rational(tc,20)
 force=f0+f1*t;G=gamma*t*t*x/2;L0=m*v*v/2+force*x;L=L0+D(G);P=s.diff(L,v);H=P*v-L;E=m*v*v/2-force*x
 B0=m*x+s.integrate(t*force,t);Bb=B0+t*s.diff(G,x);Qb=s.expand(P*t-Bb);Bclock=s.diff(G,t);Qt=s.expand(-H-Bclock);Db=s.expand(t*s.diff(L,x)+P-D(Bb));Dt=s.expand(s.diff(L,t)-D(Bclock))
 q=x0+v0*t+f0*t*t/(2*m)+f1*t**3/(6*m)+[0,t*t/2,t**3/6][trial];subs={x:q,v:s.diff(q,t),acc:s.diff(q,t,2)};path=lambda z:s.expand(z.subs(subs,simultaneous=True));R=path(D(P)-s.diff(L,x));qb=path(Qb);qt=path(Qt);db=path(Db);dt=path(Dt);rb=R*t;rt=-R*s.diff(q,t)
 shiftedL=s.expand(L.subs({x:x+e*t,v:v+e},simultaneous=True));finite=s.expand(e*B0+e*e*m*t/2+G.subs(x,x+e*t)-G);Ld=path(shiftedL-L);fd=path(finite);Ad=s.integrate(Ld,t);Ad=s.expand(Ad-Ad.subs(t,0));Bd=s.expand(fd-fd.subs(t,0));shiftedE=E.subs({x:x+e*t,v:v+e},simultaneous=True)
 expr={'position':q,'velocity':s.diff(q,t),'acceleration':s.diff(q,t,2),'newtonResidual':R,'boostedPosition':q+e*t,'boostedVelocity':s.diff(q,t)+e,'canonicalMomentum':path(P),'boostCharge':qb,'boostCanonicalCharge':path(P*t),'boostBaseBoundary':path(B0),'boostGaugeBoundary':path(Bb-B0),'boostBoundary':path(Bb),'boostChargeRate':s.diff(qb,t),'boostDefect':db,'boostResidualTerm':rb,'clockCharge':qt,'clockCanonicalCharge':path(-H),'clockBoundary':path(Bclock),'clockChargeRate':s.diff(qt,t),'clockDefect':dt,'clockResidualTerm':rt,'physicalEnergy':path(E),'boostedPhysicalEnergy':path(shiftedE),'frameEnergyShift':path(shiftedE-E),'hamiltonian':path(H),'actionDifference':Ad,'finiteBoundary':fd,'finiteBoundaryChange':Bd,'finiteLagrangianDifference':Ld}
 boost.append(dict(p=p,query={k:at(s.expand(z),T)for k,z in expr.items()},polynomials={k:coeffs(s.expand(z))for k,z in [('position',q),('velocity',s.diff(q,t)),('residual',R),('boostCharge',qb),('clockCharge',qt),('boostChargeRate',s.diff(qb,t)),('clockChargeRate',s.diff(qt,t)),('boostDefect',db),('clockDefect',dt),('boostResidualTerm',rb),('clockResidualTerm',rt)]},flags=dict(globalClockQuasiSymmetry=zero(Dt),strictClockLagrangianSymmetry=zero(s.diff(L,t)),wholePathSatisfiesDynamics=zero(R),wholePathBoostChargeConstant=zero(s.diff(qb,t)),wholePathClockChargeConstant=zero(s.diff(qt,t)))))
 assert zero(Db) and zero(s.diff(qb,t)-db-rb) and zero(s.diff(qt,t)-dt-rt) and zero(Ad-Bd) and zero(path(shiftedL-L-D(finite)))
out=dict(method='SymPy partial jet differentiation, explicit prolongation, symbolic endpoint subtraction and exact path integration; no application imports',point=point,boost=boost)
Path('scripts/fixtures/noether-theorem-independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('Generated',len(point),'point and',len(boost),'boost/time independent cases.')
for name,rows in [('point',point),('boost',boost)]:print(name,{k:sum(r['flags'][k]for r in rows)for k in rows[0]['flags']})
