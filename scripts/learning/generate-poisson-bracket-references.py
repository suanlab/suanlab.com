"""Independent symbolic brackets +60digit flow derivatives;no production imports."""
from pathlib import Path
from fractions import Fraction as R
import json
import sympy as s
import mpmath as mp
mp.mp.dps=60
q,p,m,k,nu,t,beta,a=s.symbols('q p m k nu t beta a', real=True, nonzero=True)
def pb(f,g,qs,ps):return s.expand(sum(s.diff(f,x)*s.diff(g,y)-s.diff(f,y)*s.diff(g,x) for x,y in zip(qs,ps)))
H=p*p/(2*m)+k*q*q/2
bases=[q*p,m*nu*q*q/2,p*p/(2*m*nu),H/nu]
Q,P=s.symbols('Q P',real=True)
F,G,C=bases[1],bases[2],bases[0]
assert s.simplify(pb(F,G,[q],[p])-C)==0
assert s.simplify(pb(F,pb(G,C,[q],[p]),[q],[p])+pb(G,pb(C,F,[q],[p]),[q],[p])+pb(C,pb(F,G,[q],[p]),[q],[p]))==0
qs=s.symbols('x y z',real=True);ps=s.symbols('px py pz',real=True);ks=s.symbols('kx ky kz',real=True)
L=[qs[1]*ps[2]-qs[2]*ps[1],qs[2]*ps[0]-qs[0]*ps[2],qs[0]*ps[1]-qs[1]*ps[0]]
H3=sum(ps[j]**2/(2*m)+ks[j]*qs[j]**2/2 for j in range(3))
LM=s.Matrix([[pb(u,v,qs,ps) for v in L] for u in L]);LH=s.Matrix([pb(u,H3,qs,ps) for u in L])
assert s.simplify(LM[0,1]-L[2])==0
fnL=s.lambdify((*qs,*ps,m,*ks),s.Matrix(L),'mpmath');fnLM=s.lambdify((*qs,*ps,m,*ks),LM,'mpmath');fnLH=s.lambdify((*qs,*ps,m,*ks),LH,'mpmath')
def st(mass,f,x0,v0,time):
 w=mp.pi*f/4
 return (mp.mpf(x0)+mp.mpf(v0)*time,mp.mpf(v0)) if f==0 else (mp.mpf(x0)*mp.cos(w*time)+mp.mpf(v0)*mp.sin(w*time)/w,-w*mp.mpf(x0)*mp.sin(w*time)+mp.mpf(v0)*mp.cos(w*time))
def fmt(x):return mp.nstr(x,55)
def conv(x):
 if isinstance(x,mp.matrix):return [[fmt(x[i,j]) for j in range(x.cols)]for i in range(x.rows)]
 if isinstance(x,(list,tuple)):return [conv(v)for v in x]
 return fmt(x)
observables=[]
for j in range(240):
 par=[1+j%4,j%5,-4+j%9,-4+(j*5)%9,1+(j*3)%4,j%4,-2+(j*7)%5,j%41]
 mass,f,qc,vc,scale,code,bc,tc=par;time=mp.mpf(tc)/20;w=mp.pi*f/4;nuv=w if f else mp.mpf(1);stiff=mass*w*w;pos,vel=st(mass,f,mp.mpf(qc)/2,mp.mpf(vc)/2,time);mom=mass*vel
 A=bases[code]+beta*t*H;B=pb(A,H,[q],[p]);transA=A.subs({q:Q/a,p:a*P},simultaneous=True);transH=H.subs({q:Q/a,p:a*P},simultaneous=True);transB=pb(transA,transH,[Q],[P]);badA=A.subs({q:Q/a,p:P/a},simultaneous=True);badH=H.subs({q:Q/a,p:P/a},simultaneous=True);badB=pb(badA,badH,[Q],[P]);args=(q,p,m,k,nu,t,beta,a)
 ev=lambda expr:s.lambdify(args,expr,'mpmath')(pos,mom,mass,stiff,nuv,time,bc,scale)
 def orbitA(u):
  xx,vv=st(mass,f,mp.mpf(qc)/2,mp.mpf(vc)/2,u);pp=mass*vv;en=pp*pp/(2*mass)+stiff*xx*xx/2;return [xx*pp,mass*nuv*xx*xx/2,pp*pp/(2*mass*nuv),en/nuv][code]+bc*u*en
 correct=s.lambdify((Q,P,m,k,nu,t,beta,a),transB,'mpmath')(scale*pos,mom/scale,mass,stiff,nuv,time,bc,scale);bad=s.lambdify((Q,P,m,k,nu,t,beta,a),badB,'mpmath')(scale*pos,scale*mom,mass,stiff,nuv,time,bc,scale)
 observables.append(dict(p=par,position=fmt(pos),momentum=fmt(mom),observable=fmt(ev(A)),gradientQ=fmt(ev(s.diff(A,q))),gradientP=fmt(ev(s.diff(A,p))),poisson=fmt(ev(B)),partialTime=fmt(ev(s.diff(A,t))),totalTime=fmt(mp.diff(orbitA,time)),canonicalBracket=fmt(correct),badNaiveBracket=fmt(bad)))
positions=[(1,0,0),(1,1,1),(1,-1,R(1,2)),(0,0,0),(-1,R(1,2),1)]
velocities=[(0,1,0),(1,0,1),(-1,1,R(1,2)),(0,0,0),(R(1,2),-1,1)]
mpv=lambda x:mp.mpf(x.numerator)/x.denominator if isinstance(x,R)else mp.mpf(x)
angular=[]
for j in range(180):
 par=[1+j%4,j%5,(j*3+1)%5,(j*7+2)%5,j%5,(j*3+1)%5,j%3,j%41]
 # Include full and partial rotational symmetry,which independent modular codes would miss.
 if j%6==0:par[1:4]=[2,2,2]
 if j%6==1:par[2:4]=[1,1]
 mass,*_=par;fs=par[1:4];pc,vc,axis,tc=par[4:];time=mp.mpf(tc)/20
 def orbit(u):
  states=[st(mass,fs[i],mpv(positions[pc][i]),mpv(velocities[vc][i]),u)for i in range(3)];return [v[0]for v in states],[mass*v[1]for v in states]
 pos,mom=orbit(time);stiffs=[mass*(mp.pi*f/4)**2 for f in fs];args=(*pos,*mom,mass,*stiffs)
 values=fnL(*args);matrix=fnLM(*args);torque=fnLH(*args);deriv=[mp.diff(lambda u:fnL(*orbit(u)[0],*orbit(u)[1],mass,*stiffs)[i],time)for i in range(3)]
 angular.append(dict(p=par,position=conv(pos),momentum=conv(mom),angularMomentum=conv(list(values)),bracketMatrix=conv(matrix),torque=conv(list(torque)),directDerivative=conv(deriv)))
exact=[]
for j in range(64):
 mass=1+j%4;qc=-4+j%9;vc=-4+(j*3)%9;scale=1+j%4;code=j%4;bc=-2+(j*7)%5;tc=j%41;time=R(tc,20);qq=R(qc,2)+R(vc,2)*time;pp=mass*R(vc,2);en=pp*pp/(2*mass);base=[qq*pp,mass*qq*qq/2,pp*pp/(2*mass),en][code];br=[pp*pp/mass,qq*pp,R(0),R(0)][code];aval=base+bc*time*en
 exact.append(dict(p=[mass,0,qc,vc,scale,code,bc,tc],position=str(qq),momentum=str(pp),energy=str(en),observable=str(aval),poisson=str(br),explicitTime=str(bc*en),totalTime=str(br+bc*en),badNaiveBracket=str(br/(scale*scale))))
out=dict(provenance='SymPy polynomial differentiation and canonical substitution;mpmath60digit analytic-flow derivatives;Fraction64free states. No production imports.',observables=observables,angular=angular,exactFree=exact)
Path('scripts/fixtures/poisson-brackets-independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('Generated240observable,180angular-momentum60digit references and64exact rational free states.')
