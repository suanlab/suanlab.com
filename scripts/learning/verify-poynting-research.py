"""Independent power-only tensor inference and vacuum local-frame references.
Standard library only. Includes passive/indefinite, rank-deficient, null and zero cases.
"""
from fractions import Fraction as F
from itertools import product
import argparse,json,math

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def mv(A,x):return [dot(row,x) for row in A]
def mm(A,B):return [[dot(row,col) for col in zip(*B)] for row in A]
def tr(A):return list(map(list,zip(*A)))
def solve(A,b):
    rows=[list(map(F,row))+[F(v)] for row,v in zip(A,b)];n=len(rows)
    for j in range(n):
        available=[i for i in range(j,n) if rows[i][j]]
        if not available:raise ValueError('Power probes do not identify the symmetric tensor')
        k=available[0];rows[j],rows[k]=rows[k],rows[j];p=rows[j][j];rows[j]=[v/p for v in rows[j]]
        for i in range(n):
            if i!=j:
                p=rows[i][j];rows[i]=[a-p*b for a,b in zip(rows[i],rows[j])]
    return [row[-1] for row in rows]
def near(a,b,tol=2e-10):assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)),(a,b)
def conductivity(a,b,d,h):return [[a,b-h],[b+h,d]]
def power_row(E):x,y=E;return [x*x,2*x*y,y*y]
def rest_fraction(f,g):
    if f==0 and g==0:raise ValueError('No normalized frame at zero field')
    if f==0 or g==0:raise ValueError('A nonzero null field has no subluminal zero-flux frame')
    return (abs(f)-abs(g))/(abs(f)+abs(g))

def verify():
    c=dict(powerBlindCurrents=0,symmetricTensorReconstructions=0,rankDeficientProbeRejections=0,powerNoiseVariances=0,passivityClassifications=0,negativePowerWitnesses=0,axialCurrentCounterexamples=0,exactStressTransforms=0,zeroFluxFrames=0,electricMagneticRestClasses=0,nullFrameRejections=0,zeroFieldRejections=0,rapidityFieldChecks=0,boostCompositionChecks=0,nearNullChecks=0,nearNullDecimalChecks=0)
    vectors=[list(map(F,v)) for v in [(1,0),(0,1),(1,1),(1,-1),(2,3),(-2,1),(0,0)]]
    for a,b,d,h,E in product(map(F,[0,1,2]),map(F,[-1,0,1]),map(F,[0,2,3]),map(F,[-2,0,1,3]),vectors):
        J=mv(conductivity(a,b,d,h),E);expected=dot(power_row(E),[a,b,d])
        assert dot(J,E)==expected
        assert dot(mv([[0,-h],[h,0]],E),E)==0;c['powerBlindCurrents']+=1
    for a,b,d,delta in product(map(F,[1,2]),map(F,[-1,F(1,2),1]),map(F,[2,3]),map(F,[-1,F(1,100),F(1,10),1,2])):
        A=[power_row(E) for E in [[F(1),F(0)],[F(0),F(1)],[F(1),delta]]];p=mv(A,[a,b,d]);answer=solve(A,p)
        assert answer==[a,b,d] and (p[2]-p[0]-delta*delta*p[1])/(2*delta)==b;c['symmetricTensorReconstructions']+=1
        errors=[]
        for noise in product([F(-1,10),F(1,10)],repeat=3):
            noisy=[x+y for x,y in zip(p,noise)];errors.append(solve(A,noisy)[1]-b)
        assert sum(errors)==0 and sum(e*e for e in errors)/8==F(1,100)*(2+delta**4)/(4*delta*delta);c['powerNoiseVariances']+=1
        if delta==1:
            singular=[power_row(E) for E in [[F(1),F(0)],[F(0),F(1)],[F(1),F(0)]]]
            try:solve(singular,mv(singular,[a,b,d]))
            except ValueError:c['rankDeficientProbeRejections']+=1
            else:raise AssertionError('Dependent power design accepted')
    for a,b,d in product(map(F,[-1,0,1,2]),map(F,[-2,-1,0,1,2]),map(F,[-1,0,1,2])):
        psd=a>=0 and d>=0 and a*d>=b*b;c['passivityClassifications']+=1
        if psd:
            for E in vectors:assert dot(power_row(E),[a,b,d])>=0
        else:
            if a<0:E=[F(1),F(0)]
            elif d<0:E=[F(0),F(1)]
            elif a>0:E=[-b/a,F(1)]
            else:E=[-(d+1)/(2*b),F(1)]
            assert dot(power_row(E),[a,b,d])<0;c['negativePowerWitnesses']+=1
    for rate,acc,x,y,q in product(map(F,[0,F(1,2),1]),map(F,[0,F(1,4)]),map(F,[0,1]),map(F,[-1,F(1,2)]),map(F,[F(-1,2),0,F(1,4)])):
        E=[rate*y/2,-rate*x/2,F(0)];J=[-acc*y/2,acc*x/2,F(0)];added=q*(1+x*x-y*y)
        shifted=[J[0],J[1],added]
        assert dot(E,shifted)==dot(E,J) and shifted[2]-J[2]==added
        # q(x,y,t), independent of z: its only divergence contribution is d_z q=0.
        assert dot(E,[0,0,added])==0;c['axialCurrentCounterexamples']+=1
    amps=list(map(F,[-2,-1,F(-1,4),0,F(1,4),1,2]))
    for f,g,alpha in product(amps,amps,map(F,[F(-4,5),F(-3,5),0,F(3,5),F(4,5)])):
        u=f*f+g*g;s=f*f-g*g;gamma2=1/(1-alpha*alpha)
        up=gamma2*((1+alpha*alpha)*u-2*alpha*s);sp=gamma2*((1+alpha*alpha)*s-2*alpha*u)
        fp2=(1-alpha)/(1+alpha)*f*f;gp2=(1+alpha)/(1-alpha)*g*g
        assert up==fp2+gp2 and sp==fp2-gp2 and up*up-sp*sp==u*u-s*s==4*f*f*g*g
        E=f+g;D=f-g
        assert up==gamma2*((E-alpha*D)**2+(D-alpha*E)**2)/2
        assert sp==gamma2*(E-alpha*D)*(D-alpha*E);c['exactStressTransforms']+=1
    for f,g in product(amps,repeat=2):
        if f and g:
            alpha=rest_fraction(f,g);gamma2=1/(1-alpha*alpha);u=f*f+g*g;s=f*f-g*g
            assert abs(alpha)<1 and gamma2*((1+alpha*alpha)*s-2*alpha*u)==0
            assert gamma2*((1+alpha*alpha)*u-2*alpha*s)==2*abs(f*g)
            assert s/u==2*alpha/(1+alpha*alpha);c['zeroFluxFrames']+=1
            if f*g>0:assert f-g-alpha*(f+g)==0
            else:assert f+g-alpha*(f-g)==0
            c['electricMagneticRestClasses']+=1
        else:
            try:rest_fraction(f,g)
            except ValueError:
                if f==g==0:c['zeroFieldRejections']+=1
                else:c['nullFrameRejections']+=1
            else:raise AssertionError('Degenerate rest frame invented')
    for f,g,eta in product([-2.,-.25,0.,.25,1.],[-1.,0.,.25,2.],[-2.,-.5,0.,.5,2.]):
        a=math.tanh(eta);gamma=math.cosh(eta);E=f+g;D=f-g
        Ep=gamma*(E-a*D);Dp=gamma*(D-a*E)
        near((Ep+Dp)/2,math.exp(-eta)*f);near((Ep-Dp)/2,math.exp(eta)*g)
        L=[[gamma,-math.sinh(eta)],[-math.sinh(eta),gamma]];u=(E*E+D*D)/2;s=E*D
        T=mm(mm(L,[[u,s],[s,u]]),tr(L));near(T[0][0],(Ep*Ep+Dp*Dp)/2);near(T[0][1],Ep*Dp);c['rapidityFieldChecks']+=1
    for eta1,eta2 in product([-.75,0,.5],[0,.25,1]):
        A=lambda e:[[math.cosh(e),-math.sinh(e)],[-math.sinh(e),math.cosh(e)]]
        actual=mm(A(eta1),A(eta2));expected=A(eta1+eta2)
        for row,ref in zip(actual,expected):
            for a,b in zip(row,ref):near(a,b)
        c['boostCompositionChecks']+=1
    for r in [1e-2,1e-6,1e-12]:
        eta=.5*math.log(1/r);fp=math.exp(-eta);gp=math.exp(eta)*r;near(fp,gp,1e-14)
        near(fp*fp+gp*gp,2*r,1e-14);near(1-rest_fraction(1.,r),2*r/(1+r),1e-15);c['nearNullChecks']+=1
    from decimal import Decimal,localcontext
    with localcontext() as ctx:
        ctx.prec=200
        for p in [20,40,60]:
            r=Decimal(10)**(-p);a=(1-r)/(1+r);w=(1-r*r)/(1+r*r)
            assert a<1 and w<1 and abs((1-a)-2*r/(1+r))<Decimal('1e-95')
            assert float(a)==1.0 and w>0;c['nearNullDecimalChecks']+=1
    return c

def references():
    induction=[]
    for position,t in product([[1,0,0],[0,1,0],[1,-.5,.25],[0,0,0]],[0,.5,1]):
        x,y,_=position;rate=.5+.25*t;E=[rate*y/2,-rate*x/2,0];J=[-.25*y/2,.25*x/2,0]
        induction.append(dict(position=position,time=t,E=E,J=J,power=dot(E,J)))
    pulses=[]
    for beta,z,shift in product([0,.25,.5,.75,1],[-1,0,1],[0,.5,1]):
        f=beta*math.exp(-(z-shift)**2/2);g=(1-beta)*math.exp(-(z+shift)**2/2)
        pulses.append(dict(beta=beta,z=z,shift=shift,amplitude=1,f=f,g=g,u=f*f+g*g,s=f*f-g*g,rest=None if f*g==0 else rest_fraction(f,g)))
    for z in [-1,0,1]:pulses.append(dict(beta=.5,z=z,shift=.5,amplitude=0,f=0,g=0,u=0,s=0,rest=None))
    return dict(induction=induction,pulses=pulses)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');p.add_argument('--references',action='store_true');args=p.parse_args()
    print(json.dumps(references() if args.references else verify(),sort_keys=True))
