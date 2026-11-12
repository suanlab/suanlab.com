"""Independent lumped electromechanics and finite-mode observation references.
Standard library only; preserves uncoupled, lossless, critical and blind cases.
"""
from fractions import Fraction as F
from itertools import product
import math,json,argparse

def near(a,b,tol=3e-9):assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)),(a,b)
def mm(A,B):return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]
def mv(A,x):return [sum(a*b for a,b in zip(row,x)) for row in A]
def expm(A,t):
    norm=max(sum(abs(x*t) for x in row) for row in A)
    scaling=max(0,math.ceil(math.log2(max(1,norm/.5))))
    X=[[x*t/2**scaling for x in row] for row in A];T=[[1.,0.],[0.,1.]];result=[r[:] for r in T]
    for n in range(1,40):
        T=[[x/n for x in row] for row in mm(T,X)]
        result=[[a+b for a,b in zip(row,other)] for row,other in zip(result,T)]
    for _ in range(scaling):result=mm(result,result)
    return result

def free(zeta,tau,sign=1):
    if zeta<1:
        d=math.sqrt(1-zeta*zeta);S=math.sin(d*tau)/d;C=math.cos(d*tau)
        return math.exp(-zeta*tau)*(C+zeta*S),-sign*math.exp(-zeta*tau)*S
    if zeta==1:return math.exp(-tau)*(1+tau),-sign*tau*math.exp(-tau)
    d=math.sqrt(zeta*zeta-1);fast=math.exp((-zeta-d)*tau);slow=math.exp((-zeta+d)*tau)
    S=(slow-fast)/(2*d)
    return (slow+fast)/2+zeta*S,-sign*S

def rk4(A,t,n=2000):
    x=[1.,0.];h=t/n
    for _ in range(n):
        k1=mv(A,x);k2=mv(A,[a+h*b/2 for a,b in zip(x,k1)]);k3=mv(A,[a+h*b/2 for a,b in zip(x,k2)]);k4=mv(A,[a+h*b for a,b in zip(x,k3)])
        x=[a+h*(b+2*c+2*d+e)/6 for a,b,c,d,e in zip(x,k1,k2,k3,k4)]
    return x

def simpson(f,t,n=4096):
    h=t/n
    return h/3*(f(0)+f(t)+sum((4 if j%2 else 2)*f(j*h) for j in range(1,n)))

def invert(a0,y,c,s):
    if s==0:raise ValueError('Initial magnetic amplitude is unobservable')
    return (c*a0-y)/s

def fit_known_a(a0,observations):
    D=sum(s*s for c,s,y in observations)
    if D==0:raise ValueError('All supplied times are blind')
    return sum(s*(c*a0-y) for c,s,y in observations)/D

def verify():
    counts=dict(exactPowerBalances=0,variableInductanceBalances=0,closedDampingTrajectories=0,independentRkTrajectories=0,dampingDerivativeChecks=0,integratedEnergyBalances=0,prescribedSpeedResponses=0,uncoupledCases=0,orientationCases=0,exactModeRotations=0,singleTimeInversions=0,blindInversionRejections=0,noiseCovariances=0,multiTimeFits=0,blindMultiTimeRejections=0,temporalModeSchedules=0,finiteBandBounds=0,spatialAliases=0,nearBlindAmplifications=0)
    for m,L,R,K,u,I,force in product(map(F,[F(1,2),1,2]),map(F,[F(1,2),1,2]),map(F,[0,F(1,2),2]),map(F,[-2,0,1]),map(F,[-1,F(1,2)]),map(F,[F(-1,4),0,F(3,4)]),map(F,[0,F(3,10)])):
        ud=(force+K*I)/m;Id=(-R*I-K*u)/L
        assert m*u*ud+L*I*Id==force*u-R*I*I;counts['exactPowerBalances']+=1
        for Lx in map(F,[F(-1,3),F(1,2)]):
            ud=(force+K*I+Lx*I*I/2)/m;Id=(-R*I-K*u-Lx*u*I)/L
            assert m*u*ud+L*I*Id+Lx*u*I*I/2==force*u-R*I*I;counts['variableInductanceBalances']+=1
    for zeta,tau,sign in product([0,.25,.5,1,2,4],[0,.1,.5,1,2,4],[-1,1]):
        v,j=free(zeta,tau,sign);A=[[0,sign],[-sign,-2*zeta]]
        ref=mv(expm(A,tau),[1,0]);near(v,ref[0],1e-11);near(j,ref[1],1e-11);counts['closedDampingTrajectories']+=1
        if tau:
            h=1e-4
            derivative=[sum(w*free(zeta,tau+k*h,sign)[i] for k,w in [(-2,1),(-1,-8),(1,8),(2,-1)])/(12*h) for i in range(2)]
            for actual,expected in zip(derivative,mv(A,[v,j])):near(actual,expected,2e-8)
            counts['dampingDerivativeChecks']+=1
        if tau in [.5,1,2,4]:
            heat=simpson(lambda t:2*zeta*free(zeta,t,sign)[1]**2,tau)
            near((v*v+j*j)/2+heat,.5,1e-10);counts['integratedEnergyBalances']+=1
        if zeta in [0,.5,1,2] and tau in [.5,1,2]:
            ref=rk4(A,tau)
            near(v,ref[0],1e-10);near(j,ref[1],1e-10);counts['independentRkTrajectories']+=1
    for R,L,K,u,t in product([0,.5,1,2],[.5,1,2],[-1,0,1],[-.2,.2],[0,.25,1,2]):
        I=-K*u*t/L if R==0 else K*u*math.expm1(-R*t/L)/R
        Id=-K*u/L*math.exp(-R*t/L)
        near(L*Id+R*I,-K*u,1e-13);near(-K*u*I,L*I*Id+R*I*I,1e-13);counts['prescribedSpeedResponses']+=1
    for R,L,t in product([0,.5,2],[.5,1,2],[0,.5,2]):
        A=[[0,0],[0,-R/L]];actual=mv(expm(A,t),[.2,-.3])
        near(actual[0],.2);near(actual[1],-.3*math.exp(-R*t/L));counts['uncoupledCases']+=1
    for B,H,u in product(map(F,[F(-1,2),F(1,2)]),map(F,[F(1,2),2]),map(F,[F(-1,5),0,F(1,5)])):
        I=F(3,10);emf=-B*H*u;force=B*H*I
        assert (-u*B)*H==emf and I*H*B==force and -force*u==emf*I
        assert (-emf)*(-I)==emf*I and (-I)*(-H)*B==force;counts['orientationCases']+=1
    pairs=[(F(1),F(0)),(F(0),F(1)),(F(-1),F(0)),(F(0),F(-1)),(F(3,5),F(4,5)),(F(3,5),F(-4,5)),(F(-5,13),F(12,13)),(F(-5,13),F(-12,13))]
    initial=[(F(3,5),F(4,5)),(F(-1,2),F(1)),(F(2),F(-3,2))]
    for (c,s),(a,b) in product(pairs,initial):
        y=c*a-s*b;bt=s*a+c*b
        assert y*y+bt*bt==a*a+b*b;counts['exactModeRotations']+=1
        if s:
            assert invert(a,y,c,s)==b;counts['singleTimeInversions']+=1
        else:
            assert c*a-s*(b+1)==y
            try:invert(a,y,c,s)
            except ValueError:counts['blindInversionRejections']+=1
            else:raise AssertionError('Blind inverse accepted')
    for c,s in pairs:
        if not s:continue
        for a,b,d in product([F(0),F(1,10)],[F(1,10),F(1,5)],[F(0),F(1,20)]):
            errors=[(c*a*X-d*X-b*Y)/s for X,Y in product([-1,1],repeat=2)]
            variance=sum(e*e for e in errors)/4
            assert sum(errors)==0 and variance==(c*c*a*a+d*d+b*b-2*c*a*d)/(s*s);counts['noiseCovariances']+=1
    for p1,p2,(a,b) in product(pairs,pairs,initial):
        observations=[(c,s,c*a-s*b) for c,s in [p1,p2]];D=p1[1]**2+p2[1]**2
        if D:
            assert fit_known_a(a,observations)==b
            errors=[]
            for e1,e2 in product([F(-1,10),F(1,10)],repeat=2):
                noisy=[(c,s,y+e) for (c,s,y),e in zip(observations,[e1,e2])]
                errors.append(fit_known_a(a,noisy)-b)
            assert sum(e*e for e in errors)/4==F(1,100)/D;counts['multiTimeFits']+=1
        else:
            try:fit_known_a(a,observations)
            except ValueError:counts['blindMultiTimeRejections']+=1
            else:raise AssertionError('Blind multi-time fit accepted')
        lam=F(1,3);ridge=sum(s*(c*a-y) for c,s,y in observations)/(D+lam)
        assert ridge==b*D/(D+lam)
    for n,fraction in product(range(1,9),map(F,[0,F(1,6),F(1,4),F(1,2),F(3,4),1,F(5,4)])):
        phase=n*fraction;blind=phase.denominator==1
        sine=0. if blind else math.sin(math.pi*float(phase));cosine=(-1.)**int(phase) if blind else math.cos(math.pi*float(phase))
        a,b=.6,.8;y=a*cosine-b*sine
        if blind:near(y,a*cosine)
        else:near(invert(a,y,cosine,sine),b,1e-11)
        for z in [-.7,0,.4]:
            f=lambda x:(a*math.cos(n*x)+b*math.sin(n*x))/2
            g=lambda x:(a*math.cos(n*x)-b*math.sin(n*x))/2
            near(f(z-math.pi*float(fraction))+g(z+math.pi*float(fraction)),y*math.cos(n*z),1e-11)
        counts['temporalModeSchedules']+=1
    for K in range(1,21):
        ss=[math.sin(n*math.pi/(2*K)) for n in range(1,K+1)]
        assert min(ss)>0;near(min(ss),math.sin(math.pi/(2*K)),1e-13);counts['finiteBandBounds']+=1
    for N,n in product([8,12,16],[1,2,3]):
        for j in range(N):near(math.cos(n*2*math.pi*j/N),math.cos((n+N)*2*math.pi*j/N),1e-12)
        counts['spatialAliases']+=1
    for delta in [1e-2,1e-4,1e-6]:
        for phase in [delta,math.pi-delta,math.pi+delta]:
            c,s=math.cos(phase),math.sin(phase);a,b=.6,.8;y=c*a-s*b+.01
            near(invert(a,y,c,s)-b,-.01/s,1e-11);counts['nearBlindAmplifications']+=1
    return counts

def references():
    loops=[dict(B=B,height=H,u=u,emf=-B*H*u,current=.3,force=B*H*.3) for B,H,u in product([-.5,.5],[.5,2],[-.2,0,.2])]
    pulses=[]
    for beta,t in product([0,.25,.5,1],[0,.5,1]):
        E=math.exp(-t*t/2);pulses.append(dict(beta=beta,z=0,time=t,E=E,cB=(2*beta-1)*E))
    for beta,z in product([0,1],[-1,1]):
        t=.5;right=beta*math.exp(-(z-t)**2/2);left=(1-beta)*math.exp(-(z+t)**2/2)
        pulses.append(dict(beta=beta,z=z,time=t,E=right+left,cB=right-left))
    return dict(loops=loops,pulses=pulses)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');p.add_argument('--references',action='store_true');a=p.parse_args()
    print(json.dumps(references() if a.references else verify(),sort_keys=True))
