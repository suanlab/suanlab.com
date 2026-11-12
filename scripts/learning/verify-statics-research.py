"""Portable independent checks for the radial-density inverse and MQS skin models.
No product imports. Shell-potential quadrature and gradients check the sphere;
angular Bessel integrals, radial derivatives and power quadrature check the wire.
"""
import argparse,cmath,json,math
from fractions import Fraction as F

def I(n,z):
    assert n in (0,1)
    term=(z/2)**n;total=term
    for m in range(1,61):
        term*=z*z/(4*m*(m+n));total+=term
    return total

def simpson(fn,a,b,n=512):
    h=(b-a)/n
    return h/3*(fn(a)+fn(b)+sum((4 if i%2 else 2)*fn(a+i*h) for i in range(1,n)))

def angular(n,z):return simpson(lambda t:cmath.exp(z*math.cos(t))*math.cos(n*t),0,math.pi)/math.pi

def skin(omega,x):
    z=cmath.sqrt(1j*omega);den=I(1,z)
    return z*I(0,z*x)/(2*den),I(1,z*x)/den,z*I(0,z)/(2*den)

def sphere(alpha,x):
    if x<=1:return x*(1+3*alpha/5*(x*x-1)),(3-x*x)/2-3*alpha/20*(1-x*x)**2
    return 1/x**2,1/x

# Four-point Gauss-Legendre, independent from the field/potential formulas.
NODES=[-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526]
WEIGHTS=[.3478548451374538,.6521451548625461,.6521451548625461,.3478548451374538]
def gauss(fn,a,b):
    return (b-a)/2*sum(w*fn((a+b)/2+(b-a)/2*t) for t,w in zip(NODES,WEIGHTS))

def shell_potential(alpha,x):
    if x==0:return gauss(lambda s:3*s*(1+alpha*(s*s-.6)),0,1)
    inner=min(x,1)
    return gauss(lambda s:3*s*s*(1+alpha*(s*s-.6)),0,inner)/x+(gauss(lambda s:3*s*(1+alpha*(s*s-.6)),inner,1) if x<1 else 0)

def near(a,b,atol=3e-9,rtol=3e-8):
    assert math.isfinite(abs(a)) and abs(a-b)<=atol+rtol*abs(b),(a,b)

def verify():
    c=dict(sphereMassIntegrals=0,shellPotentials=0,shellFieldGradients=0,interiorRecoveries=0,exteriorAmbiguities=0,probeSensitivityChecks=0,besselAngularChecks=0,radialDifferentialChecks=0,currentNormalizations=0,complexPowerChecks=0,dcLimitSequences=0)
    for af in [F(-1),F(-1,2),F(0),F(1,2),F(1)]:
        assert 3*(F(1,3)+af*(F(1,5)-F(3,5)*F(1,3)))==1
        for xf in [F(i,20) for i in range(21)]:assert 1+af*(xf*xf-F(3,5))>0
        alpha=float(af);near(gauss(lambda s:3*s*s*(1+alpha*(s*s-.6)),0,1),1);c['sphereMassIntegrals']+=1
        for x in [0,.1,.35,.65,.9,1,1.2,2]:
            near(shell_potential(alpha,x),sphere(alpha,x)[1]);c['shellPotentials']+=1
        for x in [.1,.35,.65,.9,1.2,2]:
            h=1e-4;V=lambda t:shell_potential(alpha,t)
            field=(V(x+2*h)-8*V(x+h)+8*V(x-h)-V(x-2*h))/(12*h)
            near(field,sphere(alpha,x)[0]);c['shellFieldGradients']+=1
        for x in [.1,.3,.5,.7,.9]:
            E=sphere(alpha,x)[0];recovered=(x-E)/(.6*x*(1-x*x));near(recovered,alpha)
            c['interiorRecoveries']+=1
        for x in [1,1.2,2,4]:
            assert sphere(alpha,x)==sphere(0,x);c['exteriorAmbiguities']+=1
    optimum=1/math.sqrt(3);peak=.6*optimum*(1-optimum**2)
    for i in range(101):
        x=i/100;assert .6*x*(1-x*x)<=peak+1e-15;c['probeSensitivityChecks']+=1
    for om in [.01,.1,1,4,10,32]:
        z=cmath.sqrt(1j*om)
        for x in [0,.2,.5,.8,1]:
            for n in [0,1]:near(I(n,z*x),angular(n,z*x),2e-10,2e-10);c['besselAngularChecks']+=1
        j=lambda x:skin(om,x)[0];b=lambda x:skin(om,x)[1];Z=skin(om,1)[2]
        for x in [.15,.45,.75]:
            h=1e-3
            dj=(j(x-2*h)-8*j(x-h)+8*j(x+h)-j(x+2*h))/(12*h)
            d2j=(-j(x+2*h)+16*j(x+h)-30*j(x)+16*j(x-h)-j(x-2*h))/(12*h*h)
            near(d2j+dj/x,1j*om*j(x),3e-7,3e-7);c['radialDifferentialChecks']+=1
            near(dj,1j*om*b(x)/2,3e-8,3e-8);c['radialDifferentialChecks']+=1
            db=(b(x-2*h)-8*b(x-h)+8*b(x+h)-b(x+2*h))/(12*h)
            near(db+b(x)/x,2*j(x),3e-8,3e-8);c['radialDifferentialChecks']+=1
        near(2*simpson(lambda x:j(x)*x,0,1,1024),1,5e-10,5e-10);c['currentNormalizations']+=1
        near(2*simpson(lambda x:abs(j(x))**2*x,0,1,1024),Z.real,2e-9,2e-9);c['complexPowerChecks']+=1
        near(om/2*simpson(lambda x:abs(b(x))**2*x,0,1,1024),Z.imag,2e-9,2e-9);c['complexPowerChecks']+=1
        assert Z.real>=1-1e-12 and Z.imag>0 and abs(I(1,z))>0
    errors=[]
    for om in [1e-3,1e-4,1e-5,1e-6]:
        errors.append(max(abs(skin(om,x)[0]-1)+abs(skin(om,x)[1]-x) for x in [0,.25,.5,.75,1]))
        near(skin(om,1)[2],1+1j*om/8+om*om/192,1e-12,1e-12)
    assert all(b<a for a,b in zip(errors,errors[1:])) and errors[-1]<2e-7;c['dcLimitSequences']=1
    return c

def references():
    return dict(sphere=[dict(R=R,x=x,Q=1e-9,k=9e9,field=9*x/R**2 if x<=1 else 9/(R*x)**2,potential=9/R*sphere(0,x)[1]) for R in [.5,1,2] for x in [0,.25,.6,1,1.5,2]],cylinder=[dict(R=R,x=x,current=current,mu=4*math.pi*1e-7,field=4*math.pi*1e-7*current/(2*math.pi*R)*(x if x<=1 else 1/x)) for R in [.5,1,2] for current in [-.5,.75] for x in [0,.25,.6,1,2]])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--references',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    print(json.dumps(references() if a.references else verify(),sort_keys=True))
