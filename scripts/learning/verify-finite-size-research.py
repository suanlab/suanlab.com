"""Independent Gibbs enumeration, rational mixture laws and shifted-root checks."""
from fractions import Fraction as F
from itertools import product
from math import exp, log, cosh, tanh, comb, sqrt, pi, cos, isqrt
import cmath,json

def mixture(n,p,c):
    return [(1-c)*comb(n,j)*p**j*(1-p)**(n-j)+c*((1-p if j==0 else 0)+(p if j==n else 0)) for j in range(n+1)]

def simpson(f,lo,hi,count=20000):
    h=(hi-lo)/count
    return h/3*(f(lo)+f(hi)+sum((4 if j%2 else 2)*f(lo+j*h) for j in range(1,count)))

def main():
    chains=cuts=moments=characteristics=escapes=fields=roots=quadratures=0
    for n in range(2,11):
        a=n//2;states=list(product((-1,1),repeat=n))
        for k in (0.,.25,1.,2.):
            w=[exp(k*sum(s[i]*s[i+1] for i in range(n-1))) for s in states]
            w0=[exp(k*sum(s[i]*s[i+1] for i in range(n-1) if i!=a-1)) for s in states]
            z,z0=sum(w),sum(w0);ps=[x/z for x in w];qs=[x/z0 for x in w0]
            assert abs(z/(2*(2*cosh(k))**(n-1))-1)<2e-13
            assert abs(z/z0-cosh(k))<2e-13*max(1,cosh(k))
            kl=sum(p*log(p/q) for p,q in zip(ps,qs));tv=sum(abs(p-q) for p,q in zip(ps,qs))/2
            assert abs(kl-(k*tanh(k)-log(cosh(k))))<2e-13
            assert abs(tv-tanh(k)/2)<2e-13
            left={};left0={};right={};right0={}
            for s,p,q in zip(states,ps,qs):
                left[s[:a]]=left.get(s[:a],0)+p;left0[s[:a]]=left0.get(s[:a],0)+q
                right[s[a:]]=right.get(s[a:],0)+p;right0[s[a:]]=right0.get(s[a:],0)+q
            assert max(abs(left[x]-left0[x]) for x in left)<2e-13
            mi=sum(p*log(p/(left[s[:a]]*right[s[a:]])) for s,p in zip(states,ps))
            assert abs(mi-kl)<2e-13
            corr=sum(p*s[0]*s[-1] for s,p in zip(states,ps))
            assert abs(corr-tanh(k)**(n-1))<2e-13
            chains+=1
    for n in range(2,9):
        a=n//2;states=list(product((-1,1),repeat=n))
        for k in (.3,1.,2.):
            for mode in (0,1,2):
                hs=[0 if mode==0 else .2*(-1)**i if mode==1 else .4 for i in range(n)]
                z=z0=0.
                for s in states:
                    base=sum(h*x for h,x in zip(hs,s))+k*sum(s[i]*s[i+1] for i in range(n-1) if i!=a-1)
                    z0+=exp(base);z+=exp(base+k*s[a-1]*s[a])
                assert abs(log(z/z0))<=k+1e-12;cuts+=1
    for n in range(1,25):
        for p in (F(1,3),F(1,2),F(2,3)):
            for c in (F(1,4),F(1,2)):
                iid=mixture(n,p,F(0));v=p*(1-p)
                for power in (1,2):
                    cn=c/n**power;law=mixture(n,p,cn)
                    mean=sum(m*F(j,n) for j,m in enumerate(law));variance=sum(m*(F(j,n)-p)**2 for j,m in enumerate(law))
                    assert sum(law)==1 and mean==p and variance==v*(1+(n-1)*cn)/n
                    assert sum(abs(m-q) for m,q in zip(law,iid))/2<=cn
                    assert n*cn*v/(n*variance)==n*cn/(1+(n-1)*cn)
                    moments+=1
    for n in (16,64,256,1024):
        p=F(1,2);v=p*(1-p)
        for power in (.5,1,2):
            cn=F(1,2*isqrt(n)) if power==.5 else F(1,2*n**power);law=mixture(n,p,cn)
            for t in (.5,1.,2.):
                actual=sum(float(m)*cmath.exp(1j*t*sqrt(n)*(j/n-.5)) for j,m in enumerate(law))
                expected=(1-float(cn))*cos(t/(2*sqrt(n)))**n+float(cn)*cos(t*sqrt(n)/2)
                assert abs(actual-expected)<3e-13
                assert abs(actual-exp(-t*t/8))<=2*float(cn)+t**4/n
                characteristics+=1
            if power==1:
                tail=sum(m*n*(F(j,n)-p)**2 for j,m in enumerate(law) if n*(F(j,n)-p)**2>1)
                assert tail>=n*cn*v==F(1,8);escapes+=1
    for n in range(1,17):
        for kappa in range(4):
            weight=n**kappa;center=-log(weight)/n
            for g in (-.7,-.1,0.,.4):
                z=1+weight*exp(n*g);p=weight*exp(n*g)/z;correction=log(z)/n-max(0,g)
                assert -1e-14<=correction<=log(1+weight)/n+1e-14
                step=1e-8/n;value=weight*cmath.exp(n*complex(g,step));derivative=(value/(1+value)).imag/step
                assert abs(derivative-n*p*(1-p))<3e-9
                fields+=1
            logistic=lambda g: weight*exp(n*g)/(1+weight*exp(n*g))
            assert abs(logistic(center)-.5)<1e-14
            assert abs(logistic(center-log(9)/n)-.1)<1e-14
            assert abs(logistic(center+log(9)/n)-.9)<1e-14
            for j in range(n):
                z=cmath.exp(complex(center,(2*j+1)*pi/n));assert abs(1+weight*z**n)<8e-14*n
                turn=F(2*j+1,2*n);assert (n*turn)%1==F(1,2);roots+=1
    previous=0
    density=lambda u: exp(-abs(u))/(1+exp(-abs(u)))**2
    assert abs(simpson(density,-30,30)-1)<3e-12
    for n in (8,32,128,512):
        center=-2*log(n)/n;value=simpson(lambda u: cos(center+u/n)*density(u),-30,30)
        assert previous<value<1;previous=value;quadratures+=1
    assert previous>.999
    return {'status':'passed','openChainPreparations':chains,'fieldCutBounds':cuts,'mixtureMomentPreparations':moments,'characteristicComparisons':characteristics,'rareTailEscapes':escapes,'positiveFieldPreparations':fields,'complexRoots':roots,'responseQuadratures':quadratures}

if __name__=='__main__':print(json.dumps(main(),indent=2))
