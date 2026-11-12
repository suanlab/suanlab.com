"""Independent thin-lens uncertainty and aperture regularity checks; stdlib only."""
import json
import math
import sys


def simpson(f,a,b,n=4000):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))
def close(a,b,tol=1e-9):assert abs(a-b)<=tol*(1+abs(b)),(a,b)
def sinc(x):return 1-x*x/6+x**4/120-x**6/5040 if abs(x)<1e-4 else math.sin(x)/x
def dsinc(x):return -x/3+x**3/30-x**5/840 if abs(x)<1e-4 else (x*math.cos(x)-math.sin(x))/(x*x)
def aperture(x,a,e):return 1 if abs(x)<=a else max(0,(a+e-abs(x))/e)
def fourier(p,a,e):return (2*a+e)/math.sqrt(2*math.pi)*sinc((a+e/2)*p)*sinc(e*p/2)
def fourier_derivative(p,a,e):
    c=(2*a+e)/math.sqrt(2*math.pi);aa=a+e/2;b=e/2
    return c*(aa*dsinc(aa*p)*sinc(b*p)+b*sinc(aa*p)*dsinc(b*p))
def moments(a,e):
    norm=2*a+2*e/3
    x2=(2*a**3/3+2*a*a*e/3+a*e*e/3+e**3/15)/norm
    return norm,x2,1/(e*(a+e/3))


def verify():
    counts=dict(imageDensityChecks=0,truncatedImageMoments=0,tailProbabilities=0,fixedSensorMoments=0,localRemainderBounds=0,apertureMoments=0,fourierFieldChecks=0,fourierDerivativeChecks=0,hardEdgeCutoffs=0,spectralMomentBounds=0,propagatedMomentBounds=0,spectralRefinements=0)
    for f in [.05,.1,.2]:
        for a in [.001,.002]:
            for ratio in [.25,.5,.75]:
                mu=ratio*a;left=a-mu;right=a+mu;assert f+mu-a>0
                for scale in [1,2,4,8]:
                    cutoff=scale*f*f/left;h=f*f/cutoff
                    tail=(min(h,right)+min(h,left))/(2*a);close(tail,f*f/(a*cutoff));counts['tailProbabilities']+=1
                    actual=(simpson(lambda d:f**4/(d*d),h,left,8000)+simpson(lambda d:f**4/(d*d),h,right,8000))/(2*a)
                    expected=f*f*cutoff/a-f**4/(2*a)*(1/left+1/right)
                    close(actual,expected,2e-9);counts['truncatedImageMoments']+=1
                for sign,endpoint in [(-1,left),(1,right)]:
                    for scale in [1.1,2,5]:
                        w=sign*scale*f*f/endpoint;step=abs(w)*1e-5
                        # Direct length of transformed input interval, before density formula.
                        lo,hi=sorted([f*f/(w-step),f*f/(w+step)])
                        actual=max(0,min(hi,right)-max(lo,-left))/(2*step*2*a)
                        close(actual,f*f/(2*a*w*w),2e-9);counts['imageDensityChecks']+=1
                for z in [0,.1,.2]:
                    for theta in [-.02,0,.03]:
                        def sensor(d):
                            u=f+d;lens=.001+u*theta;out=theta-lens/f
                            return lens+z*out
                        mean=simpson(sensor,mu-a,mu+a,2000)/(2*a)
                        variance=simpson(lambda d:(sensor(d)-mean)**2,mu-a,mu+a,2000)/(2*a)
                        expected=(1-z/f)*.001+((f+mu)*(1-z/f)+z)*theta
                        close(mean,expected,1e-13);close(variance,theta*theta*(1-z/f)**2*a*a/3,1e-16);counts['fixedSensorMoments']+=1
        mu=.01
        for ratio in [.1,.25,.5]:
            a=ratio*mu
            for error in [-a,-a/2,0,a/2,a]:
                exact=f+f*f/(mu+error);linear=f+f*f/mu-f*f*error/(mu*mu)
                remainder=f*f*error*error/(mu*mu*(mu+error))
                close(exact-linear,remainder);assert abs(remainder)<=f*f*a*a/(mu*mu*(abs(mu)-a))+1e-12;counts['localRemainderBounds']+=1
    for a in [.5,1,2]:
        for ratio in [.5,.2,.1,.05]:
            e=a*ratio;norm,x2,p2=moments(a,e)
            integral=lambda f:2*(simpson(f,0,a,2000)+simpson(f,a,a+e,2000))
            actual_norm=integral(lambda x:aperture(x,a,e)**2)
            actual_x2=integral(lambda x:x*x*aperture(x,a,e)**2)/actual_norm
            actual_derivative=2*simpson(lambda x:1/(e*e),a,a+e,2000)
            difference=2*simpson(lambda x:aperture(x,a,e)**2,a,a+e,2000)
            close(actual_norm,norm,1e-12);close(actual_x2,x2,1e-12);close(actual_derivative/norm,p2,1e-11);close(difference,2*e/3,1e-12);counts['apertureMoments']+=1
            for p in [0,.3/a,1/a,3/a,7/a]:
                actual=integral(lambda x:aperture(x,a,e)*math.cos(p*x))/math.sqrt(2*math.pi)
                derivative=-integral(lambda x:x*aperture(x,a,e)*math.sin(p*x))/math.sqrt(2*math.pi)
                close(actual,fourier(p,a,e),2e-11);close(derivative,fourier_derivative(p,a,e),2e-11);counts['fourierFieldChecks']+=1;counts['fourierDerivativeChecks']+=1
        for number in [2,4,8,16]:
            cutoff=number*math.pi/a
            actual=simpson(lambda p:2/math.pi*math.sin(a*p)**2,-cutoff,cutoff,8000)/(2*a)
            expected=(cutoff-math.sin(2*a*cutoff)/(2*a))/(math.pi*a)
            close(actual,expected,1e-12);counts['hardEdgeCutoffs']+=1
    a=1
    for ratio in [.5,.2,.1,.05]:
        e=ratio;norm,x2,p2=moments(a,e);cutoff=200/e;n=2*math.ceil(25*cutoff);b=.3*a*a
        actual=simpson(lambda p:p*p*fourier(p,a,e)**2,-cutoff,cutoff,n)
        refined=simpson(lambda p:p*p*fourier(p,a,e)**2,-cutoff,cutoff,2*n)
        assert abs(actual-refined)<1e-5;counts['spectralRefinements']+=1
        tail_p2=16/(math.pi*e*e*cutoff)
        assert -1e-7<=norm*p2-refined<=tail_p2+1e-7;counts['spectralMomentBounds']+=1
        propagated=simpson(lambda p:fourier_derivative(p,a,e)**2+b*b*p*p*fourier(p,a,e)**2,-cutoff,cutoff,2*n)
        constant=4/(e*math.sqrt(2*math.pi));tail_derivative=4*constant**2*((a+e)**2/(3*cutoff**3)+4/(5*cutoff**5))
        assert -1e-7<=norm*(x2+b*b*p2)-propagated<=tail_derivative+b*b*tail_p2+1e-7;counts['propagatedMomentBounds']+=1
    return counts


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Ray/wave research checks passed: '+json.dumps(result,sort_keys=True))
