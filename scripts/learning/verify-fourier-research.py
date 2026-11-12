"""Independent phase-cycle, pupil-ambiguity and diversity checks; standard library only."""
import cmath
import itertools
import json
import math
import sys


def close(a,b,tol=1e-10):assert abs(a-b)<=tol*(1+abs(b)),(a,b)
def simpson(f,a,b,n=2400):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))
def recover(values):return complex(values[0]-values[2],values[1]-values[3])/4
def readings(r,s,offset=0):return [abs(r*cmath.exp(1j*(phi+offset))+s)**2 for phi in [0,math.pi/2,math.pi,3*math.pi/2]]
def sinc(x):return 1-x*x/6+x**4/120-x**6/5040 if abs(x)<1e-4 else math.sin(x)/x
def pupil(q,b,t):return cmath.exp(1j*(q*t*t+b*t))
def kernel(q,b,obstruction,x):
    return simpson(lambda t:pupil(q,b,t)*cmath.exp(2j*math.pi*x*t),-1,-obstruction)+simpson(lambda t:pupil(q,b,t)*cmath.exp(2j*math.pi*x*t),obstruction,1)
def otf_integral(q,b,obstruction,t):
    intervals=[[-1,-obstruction],[obstruction,1]];out=0j
    for left in intervals:
        for right in intervals:
            lo=max(left[0],right[0]+t);hi=min(left[1],right[1]+t)
            if hi>lo:out+=simpson(lambda u:pupil(q,b,u)*pupil(q,b,u-t).conjugate(),lo,hi)
    return out/(2*(1-obstruction))
def clear_otf(q,b,t):
    length=max(0,2-abs(t))
    return length/2*sinc(q*t*length)*cmath.exp(1j*b*t)


def verify():
    counts=dict(phaseCycles=0,intensityReadings=0,readErrorBounds=0,ratioErrorBounds=0,phaseCalibrationStates=0,noiseEnumerations=0,referenceAmbiguities=0,pupilConjugacyStates=0,pointSpreadAmbiguities=0,otfAmbiguities=0,clearPupilIntegrals=0,diversitySeriesBounds=0,certifiedNonzeroDiversityCases=0)
    for rm in [0,.2,.8,1]:
        for rp in [-.7,0,.6]:
            for sm in [0,.3,.5]:
                for sp in [-1,.4]:
                    r=rm*cmath.exp(1j*rp);s=sm*cmath.exp(1j*sp);values=readings(r,s);z=r.conjugate()*s;actual=recover(values);base=abs(r)**2+abs(s)**2
                    close(actual,z);close((values[0]+values[2])/2,base);close((values[1]+values[3])/2,base);assert 2*abs(z)<=base+1e-12 and min(values)>=0
                    counts['phaseCycles']+=1;counts['intensityReadings']+=4
                    for j in range(4):
                        for sign in [-1,1]:
                            eta=[0]*4;eta[j]=sign*1e-4;out=recover([x+e for x,e in zip(values,eta)])
                            assert abs(out-z)<=1e-4/(2*math.sqrt(2))+1e-12;counts['readErrorBounds']+=1
                            if rm>0:
                                close(z/abs(r)**2,s/r);assert abs(out/abs(r)**2-s/r)<=1e-4/(2*math.sqrt(2)*abs(r)**2)+1e-11;counts['ratioErrorBounds']+=1
                    for offset in [-.2,0,.3]:
                        actual=recover(readings(r,s,offset));close(actual,z*cmath.exp(-1j*offset));close(abs(actual-z),2*abs(z)*abs(math.sin(offset/2)));counts['phaseCalibrationStates']+=1
    errors=[]
    for eta in itertools.product([-1,1],repeat=4):errors.append(recover(eta));counts['noiseEnumerations']+=1
    close(sum(z.real*z.real for z in errors)/16,1/8);close(sum(z.imag*z.imag for z in errors)/16,1/8);close(sum(abs(z)**2 for z in errors)/16,1/4);close(sum(z.real*z.imag for z in errors)/16,0)
    for phase in [-1,.4,1.2]:
        first=readings(.8,.5*cmath.exp(1j*phase));second=readings(.5,.8*cmath.exp(1j*phase))
        for x,y in zip(first,second):close(x,y)
        assert abs(.8**2-.5**2)>.3;counts['referenceAmbiguities']+=1
    for q in [.5,2,4]:
        for b in [-.4,.3]:
            for obstruction in [0,.4]:
                for t in [-.8,-.2,0,.3,.9]:
                    close(pupil(-q,b,t),pupil(q,b,-t).conjugate());counts['pupilConjugacyStates']+=1
                for x in [-1.2,-.2,0,.4,1]:
                    a=kernel(q,b,obstruction,x);other=kernel(-q,b,obstruction,x)
                    close(other,a.conjugate(),2e-10);close(abs(a)**2,abs(other)**2);counts['pointSpreadAmbiguities']+=1
                for t in [-1.5,-.6,0,.2,1.2]:
                    a=otf_integral(q,b,obstruction,t);other=otf_integral(-q,b,obstruction,t);close(a,other,2e-10);counts['otfAmbiguities']+=1
    for q in [-3,-1,0,1,2,3]:
        for b in [0,.3]:
            for t in [-1.5,-.75,.1,.25,.5,1]:
                close(otf_integral(q,b,0,t),clear_otf(q,b,t),2e-10);counts['clearPupilIntegrals']+=1
    for q in [.5,2,4]:
        for d in [-.5,.5,1]:
            for t in [.01,.05,.1,.25]:
                length=2-t;v=t*length;leading=-2*q*d*v*v/3;remainder=((q+d)**4+(d-q)**4)*v**4/120
                difference=(clear_otf(q+d,0,t)-clear_otf(-q+d,0,t)).real/(length/2)
                assert abs(difference-leading)<=remainder+1e-12;counts['diversitySeriesBounds']+=1
                if abs(leading)>remainder:
                    assert abs(difference)>=abs(leading)-remainder-1e-12;counts['certifiedNonzeroDiversityCases']+=1
    return counts


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Fourier research checks passed: '+json.dumps(result,sort_keys=True))
