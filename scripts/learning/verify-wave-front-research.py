"""Independent infinite-chain and passive-boundary checks, standard library only."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import sys


def green(n, clock):
    n=abs(n)
    if clock==0:return float(n==0)
    with localcontext() as ctx:
        ctx.prec=80
        s=Decimal(str(clock))
        term=s**(2*n)/Decimal(math.factorial(2*n));value=term
        for r in range(1,90):
            term*=-s*s/Decimal(r*(2*n+r));value+=term
        return float(value)


def green_bracket(n, s):
    term=s**(2*n)/math.factorial(2*n);value=term
    for r in range(1,32):
        term*=-s*s/Fraction(r*(2*n+r));value+=term
    following=value-term*s*s/Fraction(32*(2*n+32))
    return min(value,following),max(value,following)


def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h) for i in range(1,n)))


def incoming(t, amplitude=.1):
    z=t-2
    return amplitude*(1-z*z)**3 if abs(z)<1 else 0.0


def incoming_rate(t, amplitude=.1):
    z=t-2
    return -6*amplitude*z*(1-z*z)**2 if abs(z)<1 else 0.0


def boundary_value(t, impedance=1, spring=1, damping=.5):
    if t<=1:return 0.0
    upper=min(t,3)
    return 2*impedance/(impedance+damping)*simpson(lambda s:math.exp(-spring*(t-s)/(impedance+damping))*incoming_rate(s),1,upper,4000)


def boundary_rk4(impedance,spring,damping,duration,steps):
    y=[0.0]*4
    h=duration/steps
    def rhs(t,state):
        f=incoming_rate(t)
        rate=(2*impedance*f-spring*state[0])/(impedance+damping)
        return [rate,impedance*f*f,impedance*(rate-f)**2,damping*rate*rate]
    for j in range(steps):
        t=j*h;a=rhs(t,y);b=rhs(t+h/2,[y[i]+h*a[i]/2 for i in range(4)])
        c=rhs(t+h/2,[y[i]+h*b[i]/2 for i in range(4)]);d=rhs(t+h,[y[i]+h*c[i] for i in range(4)])
        y=[y[i]+h*(a[i]+2*b[i]+2*c[i]+d[i])/6 for i in range(4)]
    return y


def periodic_rk4(n,t,steps):
    h=t/steps;y=[1.0]+[0.0]*(2*n-1)
    def rhs(z):return z[n:]+[z[(j+1)%n]-2*z[j]+z[(j-1)%n] for j in range(n)]
    for _ in range(steps):
        a=rhs(y);b=rhs([y[i]+h*a[i]/2 for i in range(2*n)])
        c=rhs([y[i]+h*b[i]/2 for i in range(2*n)]);d=rhs([y[i]+h*c[i] for i in range(2*n)])
        y=[y[i]+h*(a[i]+2*b[i]+2*c[i]+d[i])/6 for i in range(2*n)]
    return y[:n]


def verify():
    counts=dict(initialGreenSites=0,alternatingIntervals=0,fourierCoefficients=0,latticeTailBounds=0,periodicRefinements=0,periodicCoordinates=0,materialScalings=0,passiveFrequencyStates=0,boundaryRefinements=0,transientBudgets=0,tailEnergySplits=0)
    for n in range(-12,13):
        assert green(n,0)==float(n==0)
        assert green(n,.001)==green(n,-.001)
        counts['initialGreenSites']+=1
    for s in [Fraction(1,4),Fraction(1,2),Fraction(1),Fraction(2)]:
        for n in range(13):
            value=green(n,float(s))
            independent=sum(math.cos(2*float(s)*math.sin(math.pi*j/2048))*math.cos(2*math.pi*n*j/2048) for j in range(2048))/2048
            assert abs(value-independent)<2e-14
            counts['fourierCoefficients']+=1
            if n>0 and s*s<2*n+1:
                lower,upper=green_bracket(n,s)
                leading=s**(2*n)/math.factorial(2*n)
                assert leading*(1-s*s/(2*n+1))<=lower<=upper<=leading
                assert lower>0 and upper-lower<Fraction(1,10**45)
                assert abs(value-float(lower))<1e-14
                counts['alternatingIntervals']+=1
        for first in [2,3,5,8]:
            ss=float(s)**2
            bound=2*float(s)**(2*first)/math.factorial(2*first)/(1-ss/((2*first+1)*(2*first+2)))
            actual=2*sum(abs(green(j,float(s))) for j in range(first,first+35))
            assert 0<actual<=bound*(1+1e-13)
            counts['latticeTailBounds']+=1
    for n in [8,16]:
        for t in [.5,1,2]:
            coarse,fine=periodic_rk4(n,t,600),periodic_rk4(n,t,1200)
            for j in range(n):
                images=sum(green(j+w*n,t) for w in range(-3,4))
                assert abs(coarse[j]-fine[j])<2e-10
                assert abs(fine[j]-images)<2e-11
                counts['periodicCoordinates']+=1
            counts['periodicRefinements']+=1
    for tension in [.5,1,2]:
        for spacing in [.5,.25,.125]:
            for amplitude in [.1,.1*spacing]:
                positions=[amplitude,0,0,0,0]
                explicit=sum(tension/spacing*(positions[(j+1)%5]-positions[j])**2/2 for j in range(5))
                assert abs(explicit-tension*amplitude*amplitude/spacing)<1e-14
                counts['materialScalings']+=1
    for impedance in [.5,1,2]:
        for spring in [0,.25,1,4]:
            for ratio in [0,.5,1,2]:
                damping=ratio*impedance
                for frequency in [.1,.5,1,4]:
                    r=complex(-spring,frequency*(impedance-damping))/complex(spring,frequency*(impedance+damping))
                    absorption=4*frequency*frequency*impedance*damping/(spring*spring+frequency*frequency*(impedance+damping)**2)
                    incident=impedance*frequency*frequency*.1**2/2
                    reflected=incident*abs(r)**2
                    heat=damping*frequency*frequency*abs(.1*(1+r))**2/2
                    assert abs(abs(r)**2+absorption-1)<2e-14 and -1e-14<=absorption<=1+1e-14
                    assert abs(incident-reflected-heat)<2e-14
                    if spring==0 and ratio==1:assert abs(r)==0
                    counts['passiveFrequencyStates']+=1
    exact_bump_rate_integral=72*(Fraction(1,3)-Fraction(4,5)+Fraction(6,7)-Fraction(4,9)+Fraction(1,11))
    for impedance in [.5,1,2]:
        for spring in [.25,1,2]:
            for ratio in [0,.5,1,2]:
                damping=ratio*impedance
                coarse=boundary_rk4(impedance,spring,damping,3,1200)
                fine=boundary_rk4(impedance,spring,damping,3,2400)
                direct=boundary_value(3,impedance,spring,damping)
                assert max(abs(x-y) for x,y in zip(coarse,fine))<3e-10
                assert abs(fine[0]-direct)<1e-11
                counts['boundaryRefinements']+=1
                stored=spring*direct*direct/2
                assert abs(fine[1]-float(exact_bump_rate_integral)*impedance*.01)<2e-11
                assert abs(fine[1]-fine[2]-fine[3]-stored)<3e-11
                counts['transientBudgets']+=1
                rate=spring/(impedance+damping)
                out=simpson(lambda u:impedance*rate*direct*direct*math.exp(-2*u),0,24,10000)
                heat=simpson(lambda u:damping*rate*direct*direct*math.exp(-2*u),0,24,10000)
                assert abs(out-impedance/(impedance+damping)*stored)<1e-12
                assert abs(heat-damping/(impedance+damping)*stored)<1e-12
                assert abs(out+heat-stored)<2e-12
                counts['tailEnergySplits']+=1
    return counts


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Wave-front research checks passed: '+json.dumps(result,sort_keys=True))
