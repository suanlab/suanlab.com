"""Independent cubic-envelope and periodic-energy allocation checks; standard library only."""
import cmath
import json
import math
import sys


def simpson(f,a,b,n):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))


def cubic_error(h):
    return math.sqrt(simpson(lambda q:math.exp(-q*q)/math.sqrt(math.pi)*4*math.sin(h*q**3/12)**2,-8,8,16000))


def cubic_field(x,h,n=1600):
    return simpson(lambda q:math.exp(-q*q/2)*math.cos(q*x-h*q**3/6)/math.sqrt(2*math.pi),-8,8,n)


def mechanics(q,v,mass,stiffness,alpha):
    n=len(q)
    force=[stiffness*(q[(j+1)%n]-2*q[j]+q[(j-1)%n]) for j in range(n)]
    link=[stiffness*(q[(j+1)%n]-q[j])**2/2 for j in range(n)]
    link_rate=[stiffness*(q[(j+1)%n]-q[j])*(v[(j+1)%n]-v[j]) for j in range(n)]
    local=[mass*v[j]**2/2+alpha*link[(j-1)%n]+(1-alpha)*link[j] for j in range(n)]
    rate=[v[j]*force[j]+alpha*link_rate[(j-1)%n]+(1-alpha)*link_rate[j] for j in range(n)]
    current=[-stiffness*(q[(j+1)%n]-q[j])*((1-alpha)*v[(j+1)%n]+alpha*v[j]) for j in range(n)]
    return local,rate,current,link,link_rate


def circle(values):
    n=len(values)
    return sum(x*cmath.exp(2j*math.pi*j/n) for j,x in enumerate(values))/sum(values)


def verify():
    counts=dict(spectralErrorStates=0,realSpaceMomentIntegrals=0,pointwiseBounds=0,bandTailBounds=0,allocationStates=0,localBalanceSites=0,circularMomentStates=0,travelingTransportStates=0,timeAverageBounds=0,illConditionedCenters=0)
    sixth=simpson(lambda q:q**6*math.exp(-q*q)/math.sqrt(math.pi),-8,8,16000)
    assert abs(sixth-15/8)<2e-12
    for h in [0,.1,.25,.5,1,2,4,8]:
        actual=cubic_error(h)
        assert actual<=min(2,math.sqrt(5/96)*abs(h))+1e-12
        assert abs(actual-cubic_error(-h))<1e-13
        counts['spectralErrorStates']+=1
    for h in [-2,-.5,.5,2]:
        points=[-40+j/8 for j in range(641)]
        field=[cubic_field(x,h) for x in points]
        def sampled(values):return .125/3*(values[0]+values[-1]+sum((4 if j%2 else 2)*values[j] for j in range(1,len(values)-1)))
        norm=sampled([x*x for x in field]);mean=sampled([x*y*y for x,y in zip(points,field)])/norm
        variance=sampled([(x-mean)**2*y*y for x,y in zip(points,field)])/norm
        error=math.sqrt(sampled([(y-math.exp(-x*x/2))**2 for x,y in zip(points,field)])/math.sqrt(math.pi))
        assert abs(norm-math.sqrt(math.pi))<2e-8 and abs(mean-h/4)<2e-8
        assert abs(variance-(.5+h*h/8))<2e-7 and abs(error-cubic_error(h))<2e-8
        counts['realSpaceMomentIntegrals']+=1
        for x,y in zip(points,field):
            assert abs(y-math.exp(-x*x/2))<=min(2,2*abs(h)/(3*math.sqrt(2*math.pi)))+1e-11
            counts['pointwiseBounds']+=1
    for h in [.3,1,2]:
        for cutoff in [.5,1,2]:
            actual=simpson(lambda q:math.exp(-q*q)/math.sqrt(math.pi)*4*math.sin(h*q**3/12)**2,-cutoff,cutoff,4000)+4*math.erfc(cutoff)
            assert actual<=5*h*h/96+4*math.erfc(cutoff)+1e-12
            counts['bandTailBounds']+=1
    for n in [8,16]:
        for mass in [.5,2]:
            for stiffness in [.5,2]:
                for mode in [1,3]:
                    frequency=2*math.sqrt(stiffness/mass)*math.sin(math.pi*mode/n)
                    for phase in [0,.2,.7,1.2]:
                        for traveling in [False,True]:
                            angles=[2*math.pi*mode*j/n for j in range(n)]
                            q=[.1*math.cos(x-phase) if traveling else .1*math.cos(x)*math.cos(phase) for x in angles]
                            v=[.1*frequency*math.sin(x-phase) if traveling else -.1*frequency*math.cos(x)*math.sin(phase) for x in angles]
                            symmetric=mechanics(q,v,mass,stiffness,.5);energy=sum(symmetric[0]);zs=circle(symmetric[0])
                            for alpha in [0,.25,.5,1]:
                                local,rate,current,link,link_rate=mechanics(q,v,mass,stiffness,alpha);delta=alpha-.5
                                assert min(local)>=-1e-14 and abs(sum(local)-energy)<1e-13
                                for j in range(n):
                                    assert abs(rate[j]-current[(j-1)%n]+current[j])<1e-13
                                    assert abs(current[j]-symmetric[2][j]-delta*link_rate[j])<1e-13
                                    counts['localBalanceSites']+=1
                                assert abs(sum(current)-sum(symmetric[2])-delta*sum(link_rate))<1e-13
                                z=circle(local);step=cmath.exp(2j*math.pi/n)-1
                                predicted=delta*step*sum(x*cmath.exp(2j*math.pi*j/n) for j,x in enumerate(link))/energy
                                assert abs(z-zs-predicted)<1e-13
                                assert abs(z-zs)<=2*abs(delta)*math.sin(math.pi/n)*sum(link)/energy+1e-13
                                derivative=sum(x*cmath.exp(2j*math.pi*j/n) for j,x in enumerate(rate))/energy
                                assert abs(derivative-step*sum(x*cmath.exp(2j*math.pi*j/n) for j,x in enumerate(current))/energy)<1e-13
                                counts['allocationStates']+=1;counts['circularMomentStates']+=1
    for n in [16,32]:
        for mass in [.5,1]:
            for stiffness in [.5,2]:
                modes=[2,3,5];amplitudes=[.01,.02,.015];phases=[.2,-.3,.8]
                frequencies=[2*math.sqrt(stiffness/mass)*math.sin(math.pi*k/n) for k in modes]
                energies=[mass*n*w*w*a*a/2 for w,a in zip(frequencies,amplitudes)]
                expected=sum(e*math.sqrt(stiffness/mass)*math.cos(math.pi*k/n) for e,k in zip(energies,modes))/sum(energies)
                for t in [0,.5,1.7,4]:
                    q=[sum(a*math.cos(2*math.pi*k*j/n-w*t+p) for k,a,w,p in zip(modes,amplitudes,frequencies,phases)) for j in range(n)]
                    v=[sum(w*a*math.sin(2*math.pi*k*j/n-w*t+p) for k,a,w,p in zip(modes,amplitudes,frequencies,phases)) for j in range(n)]
                    for alpha in [0,.25,.5,1]:
                        local,_,current,link,_=mechanics(q,v,mass,stiffness,alpha)
                        assert abs(sum(link)/sum(local)-.5)<1e-12
                        assert abs(sum(current)/sum(local)-expected)<2e-12
                        counts['travelingTransportStates']+=1
    frequency=2*math.sin(math.pi/8)
    for alpha in [0,.25,1]:
        for end in [.3,1,2.7,9]:
            delta=alpha-.5
            actual=simpson(lambda t:-delta*frequency*math.sin(2*frequency*t),0,end,4000)/end
            expected=delta*(math.cos(frequency*end)**2-1)/end
            assert abs(actual-expected)<2e-12 and abs(actual)<=abs(delta)/end+1e-12
            counts['timeAverageBounds']+=1
    for epsilon in [-.1,-1e-4,1e-4,.1]:
        z=circle([1+epsilon,0,0,0,1-epsilon,0,0,0])
        assert abs(z-epsilon)<1e-15
        assert abs(abs(z)-abs(epsilon))<1e-15
        counts['illConditionedCenters']+=1
    return counts


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Dispersion research checks passed: '+json.dumps(result,sort_keys=True))
