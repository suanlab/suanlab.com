"""Independent two-grid sampling and censored-Poisson information; stdlib only."""
import cmath
import json
import math
import sys


def close(a,b,tol=1e-10):assert abs(a-b)<=tol*(1+abs(b)),(a,b)
def sinc(x):return 1 if x==0 else math.sin(math.pi*x)/(math.pi*x)
def simpson(f,a,b,n=2400):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))
def cap_distribution(mean,cap):
    values=[math.exp(-mean)]
    for n in range(1,257):values.append(values[-1]*mean/n)
    tail=sum(values[cap:]);bins=values[:cap]+[tail]
    derivative=[p*(n/mean-1) for n,p in enumerate(values[:cap])]+[values[cap-1]]
    fisher=sum(d*d/p for p,d in zip(bins,derivative));average=sum(n*p for n,p in enumerate(bins));variance=sum((n-average)**2*p for n,p in enumerate(bins))
    return bins,derivative,fisher,variance,values

def code_reference(mean,sigma):
    bins,derivative,jcap,variance,_=cap_distribution(mean,3)
    cdf=lambda x:.5*math.erfc(-x/math.sqrt(2))
    kernel=[]
    for code in range(4):
        row=[]
        for charge in range(4):
            if sigma==0:value=float(code==charge)
            elif code==0:value=cdf((.5-charge)/sigma)
            elif code==3:value=.5*math.erfc((2.5-charge)/(sigma*math.sqrt(2)))
            else:value=cdf((code+.5-charge)/sigma)-cdf((code-.5-charge)/sigma)
            row.append(value)
        kernel.append(row)
    probabilities=[sum(q*p for q,p in zip(row,bins)) for row in kernel]
    slopes=[sum(q*d for q,d in zip(row,derivative)) for row in kernel]
    information=sum(d*d/p for p,d in zip(probabilities,slopes))
    return dict(mean=mean,readSigma=sigma,probabilities=probabilities,derivatives=slopes,information=information,chargeInformation=jcap,storedVariance=variance,kernel=kernel)

def reference_cases():return [code_reference(mean,sigma) for mean in [.5,2,5,10] for sigma in [0,.5,1.5]]

def verify():
    counts=dict(twoGridSystems=0,realPixelDemodulations=0,pixelIntegralChecks=0,ditherErrorBounds=0,apertureZeroChecks=0,censoredInformationStates=0,tailScoreIdentities=0,tailDerivativeChecks=0,readChannelContractions=0,conditionalScoreChecks=0,codeDerivativeChecks=0,railLikelihoodSequences=0)
    c=[.25,.4*cmath.exp(.7j)]
    for frequency in [.1,.3,.45]:
        for width in [.25,.5,1]:
            h=[sinc(frequency*width),sinc((frequency+1)*width)];a=[x*y for x,y in zip(c,h)]
            for shift in [.05,.1,.25,.5]:
                w=cmath.exp(2j*math.pi*shift);b=[a[0]+a[1],a[0]+w*a[1]]
                recovered=[(w*b[0]-b[1])/(w-1),(b[1]-b[0])/(w-1)]
                for x,y in zip(recovered,a):close(x,y)
                minimum=math.sqrt(2-2*abs(math.cos(math.pi*shift)))
                for delta in [0,shift]:
                    beta=cmath.exp(2j*math.pi*frequency*delta)*(a[0]+cmath.exp(2j*math.pi*delta)*a[1])
                    y=[(cmath.exp(2j*math.pi*frequency*(n+delta))*(a[0]+cmath.exp(2j*math.pi*delta)*a[1])).real for n in [0,1]]
                    inferred=complex(y[0],(math.cos(2*math.pi*frequency)*y[0]-y[1])/math.sin(2*math.pi*frequency));close(inferred,beta);counts['realPixelDemodulations']+=1
                    for n in [-1,0,1]:
                        center=n+delta
                        def scene(x):return 1+(c[0]*cmath.exp(2j*math.pi*frequency*x)+c[1]*cmath.exp(2j*math.pi*(frequency+1)*x)).real
                        average=simpson(scene,center-width/2,center+width/2)/width
                        expected=1+(cmath.exp(2j*math.pi*frequency*center)*(a[0]+cmath.exp(2j*math.pi*delta)*a[1])).real
                        close(average,expected,2e-11);assert average>0;counts['pixelIntegralChecks']+=1
                for e0,e1 in [(1e-4,0),(0,1e-4j),(1e-4,-1e-4j)]:
                    err=[(w*e0-e1)/(w-1),(e1-e0)/(w-1)];norm=math.sqrt(abs(e0)**2+abs(e1)**2)
                    assert math.sqrt(sum(abs(x)**2 for x in err))<=norm/minimum+1e-12
                    assert math.sqrt(sum(abs(x/y)**2 for x,y in zip(err,h)))<=norm/(minimum*min(abs(x) for x in h))+1e-12;counts['ditherErrorBounds']+=1
                counts['twoGridSystems']+=1
    for delta in [0,.13,.5,.9]:
        for n in [-1,0,1]:
            center=n+delta;average=simpson(lambda x:math.cos(2*math.pi*1.25*x+.7),center-.4,center+.4)/.8;close(average,0,2e-12);counts['apertureZeroChecks']+=1
    for mean in [.2,.5,2,5,10,20]:
        for cap in [1,3,7]:
            bins,derivative,j,var,values=cap_distribution(mean,cap);tail=bins[-1];score=values[cap-1]/tail
            close(sum(bins),1);assert 0<j<=1/mean+1e-12
            conditional_mean=sum(n*p for n,p in enumerate(values) if n>=cap)/tail
            conditional_variance=sum((n-conditional_mean)**2*p for n,p in enumerate(values) if n>=cap)/tail
            close(conditional_mean/mean-1,score);close(1/mean-j,tail*conditional_variance/(mean*mean),2e-11);counts['tailScoreIdentities']+=1
            step=1e-5*max(1,mean);finite=(cap_distribution(mean+step,cap)[0][-1]-cap_distribution(mean-step,cap)[0][-1])/(2*step);close(finite,values[cap-1],2e-9);counts['tailDerivativeChecks']+=1;counts['censoredInformationStates']+=1
        for sigma in [0,.5,1.5]:
            r=code_reference(mean,sigma);assert 0<=r['information']<=r['chargeInformation']+1e-12<=1/mean+1e-12
            bins,derivative,j,_,_=cap_distribution(mean,3);loss=0
            for code,(probability,slope,row) in enumerate(zip(r['probabilities'],r['derivatives'],r['kernel'])):
                conditional_score=sum(q*d for q,d in zip(row,derivative))/probability;close(conditional_score,slope/probability)
                conditional_variance=sum(q*p*(d/p-conditional_score)**2 for q,p,d in zip(row,bins,derivative))/probability
                loss+=probability*conditional_variance;counts['conditionalScoreChecks']+=1
            close(j-r['information'],loss);counts['readChannelContractions']+=1
            step=1e-5*max(1,mean);plus=code_reference(mean+step,sigma);minus=code_reference(mean-step,sigma)
            for p,m,d in zip(plus['probabilities'],minus['probabilities'],r['derivatives']):close((p-m)/(2*step),d,2e-9);counts['codeDerivativeChecks']+=1
    for repeats in [1,5,20]:
        probabilities=[cap_distribution(mean,3)[0][-1]**repeats for mean in [.2,.5,2,5,10,20]];assert all(x<y for x,y in zip(probabilities,probabilities[1:]));counts['railLikelihoodSequences']+=1
    return counts

if __name__=='__main__':
    if '--references' in sys.argv:print(json.dumps(reference_cases(),sort_keys=True))
    else:
        result=verify();print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Sampling research checks passed: '+json.dumps(result,sort_keys=True))
