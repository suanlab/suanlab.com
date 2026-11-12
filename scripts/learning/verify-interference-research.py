"""Independent classical phase moments and paraxial error checks; standard library only."""
import cmath
from decimal import Decimal, localcontext
from fractions import Fraction
import itertools
import json
import math
import sys


def simpson(f,a,b,n=2000):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))


def moments(phases,weights):
    total=sum(weights);p=[w/total for w in weights]
    return p,sum(w*cmath.exp(1j*x) for w,x in zip(p,phases)),sum(w*cmath.exp(2j*x) for w,x in zip(p,phases))


def aperture(fresnel,d,eta,a=1,b=1,phase=0,n=2000):
    return sum(amplitude*cmath.exp(1j*offset)*simpson(lambda t:cmath.exp(1j*math.pi*fresnel*(center+t)**2-2j*math.pi*eta*(center+t)),-.5,.5,n)
               for center,amplitude,offset in [(-d/2,a,0),(d/2,b,phase)])


def far_field(d,eta,a=1,b=1,phase=0):
    envelope=1 if eta==0 else math.sin(math.pi*eta)/(math.pi*eta)
    return envelope*(a*cmath.exp(1j*math.pi*d*eta)+b*cmath.exp(1j*phase-1j*math.pi*d*eta))


def on_axis_series(fresnel,d):
    """Integrate each power of exp(i*pi*F*u^2) exactly as a rational moment."""
    with localcontext() as ctx:
        ctx.prec=75
        pi=Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286')
        f=Fraction(fresnel);center=Fraction(d,2);low,high=center-Fraction(1,2),center+Fraction(1,2)
        real=imag=Decimal(0);coefficient=Decimal(1)
        for r in range(120):
            power=2*r+1
            moment=2*(high**power-low**power)/power
            term=coefficient*Decimal(moment.numerator)/Decimal(moment.denominator)
            if r%4==0:real+=term
            elif r%4==1:imag+=term
            elif r%4==2:real-=term
            else:imag-=term
            coefficient*=pi*Decimal(f.numerator)/Decimal(f.denominator)/Decimal(r+1)
        return complex(float(real),float(imag))


def verify():
    counts=dict(momentPopulations=0,varianceStates=0,positiveMomentForms=0,invalidMomentPairs=0,iidMeanEnumerations=0,correlatedMeanCases=0,apertureBoundStates=0,centeredMomentCases=0,independentSeriesPaths=0)
    for size in [2,3,5,8]:
        for shift in [0,.17,.8]:
            for skew in [0,1,3]:
                phases=[2*math.pi*j/size+shift for j in range(size)]
                weights=[1+skew*(j==0) for j in range(size)]
                p,m,n=moments(phases,weights)
                assert abs(n-m*m)<=1-abs(m)**2+2e-14
                matrix=[[1,m.conjugate(),n.conjugate()],[m,1,m.conjugate()],[n,m,1]]
                determinant=sum(((-1)**sum(order[i]>order[j] for i in range(3) for j in range(i+1,3)))*matrix[0][order[0]]*matrix[1][order[1]]*matrix[2][order[2]] for order in itertools.permutations(range(3)))
                assert abs(determinant-((1-abs(m)**2)**2-abs(n-m*m)**2))<2e-14
                for vector in [[1,1j,.3],[.2-.4j,-1,2j],[1,-2,1]]:
                    form=sum(complex(vector[i]).conjugate()*matrix[i][j]*vector[j] for i in range(3) for j in range(3))
                    assert abs(form.imag)<2e-14 and form.real>=-2e-14
                    counts['positiveMomentForms']+=1
                for psi in [0,.2,math.pi/4,math.pi/2]:
                    for window in [-.8,0,.3,1]:
                        baseline,contrast=1.3,1.1*window
                        values=[baseline+contrast*math.cos(psi+t) for t in phases]
                        mean=sum(w*x for w,x in zip(p,values));variance=sum(w*(x-mean)**2 for w,x in zip(p,values))
                        projection=(cmath.exp(1j*psi)*m).real
                        expected=contrast**2*((1+(cmath.exp(2j*psi)*n).real)/2-projection**2)
                        assert abs(mean-(baseline+contrast*projection))<1e-13 and abs(variance-expected)<1e-13
                        counts['varianceStates']+=1
                counts['momentPopulations']+=1
    for m,n,psi in [(.8,-.8,0),(.9j,.8,math.pi/2),(.95,0,0),(.5,-1,0)]:
        assert abs(n-m*m)>1-abs(m)**2
        assert (1+(cmath.exp(2j*psi)*n).real)/2-(cmath.exp(1j*psi)*m).real**2<0
        counts['invalidMomentPairs']+=1
    for psi in [0,math.pi/4,math.pi/2]:
        values=[2+2*math.cos(psi+t) for t in [0,math.pi]]
        mean=sum(values)/2;variance=sum((x-mean)**2 for x in values)/2
        for repetitions in range(1,7):
            means=[sum(row)/repetitions for row in itertools.product(values,repeat=repetitions)]
            actual=sum((x-mean)**2 for x in means)/len(means)
            assert abs(actual-variance/repetitions)<2e-13
            counts['iidMeanEnumerations']+=1
            correlated=sum((sum([x]*repetitions)/repetitions-mean)**2 for x in values)/2
            assert abs(correlated-variance)<2e-13
            counts['correlatedMeanCases']+=1
    for d in [1,2,4]:
        for a,b in [(1,1),(1,.4)]:
            moment=sum(amplitude*simpson(lambda t:abs((center+t)**2-d*d/4),-.5,0,1000)+amplitude*simpson(lambda t:abs((center+t)**2-d*d/4),0,.5,1000) for center,amplitude in [(-d/2,a),(d/2,b)])
            assert abs(moment-(a+b)*d/4)<1e-12
            counts['centeredMomentCases']+=1
            for fresnel in [.02,.2,1]:
                for eta in [-1,0,1]:
                    for phase in [0,.8]:
                        coarse=aperture(fresnel,d,eta,a,b,phase,1000)
                        field=aperture(fresnel,d,eta,a,b,phase,2000);far=far_field(d,eta,a,b,phase)
                        assert abs(field-coarse)<2e-8
                        m0=a+b;m2=m0*(d*d/4+1/12)
                        bound=min(2*m0,math.pi*fresnel*m2)
                        assert abs(field-far)<=bound+1e-10
                        corrected=cmath.exp(1j*math.pi*fresnel*d*d/4)*far
                        centered_bound=math.pi*fresnel*m0*d/4
                        assert abs(field-corrected)<=centered_bound+1e-10
                        assert abs(abs(field)**2-abs(far)**2)<=2*m0*min(bound,centered_bound)+1e-10
                        counts['apertureBoundStates']+=1
    for f in [Fraction(1,4),Fraction(1,16),Fraction(1,64),Fraction(1,256)]:
        d=int(2/math.sqrt(float(f)))
        field=aperture(float(f),d,0,n=4000);independent=on_axis_series(f,d)
        assert abs(field-independent)<3e-12
        assert abs(field+2)<=math.pi*math.sqrt(float(f))+1e-12
        assert abs(field-2)>=4-math.pi*math.sqrt(float(f))-1e-12
        counts['independentSeriesPaths']+=1
    return counts


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Interference research checks passed: '+json.dumps(result,sort_keys=True))
