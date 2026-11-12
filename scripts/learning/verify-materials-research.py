"""Independent finite dipole layers and passive electric-sheet scattering.
Standard library only. Includes weak limits, incompatible geometry, lossless sheets,
reactive loading, unequal ports, coherent/dark inputs and zero-input ratios.
"""
from fractions import Fraction as F
from itertools import product
import argparse,cmath,json,math

def near(a,b,tol=2e-11):assert math.isfinite(abs(a)) and abs(a-b)<=tol*(1+abs(b)),(a,b)
def layer(p,ell,eps):
    if ell<=0 or eps<=0:raise ValueError('Finite layer needs positive thickness and permittivity')
    sigma=p/ell;E=-sigma/eps
    return dict(sigma=sigma,E=E,jump=-E*ell,energy=eps*E*E*ell/2)
def sheet(y1,y2,G,X=0):
    if y1<=0 or y2<=0 or G<0:raise ValueError('Passive sheet requires positive port admittances and nonnegative conductance')
    D=y1+y2+G+1j*X
    return (y1-y2-G-1j*X)/D,2*y1/D

def boundary_solve(y1,y2,Ys,a,b):
    # Independent 2x2 elimination: c-d=b-a;
    # (y1+Ys)c+y2 d=(y1-Ys)a+y2 b.
    rhs1=b-a;rhs2=(y1-Ys)*a+y2*b
    c=(rhs2+y2*rhs1)/(y1+y2+Ys);d=c-rhs1
    return c,d

def verify():
    counts=dict(finiteLayerIdentities=0,polynomialWeakMoments=0,weakTaylorBounds=0,potentialGradientChecks=0,energyScalingLimits=0,shapeEnergyBounds=0,invalidLayerRejections=0,exactOneBeamBudgets=0,absorptionBoundChecks=0,complexBoundaryBudgets=0,independentCycleIntegrals=0,coherentPerfectInputs=0,darkInputs=0,powerNormalizedReciprocity=0,symmetricPhaseExamples=0,coherenceEnsembles=0,invalidSheetRejections=0,zeroInputCases=0,absorptionOperators=0,brightDarkEigenPairs=0,fluctuatingAmplitudeEnsembles=0)
    for p,ell,eps in product(map(F,[-2,0,F(1,3),1]),map(F,[1,F(1,4),F(1,16),F(1,64)]),map(F,[F(1,2),1,2])):
        r=layer(p,ell,eps)
        # Superpose the actual lower negative / upper positive sheet fields.
        for z in [-ell,-ell/4,F(0),ell/4,ell]:
            sg=lambda x:F(1) if x>0 else F(-1) if x<0 else F(0)
            E=r['sigma']*(sg(z-ell/2)-sg(z+ell/2))/(2*eps)
            assert E==(r['E'] if abs(z)<ell/2 else 0)
        assert r['sigma']*(-ell/2)*(-1)+r['sigma']*(ell/2)==p
        assert r['jump']==p/eps and r['energy']==p*p/(2*eps*ell);counts['finiteLayerIdentities']+=1
        for k in range(7):
            moment=r['sigma']*((ell/2)**k-(-ell/2)**k)
            assert moment==(F(0) if k%2==0 else p*ell**(k-1)/2**(k-1));counts['polynomialWeakMoments']+=1
        # phi=z+z^3: centered quotient differs from phi'(0) by ell^2/4;
        # independent C^3 estimate uses sup|phi'''|=6.
        phi=lambda z:z+z**3
        value=r['sigma']*(phi(ell/2)-phi(-ell/2))
        assert abs(value-p)<=abs(p)*ell*ell*6/24;counts['weakTaylorBounds']+=1
        V=lambda z:r['jump']*max(F(0),min(F(1),(z+ell/2)/ell))
        h=ell/32
        assert -(V(h)-V(-h))/(2*h)==r['E'];counts['potentialGradientChecks']+=1
        # Fixed sheet charge sigma=p at reference thickness 1, rather than fixed moment.
        other=layer(p*ell,ell,eps)
        assert other['jump']==p*ell/eps and other['energy']==p*p*ell/(2*eps);counts['energyScalingLimits']+=1
    # Piecewise constant E on equal subdivisions: same integral, independent variance.
    for values in [[F(-2),F(-1),F(-3)],[F(-1)]*4,[F(1),F(-1)],[F(0)]*3]:
        n=len(values);ell=F(3,5);eps=F(2);jump=-ell*sum(values)/n
        U=eps*ell*sum(e*e for e in values)/(2*n);lower=eps*jump*jump/(2*ell)
        assert U>=lower and (U==lower)==(len(set(values))==1);counts['shapeEnergyBounds']+=1
    for ell,eps in [(0,1),(-1,1),(1,0),(1,-1)]:
        try:layer(F(1),F(ell),F(eps))
        except ValueError:counts['invalidLayerRejections']+=1
        else:raise AssertionError('Invalid finite layer accepted')
    for y1,y2,G in product(map(F,[F(1,2),1,2]),map(F,[F(1,2),1,2]),map(F,[0,F(1,2),1,2,4,10**6])):
        a=y1+y2;D=a+G;r=(y1-y2-G)/D;t=2*y1/D
        R=r*r;T=y2/y1*t*t;A=G/y1*t*t
        assert R+T+A==1 and A>=0 and T>=0 and R>=0;counts['exactOneBeamBudgets']+=1
        assert y1/a-A==y1*(G-a)**2/(a*(G+a)**2);counts['absorptionBoundChecks']+=1
    inputs=[(1,0),(0,1),(1,1),(1,-1),(1,1j)]
    for y1,y2,G,X in product([.5,1,2],[.5,1,2],[0,.5,1,2,4],[-2,0,2]):
        Ys=complex(G,X);D=y1+y2+Ys
        r,t=sheet(y1,y2,G,X)
        # Power-normalized transmission is reciprocal even for unequal ports.
        near(t*math.sqrt(y2/y1),(2*y2/D)*math.sqrt(y1/y2));counts['powerNormalizedReciprocity']+=1
        for a,b in inputs:
            c,d=boundary_solve(y1,y2,Ys,a,b);E=a+c
            near(E,b+d);near(y1*(a-c)-y2*(d-b),Ys*E)
            incoming=(y1*abs(a)**2+y2*abs(b)**2)/2
            outgoing=(y1*abs(c)**2+y2*abs(d)**2)/2
            absorbed=G*abs(E)**2/2
            near(incoming-outgoing,absorbed);assert outgoing<=incoming+1e-12;counts['complexBoundaryBudgets']+=1
        # Construct the power-normalized matrix from independent boundary responses.
        col1=boundary_solve(y1,y2,Ys,1/math.sqrt(y1),0)
        col2=boundary_solve(y1,y2,Ys,0,1/math.sqrt(y2))
        M=[[math.sqrt(y1)*col1[0],math.sqrt(y1)*col2[0]],
           [math.sqrt(y2)*col1[1],math.sqrt(y2)*col2[1]]]
        a_sum=y1+y2;v=[math.sqrt(y1),math.sqrt(y2)];loss=4*a_sum*G/abs(D)**2
        for i in range(2):
            for j in range(2):
                actual=(1 if i==j else 0)-sum(M[k][i].conjugate()*M[k][j] for k in range(2))
                near(actual,loss*v[i]*v[j]/a_sum)
        assert -1e-12<=loss<=1+1e-12;counts['absorptionOperators']+=1
        bright=v;dark=[v[1],-v[0]];eigen=(a_sum-Ys)/D
        for i in range(2):
            near(sum(M[i][j]*bright[j] for j in range(2)),eigen*bright[i])
            near(sum(M[i][j]*dark[j] for j in range(2)),-dark[i])
        counts['brightDarkEigenPairs']+=1
        # Actual real phasor fields, separately integrated around one carrier period.
        a,b=1,1j;c,d=boundary_solve(y1,y2,Ys,a,b);E=a+c;K=Ys*E
        n=128
        mean=sum((E*cmath.exp(-2j*math.pi*j/n)).real*(K*cmath.exp(-2j*math.pi*j/n)).real for j in range(n))/n
        near(mean,G*abs(E)**2/2);counts['independentCycleIntegrals']+=1
    for y1,y2 in product([.5,1,2],[.5,1,2]):
        G=y1+y2;c,d=boundary_solve(y1,y2,G,1,1)
        near(c,0);near(d,0);counts['coherentPerfectInputs']+=1
        c,d=boundary_solve(y1,y2,G,y2,-y1)
        near(y2+c,0);near(-y1+d,0);counts['darkInputs']+=1
    for p,phasor in [(0,1),(.5,1j),(1,-1),(1.5,-1j)]:
        c,d=boundary_solve(1,1,2,1,phasor)
        absorbed=1-(abs(c)**2+abs(d)**2)/2
        near(absorbed,abs(1+phasor)**2/4);counts['symmetricPhaseExamples']+=1
    for phases in [[1,1,1,1],[1,-1,1,-1],[1,1j,-1,-1j],[1,1,1,-1]]:
        fractions=[]
        for b in phases:
            c,d=boundary_solve(1,1,2,1,b);fractions.append(1-(abs(c)**2+abs(d)**2)/2)
        gamma=sum(complex(b).conjugate() for b in phases)/len(phases)
        near(sum(fractions)/len(fractions),(1+gamma.real)/2);counts['coherenceEnsembles']+=1
    # Equal mean-square inputs but varying total input power, including a zero run.
    ensemble=[(2,1),(1,2),(1,-1),(0,0)];Pin=Pabs=cross=Ileft=Iright=0
    ratios=[]
    for a,b in ensemble:
        c,d=boundary_solve(1,1,2,a,b);incoming=(abs(a)**2+abs(b)**2)/2
        absorbed=incoming-(abs(c)**2+abs(d)**2)/2
        Pin+=incoming;Pabs+=absorbed;cross+=a*complex(b).conjugate();Ileft+=abs(a)**2;Iright+=abs(b)**2
        if incoming:ratios.append(absorbed/incoming)
    near(Ileft,Iright);near(Pabs/Pin,(1+(cross/Ileft).real)/2)
    assert abs(sum(ratios)/len(ratios)-Pabs/Pin)>.1;counts['fluctuatingAmplitudeEnsembles']+=1
    for args in [(0,1,1),(1,0,1),(1,1,-1)]:
        try:sheet(*args)
        except ValueError:counts['invalidSheetRejections']+=1
        else:raise AssertionError('Invalid passive sheet accepted')
    for G in [0,1,2]:
        c,d=boundary_solve(1,1,G,0,0);assert c==d==0;counts['zeroInputCases']+=1
    return counts

def references():
    static=[dict(factor=factor,x=x,potential=-.5*x) for factor,x in product([1,2,4],[-1,0,1])]
    waves=[]
    for eps2,mu2 in product([.25,1,4],[.25,1,4]):
        y1=1.;y2=math.sqrt(eps2/mu2);D=y1+y2
        waves.append(dict(eps2=eps2,mu2=mu2,y1=y1,y2=y2,r=(y1-y2)/D,t=2*y1/D,impliedTransparentG=y1-y2))
    return dict(static=static,waves=waves)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');p.add_argument('--references',action='store_true');args=p.parse_args()
    print(json.dumps(references() if args.references else verify(),sort_keys=True))
