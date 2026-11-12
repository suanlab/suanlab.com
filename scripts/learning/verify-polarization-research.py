"""Independent spectral polarization and calibrated tomography checks; stdlib only."""
import cmath
import itertools
import json
import math
import sys


def simpson(f,a,b,n=16000):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))


def characteristic(s,n=16000):
    return simpson(lambda q:math.exp(-q*q/2)/math.sqrt(2*math.pi)*cmath.exp(1j*s*q),-9,9,n)


def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def transpose(a):return list(map(list,zip(*a)))
def dagger(a):return [[x.conjugate() for x in col] for col in zip(*a)]
def outer(e):return [[x*y.conjugate() for y in e] for x in e]
def stokes(c):return [float((c[0][0]+c[1][1]).real),float((c[0][0]-c[1][1]).real),2*c[0][1].real,-2*c[0][1].imag]
def degree(c):
    s=stokes(c)
    return math.sqrt(sum(x*x for x in s[1:]))/s[0]
def retarder(theta,delta):
    c,s=math.cos(theta),math.sin(theta);r=[[c,-s],[s,c]]
    return mm(mm(r,[[1,0],[0,cmath.exp(1j*delta)]]),transpose(r))
def inverse(a):
    n=len(a);b=[list(row)+[float(i==j) for j in range(n)] for i,row in enumerate(a)]
    for j in range(n):
        k=max(range(j,n),key=lambda k:abs(b[k][j]));assert abs(b[k][j])>1e-15
        b[j],b[k]=b[k],b[j];pivot=b[j][j];b[j]=[x/pivot for x in b[j]]
        for k in range(n):
            if k!=j:
                factor=b[k][j];b[k]=[x-factor*y for x,y in zip(b[k],b[j])]
    return [row[n:] for row in b]


def eigenvalues(a):
    b=[row[:] for row in a];n=len(b)
    for _ in range(100):
        p,q=max(((i,j) for i in range(n) for j in range(i+1,n)),key=lambda ij:abs(b[ij[0]][ij[1]]))
        if abs(b[p][q])<1e-15:break
        angle=.5*math.atan2(2*b[p][q],b[q][q]-b[p][p]);c,s=math.cos(angle),math.sin(angle)
        rotation=[[float(i==j) for j in range(n)] for i in range(n)]
        rotation[p][p]=rotation[q][q]=c;rotation[p][q]=s;rotation[q][p]=-s
        b=mm(mm(transpose(rotation),b),rotation)
    assert max(abs(b[i][j]) for i in range(n) for j in range(n) if i!=j)<1e-12
    return sorted(b[i][i] for i in range(n))


def instrument(delta):
    axes=[[1,0,0],[0,1,0],[math.cos(delta),0,math.sin(delta)]]
    return [[.5]+[sign*x/2 for x in axis] for axis in axes for sign in [1,-1]]
def predict(a,s):return [sum(x*y for x,y in zip(row,s)) for row in a]
def reconstruct(y,delta):
    return [sum(y)/3,y[0]-y[1],y[2]-y[3],((y[4]-y[5])-math.cos(delta)*(y[0]-y[1]))/math.sin(delta)]
def close(a,b,tol=1e-10):assert abs(a-b)<tol*(1+abs(b)),(a,b)


def verify():
    counts=dict(gaussianSpectra=0,spectralRefinements=0,pureFrequencyStates=0,frequencyCompensations=0,fixedRetarderStates=0,finiteSpectrumBounds=0,instrumentSpectra=0,reconstructionBounds=0,noiseEnumerations=0,noiseCovariances=0,calibrationStates=0,calibrationDerivatives=0,positiveReadingCounterexamples=0,blindPairs=0)
    for s in [-2,-1,-.5,0,.5,1,2,4]:
        chi=characteristic(s);close(chi,math.exp(-s*s/2),2e-13)
        for delta in [-.7,0,math.pi/2]:
            cross=cmath.exp(-1j*delta)*chi.conjugate()/2;c=[[.5,cross],[cross.conjugate(),.5]]
            expected=[1,0,math.exp(-s*s/2)*math.cos(delta),math.exp(-s*s/2)*math.sin(delta)]
            for x,y in zip(stokes(c),expected):close(x,y)
            close(degree(c),math.exp(-s*s/2));counts['gaussianSpectra']+=1
        if s in [.5,1,2,4]:
            close(chi,characteristic(s,32000),2e-13);counts['spectralRefinements']+=1
    for s in [.2,1,2]:
        for delta in [-.7,0,math.pi/2]:
            for q in [-3,-1,0,.6,2]:
                phase=delta+s*q;e=[1/math.sqrt(2),cmath.exp(1j*phase)/math.sqrt(2)];c=outer(e)
                close(degree(c),1);close(sum(abs(x)**2 for x in e),1);counts['pureFrequencyStates']+=1
                out=mm([[1,0],[0,cmath.exp(-1j*phase)]],[[e[0]],[e[1]]]);close(out[0][0],1/math.sqrt(2));close(out[1][0],1/math.sqrt(2));counts['frequencyCompensations']+=1
    for s in [.3,1,2]:
        for delta in [0,.7]:
            z=math.exp(-s*s/2)*cmath.exp(-1j*delta)/2;c=[[.5,z],[z.conjugate(),.5]]
            for theta in [-.4,0,.7]:
                for retardance in [.2,math.pi/2,math.pi]:
                    j=retarder(theta,retardance);close(degree(mm(mm(j,c),dagger(j))),degree(c));counts['fixedRetarderStates']+=1
    spectra=[([-1,1],[.5,.5]),([-math.sqrt(3),0,math.sqrt(3)],[1/6,2/3,1/6]),([-math.sqrt(2),1/math.sqrt(2)],[1/3,2/3])]
    for nodes,weights in spectra:
        close(sum(x*w for x,w in zip(nodes,weights)),0);close(sum(x*x*w for x,w in zip(nodes,weights)),1)
        for s in [0,.2,.6,1,math.pi,2*math.pi]:
            chi=sum(w*cmath.exp(1j*s*x) for x,w in zip(nodes,weights));assert abs(chi)+1e-12>=max(0,1-s*s/2)
            fields=[[1/math.sqrt(2),cmath.exp(1j*(.7+s*x))/math.sqrt(2)] for x in nodes]
            c=[[sum(w*outer(e)[i][j] for e,w in zip(fields,weights)) for j in range(2)] for i in range(2)]
            close(c[0][1],cmath.exp(-.7j)*chi.conjugate()/2);close(degree(c),abs(chi));counts['finiteSpectrumBounds']+=1
    states=[[1,0,0,0],[1,.3,.2,.4],[2,.8,-.6,1.2],[1,0,0,1]]
    for deg in [.5,1,5,10,30,60,90]:
        delta=math.radians(deg);c,s=math.cos(delta),math.sin(delta);a=instrument(delta);gram=mm(transpose(a),a)
        expected=sorted([1.5,.5,(1+c)/2,(1-c)/2]);actual=eigenvalues(gram)
        for x,y in zip(actual,expected):close(x,y,1e-12)
        condition=math.sqrt(max(actual)/min(actual));close(condition,math.sqrt(3/(1-abs(c))),1e-8);counts['instrumentSpectra']+=1
        b=mm(inverse(gram),transpose(a));identity=mm(b,a)
        for i in range(4):
            for j in range(4):close(identity[i][j],float(i==j),1e-9)
        for state in states:
            exact=predict(a,state);assert min(exact)>=-1e-14
            for j in range(6):
                for sign in [-1,1]:
                    eta=[0]*6;eta[j]=sign*1e-4;y=[x+e for x,e in zip(exact,eta)];out=reconstruct(y,delta)
                    via_matrix=predict(b,y)
                    for x,z in zip(out,via_matrix):close(x,z,1e-8)
                    error=math.sqrt(sum((x-z)**2 for x,z in zip(out,state)));assert error<=math.sqrt(2/(1-abs(c)))*1e-4+1e-11;counts['reconstructionBounds']+=1
        errors=[]
        for signs in itertools.product([-1,1],repeat=6):
            errors.append(reconstruct(signs,delta));counts['noiseEnumerations']+=1
        cov=[[sum(e[i]*e[j] for e in errors)/64 for j in range(4)] for i in range(4)]
        expected=[[2/3,0,0,0],[0,2,0,-2*c/s],[0,0,2,0],[0,-2*c/s,0,2*(1+c*c)/(s*s)]]
        for i in range(4):
            for j in range(4):close(cov[i][j],expected[i][j],1e-10)
        counts['noiseCovariances']+=1
    for deg in [1,5,30,90]:
        delta=math.radians(deg)
        for state in states:
            for shift in [-.5,.5]:
                offset=math.radians(shift);out=reconstruct(predict(instrument(delta+offset),state),delta)
                expected=(math.cos(delta+offset)-math.cos(delta))/math.sin(delta)*state[1]+math.sin(delta+offset)/math.sin(delta)*state[3]
                close(out[3],expected);counts['calibrationStates']+=1
            h=1e-6;plus=reconstruct(predict(instrument(delta+h),state),delta)[3];minus=reconstruct(predict(instrument(delta-h),state),delta)[3]
            close((plus-minus)/(2*h),-state[1]+math.cos(delta)/math.sin(delta)*state[3],1e-8);counts['calibrationDerivatives']+=1
    for deg in [.5,1,5]:
        delta=math.radians(deg);y=[.5,.5,.5,.5,.51,.49];out=reconstruct(y,delta)
        assert min(y)>0 and all(abs(y[j]+y[j+1]-1)<1e-15 for j in [0,2,4]);close(out[3],.02/math.sin(delta))
        assert (abs(out[3])>out[0])==(deg<5);counts['positiveReadingCounterexamples']+=1
    a=instrument(0)
    for circular in [.3,1]:
        assert predict(a,[1,0,0,circular])==predict(a,[1,0,0,-circular]);counts['blindPairs']+=1
    return counts


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Polarization research checks passed: '+json.dumps(result,sort_keys=True))
