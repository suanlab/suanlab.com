"""Independent exact-table, spin enumeration and rational linear-system checks."""
from fractions import Fraction as F
from itertools import product
from math import exp, fsum, log, tanh, cos, sin, pi, prod, sqrt
import json

STATES=[(-1,-1),(-1,1),(1,-1),(1,1)]

def solve(a,b):
    n=len(b);r=[list(row)+[v] for row,v in zip(a,b)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if r[i][j]);r[j],r[pivot]=r[pivot],r[j]
        d=r[j][j];r[j]=[x/d for x in r[j]]
        for i in range(n):
            if i!=j:
                d=r[i][j];r[i]=[x-d*y for x,y in zip(r[i],r[j])]
    return [row[-1] for row in r]

def covariance(counts):
    ps=[F(x,sum(counts)) for x in counts];ts=[(a,b,a*b) for a,b in STATES]
    means=[sum(p*t[j] for p,t in zip(ps,ts)) for j in range(3)]
    mat=[[sum(p*(t[i]-means[i])*(t[j]-means[j]) for p,t in zip(ps,ts)) for j in range(3)] for i in range(3)]
    return ps,means,mat

def determinant3(m):
    return sum(m[0][j]*(m[1][(j+1)%3]*m[2][(j+2)%3]-m[1][(j+2)%3]*m[2][(j+1)%3]) for j in range(3))

def spectrum(n,k):
    t=tanh(k)
    return [(1-t*t)/(1-2*t*cos(2*pi*l/n)+t*t)*(1-t**n)/(1+t**n) for l in range(n)]

def gls(n,rho):
    den=n*(1-rho)+2*rho
    return [1/den]+[(1-rho)/den]*(n-2)+[1/den]

def main():
    tables=0
    for counts in product((1,2,5),repeat=4):
        ps,means,mat=covariance(counts)
        assert determinant3(mat)==256*prod(ps)>0
        k=log(float(ps[0]*ps[3]/(ps[1]*ps[2])))/4
        h1=log(float(ps[2]*ps[3]/(ps[0]*ps[1])))/4
        h2=log(float(ps[1]*ps[3]/(ps[0]*ps[2])))/4
        weights=[exp(k*a*b+h1*a+h2*b) for a,b in STATES]
        assert max(abs(w/sum(weights)-float(p)) for w,p in zip(weights,ps))<3e-15
        eff=[[mat[i][j]-mat[i][2]*mat[2][j]/mat[2][2] for j in range(2)] for i in range(2)]
        for a in [(F(1),F(0)),(F(0),F(1)),(F(1),F(-2))]:
            rhs=[*a,F(0)]
            full=solve(mat,rhs)[:2];reduced=solve(eff,list(a))
            assert full==reduced
            known=solve([row[:2] for row in mat[:2]],list(a))
            assert sum(x*(y-z) for x,y,z in zip(a,full,known))>=0
            # The projected score is orthogonal to the coupling score.
        projected=[[F(a)-means[0]-mat[0][2]/mat[2][2]*(F(a*b)-means[2]),F(b)-means[1]-mat[1][2]/mat[2][2]*(F(a*b)-means[2])] for a,b in STATES]
        for i in range(2):
            assert sum(p*s[i]*(a*b-means[2]) for p,s,(a,b) in zip(ps,projected,STATES))==0
        tables+=1

    rings=0
    for n in range(3,11):
        for k in (-1.2,-.4,0.,.7):
            states=list(product((-1,1),repeat=n));weights=[exp(k*sum(s[j]*s[(j+1)%n] for j in range(n))) for s in states];z=fsum(weights)
            predicted=spectrum(n,k)
            for l in range(n):
                q=2*pi*l/n
                actual=fsum(w*abs(sum(s[j]*complex(cos(q*j),-sin(q*j)) for j in range(n)))**2/n for s,w in zip(states,weights))/z
                assert abs(actual-predicted[l])<2e-12*max(1,actual)
                norm=1/sqrt(n) if l==0 or 2*l==n else sqrt(2/n)
                realvar=fsum(w*(norm*sum(cos(q*j)*s[j] for j in range(n)))**2 for s,w in zip(states,weights))/z
                assert abs(realvar-predicted[l])<2e-12*max(1,actual)
            assert abs(fsum(predicted)-n)<2e-12*n and min(predicted)>0
            rings+=1
    cold=0
    for n in range(3,18):
        # Enumerate ground states independently of the correlation formula.
        ground=[];best=n+1
        for s in product((-1,1),repeat=n):
            bonds=sum(s[j]*s[(j+1)%n] for j in range(n))
            if bonds<best:best=bonds;ground=[s]
            elif bonds==best:ground.append(s)
        assert len(ground)==(2*n if n%2 else 2)
        for l in range(n):
            q=2*pi*l/n
            value=sum(abs(sum(s[j]*complex(cos(q*j),-sin(q*j)) for j in range(n)))**2/n for s in ground)/len(ground)
            expected=1/(n*cos(q/2)**2) if n%2 else n if 2*l==n else 0
            assert abs(value-expected)<5e-12*max(1,expected)
        cold+=1

    precisions=0
    for n in range(2,15):
        for rho in (F(-9,10),F(-1,2),F(0),F(2,5),F(9,10)):
            sigma=[[rho**abs(i-j) for j in range(n)] for i in range(n)]
            solved=solve(sigma,[F(1)]*n);d=sum(solved);w=[x/d for x in solved]
            assert w==gls(n,rho)
            var=sum(w[i]*w[j]*sigma[i][j] for i in range(n) for j in range(n))
            assert var==1/d==(1+rho)/(n*(1-rho)+2*rho)
            ordinary=sum(map(sum,sigma))/n**2
            delta=[F(1,n)-x for x in w]
            assert ordinary-var==sum(delta[i]*delta[j]*sigma[i][j] for i in range(n) for j in range(n))>=0
            # Recover every inverse column by elimination, independently of the tridiagonal formula.
            for j in range(n):
                col=solve(sigma,[F(i==j) for i in range(n)])
                for i,value in enumerate(col):
                    expected=(1 if i in (0,n-1) else 1+rho*rho) if i==j else -rho if abs(i-j)==1 else 0
                    assert value==expected/(1-rho*rho)
            precisions+=1
    endpoints=0
    for n in range(2,33):
        w=[F(1,2*(n-1))]+[F(1,n-1)]*(n-2)+[F(1,2*(n-1))]
        assert sum(w)==1 and sum(v*(-1)**i for i,v in enumerate(w))==0
        endpoints+=1
    return {'status':'passed','positiveTables':tables,'enumeratedFiniteRings':rings,'enumeratedColdRings':cold,'rationalPrecisionPreparations':precisions,'alternatingEndpointLengths':endpoints}

if __name__=='__main__':
    print(json.dumps(main(),indent=2))
