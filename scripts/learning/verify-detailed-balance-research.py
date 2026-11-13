"""Independent path enumeration, reversible Poisson solves and time-average moments."""
from fractions import Fraction as F
from itertools import product
import json

def solve(a,b):
    n=len(b);r=[list(row)+[v] for row,v in zip(a,b)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if r[i][j]);r[j],r[pivot]=r[pivot],r[j]
        d=r[j][j];r[j]=[x/d for x in r[j]]
        for i in range(n):
            if i!=j:
                d=r[i][j];r[i]=[x-d*y for x,y in zip(r[i],r[j])]
    return [row[-1] for row in r]

def cycle_law(n,p,q):
    law={0:F(1)}
    for _ in range(n):
        new={}
        for d,m in law.items():
            for z,prob in [(1,p),(-1,q),(0,1-p-q)]:new[d+z]=new.get(d+z,F(0))+m*prob
        law=new
    return law

def kernel(weights,forward,backward,stay,barker=False):
    pi=[F(w,sum(weights)) for w in weights];den=forward+backward+stay
    q=[[F(stay if i==j else forward if (j-i)%3==1 else backward,den) for j in range(3)] for i in range(3)]
    p=[[F(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i!=j:
                ratio=pi[j]*q[j][i]/(pi[i]*q[i][j]);p[i][j]=q[i][j]*(ratio/(1+ratio) if barker else min(F(1),ratio))
        p[i][i]=1-sum(p[i])
    return pi,p

def poisson_variance(pi,p,phi):
    g=solve([[F(i==j)-p[i][j]+pi[j] for j in range(3)] for i in range(3)],phi)
    assert sum(x*y for x,y in zip(pi,g))==0
    assert [g[i]-sum(p[i][j]*g[j] for j in range(3)) for i in range(3)]==phi
    variance=sum(pi[i]*phi[i]**2 for i in range(3))
    return 2*sum(pi[i]*phi[i]*g[i] for i in range(3))-variance

def binary_error(n,a,b,u):
    pi=a/(a+b);lam=1-a-b;means=[u]
    for _ in range(n-1):means.append(a+lam*means[-1])
    vs=[x*(1-x) for x in means];bias=sum(means)/n-pi
    variance=(sum(vs)+2*sum(vs[i]*lam**(j-i) for i in range(n) for j in range(i+1,n)))/n**2
    return bias,variance,variance+bias*bias

def main():
    cycle_checks=singular=poisson_checks=binary_checks=lazy_checks=resets=0
    for p,q in [(F(1,2),F(1,4)),(F(1,4),F(1,2)),(F(2,5),F(2,5)),(F(3,5),F(1,10)),(F(2,3),F(1,3))]:
        r=1-p-q;transition=[[r if i==j else p if (j-i)%3==1 else q for j in range(3)] for i in range(3)]
        for n in range(1,6):
            bins={}
            for path in product(range(3),repeat=n+1):
                mass=F(1,3);reverse=F(1,3);d=0
                for i,j in zip(path,path[1:]):mass*=transition[i][j];reverse*=transition[j][i];d+=0 if i==j else 1 if (j-i)%3==1 else -1
                if mass:
                    assert reverse and mass/reverse==(p/q)**d
                    bins[d]=bins.get(d,F(0))+mass
            law=cycle_law(n,p,q);assert {d:m for d,m in law.items() if m}==bins
            assert sum(bins.values())==1
            mean=sum(m*d for d,m in bins.items());variance=sum(m*(d-mean)**2 for d,m in bins.items())
            assert mean==n*(p-q) and variance==n*(p+q-(p-q)**2)
            for d,m in bins.items():assert m==bins[-d]*(p/q)**d
            for k in (-2,-1,0,1):assert sum(m*(p/q)**(k*d) for d,m in bins.items())==(r+p*(p/q)**k+q*(p/q)**(-k))**n
            cycle_checks+=1
    for r in (F(0),F(1,4),F(3,4)):
        for n in range(1,6):
            p=1-r;finite_integral=F(0)
            for path in product(range(3),repeat=n+1):
                mass=reverse=F(1,3)
                for i,j in zip(path,path[1:]):
                    mass*=r if i==j else p if (j-i)%3==1 else 0
                    reverse*=r if i==j else p if (i-j)%3==1 else 0
                if mass:finite_integral+=reverse
            assert finite_integral==r**n;singular+=1

    for weights in [(1,2,4),(1,1,1),(3,1,2),(2,5,1)]:
        for forward,backward,stay in [(1,1,0),(2,1,0),(1,3,1),(4,1,2),(2,3,4),(1,2,2)]:
            pi,m=kernel(weights,forward,backward,stay);_,b=kernel(weights,forward,backward,stay,True)
            for i in range(3):
                for j in range(3):
                    assert pi[i]*m[i][j]==pi[j]*m[j][i] and pi[i]*b[i][j]==pi[j]*b[j][i]
                    if i!=j:assert m[i][j]>=b[i][j]
            for values in [(1,0,0),(0,1,0),(0,0,1),(-1,0,2)]:
                mean=sum(x*y for x,y in zip(pi,values));phi=[F(x)-mean for x in values]
                vm=poisson_variance(pi,m,phi);vb=poisson_variance(pi,b,phi);assert 0<vm<=vb
                # Independent covariance power series, not the linear-system formula.
                for p,expected in [(m,vm),(b,vb)]:
                    pf=list(map(float,phi));vf=float(sum(pi[i]*phi[i]**2 for i in range(3)));series=vf
                    for _ in range(2048):
                        pf=[sum(float(p[i][j])*pf[j] for j in range(3)) for i in range(3)]
                        series+=2*sum(float(pi[i]*phi[i])*pf[i] for i in range(3))
                    assert abs(series-float(expected))<3e-10*max(1,abs(float(expected)))
                poisson_checks+=1

    for a,b in [(F(9,10),F(9,10)),(F(1,5),F(3,5)),(F(0),F(1,2)),(F(1,2),F(0)),(F(1),F(1)),(F(1,2),F(1,2))]:
        transition=[[1-a,a],[b,1-b]];pi=a/(a+b)
        for u in (F(0),F(3,10),F(1)):
            for n in range(1,9):
                mean=second=F(0)
                for path in product((0,1),repeat=n):
                    mass=u if path[0] else 1-u
                    for i,j in zip(path,path[1:]):mass*=transition[i][j]
                    avg=F(sum(path),n);mean+=mass*avg;second+=mass*avg*avg
                bias,var,mse=binary_error(n,a,b,u)
                assert bias==mean-pi and var==second-mean*mean and mse==second-2*pi*mean+pi*pi
                if a==b==1:assert mse==(F(0) if n%2==0 else F(1,4*n*n))
                binary_checks+=1
    for a,b in [(F(9,10),F(9,10)),(F(3,4),F(1,2)),(F(1,4),F(1,2))]:
        pi=a/(a+b);v=pi*(1-pi);lam=1-a-b;previous=None
        for delta in (F(0),F(1,4),F(1,2),F(3,4)):
            ad,bd=(1-delta)*a,(1-delta)*b;ld=1-ad-bd
            assert ld==delta+(1-delta)*lam and ad/(ad+bd)==pi
            coefficient=v*(1+ld)/(1-ld)
            if previous is not None:assert coefficient>previous
            previous=coefficient;lazy_checks+=1
        if lam<0:
            delta=-lam/(1-lam);assert (1-delta)*a==pi and (1-delta)*b==1-pi;resets+=1
    return {'status':'passed','cyclePathPreparations':cycle_checks,'oneWaySupportPreparations':singular,'poissonObservableComparisons':poisson_checks,'enumeratedMeanPreparations':binary_checks,'lazyKernelPreparations':lazy_checks,'exactResetKernels':resets}

if __name__=='__main__':print(json.dumps(main(),indent=2))
