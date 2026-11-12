"""Independent full-grid Yee updates and exact periodic Gauss constraint checks.
Standard library only; no product imports. Boundary cases are retained explicitly.
"""
from fractions import Fraction as F
import argparse,json,math

def step(a,b,theta):
    b=b+theta*a
    return a-theta*b,b

def invariant(a,b,t):return a*a+b*b+t*a*b

def near(a,b,tol=2e-8):assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)),(a,b)
def mv(A,x):return [sum(a*b for a,b in zip(row,x)) for row in A]
def transpose(A):return list(map(list,zip(*A)))
def matmul(A,B):return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]
def solve(A,b):
    a=[list(map(F,row))+[F(v)] for row,v in zip(A,b)];n=len(a)
    for j in range(n):
        pivot=next(i for i in range(j,n) if a[i][j])
        a[j],a[pivot]=a[pivot],a[j];q=a[j][j];a[j]=[x/q for x in a[j]]
        for i in range(n):
            if i!=j:
                q=a[i][j];a[i]=[x-q*y for x,y in zip(a[i],a[j])]
    return [row[-1] for row in a]

def divergence(n,dx=F(1)):
    return [[(F(i==j)-F(j==(i-1)%n))/dx for j in range(n)] for i in range(n)]
def project(E,rho,eps=F(1),dx=F(1)):
    n=len(E)
    if sum(rho):raise ValueError('Periodic Gauss law requires zero net charge')
    G=divergence(n,dx);GT=transpose(G);H=matmul(G,GT);g=[eps*a-b for a,b in zip(mv(G,E),rho)]
    augmented=[row+[F(1)] for row in H]+[[F(1)]*n+[F(0)]]
    p=solve(augmented,g+[F(0)])[:-1]
    correction=[-x/eps for x in mv(GT,p)]
    return [a+b for a,b in zip(E,correction)],correction,p

def verify():
    c=dict(exactModalInvariants=0,stableModeBounds=0,criticalClosedForms=0,supercriticalSpectra=0,fullGridTrajectories=0,dispersionPairs=0,gaussProjections=0,independentPrefixSolutions=0,energyIdentities=0,continuityDefects=0,effectiveCurrentChecks=0,incompatibleChargeRejections=0)
    for theta in map(F,[0,F(1,4),F(1,2),1,F(3,2),F(19,10),2,F(9,4)]):
        for a0,b0 in [(F(1),F(0)),(F(0),F(1)),(F(2),F(-1))]:
            a,b=a0,b0;I0=invariant(a,b,theta)
            for n in range(41):
                assert invariant(a,b,theta)==I0;c['exactModalInvariants']+=1
                if theta<2:
                    assert a*a+b*b<=I0/(1-theta/2);c['stableModeBounds']+=1
                a,b=step(a,b,theta)
    for n in range(41):
        a,b=F(1),F(0)
        for _ in range(n):a,b=step(a,b,F(2))
        assert a==(-1)**n*(2*n+1) and b==(-1)**(n+1)*2*n
        assert invariant(a,b,F(2))==1 and a*a+b*b==8*n*n+4*n+1;c['criticalClosedForms']+=1
    for t in [2.01,2.1,2.25,3]:
        tr=2-t*t;rho=(-tr+math.sqrt(tr*tr-4))/2
        assert rho>1;near(rho*rho+tr*rho+1,0);c['supercriticalSpectra']+=1
    for N in [8,12,16]:
        for m in [1,N//4,N//2]:
            q=2*math.pi*m/N
            for s in [.5,.9,1,1.1]:
                a,b=.7,-.2;E=[a*math.cos(q*j) for j in range(N)];B=[b*math.sin(q*(j+.5)) for j in range(N)];theta=2*s*math.sin(q/2)
                for _ in range(10):
                    B=[B[j]-s*(E[(j+1)%N]-E[j]) for j in range(N)]
                    E=[E[j]-s*(B[j]-B[(j-1)%N]) for j in range(N)]
                    a,b=step(a,b,theta)
                    for j in range(N):near(E[j],a*math.cos(q*j));near(B[j],b*math.sin(q*(j+.5)))
                c['fullGridTrajectories']+=1
    for s in [.25,.5,.9,1]:
        for fraction in [.1,.25,.5,.75,.99,1]:
            q=math.pi*fraction;nu=2*math.asin(s*math.sin(q/2));near(math.cos(nu),1-2*s*s*math.sin(q/2)**2,1e-13)
            if s==1:near(nu,q,1e-13)
            c['dispersionPairs']+=1
    for n in [3,4,5,8]:
        for eps in map(F,[F(1,2),1,2]):
            for dx in map(F,[F(1,2),1,2]):
                G=divergence(n,dx);E=[F((j%4==0)-(j%4==2)) for j in range(n)];mean=sum(E)/n
                for alpha in map(F,[0,F(1,2),1,2]):
                    rho=[alpha*eps*x for x in mv(G,E)];Ec,de,p=project(E,rho,eps,dx)
                    assert mv(G,Ec)==[r/eps for r in rho] and sum(de)==0 and sum(p)==0
                    assert project(Ec,rho,eps,dx)[0]==Ec;c['gaussProjections']+=1
                    prefix=[F(0)]
                    for j in range(1,n):prefix.append(prefix[-1]+dx*rho[j]/eps)
                    base=mean-sum(prefix)/n;assert Ec==[base+x for x in prefix];c['independentPrefixSolutions']+=1
                    U=lambda x:eps*dx*sum(v*v for v in x)/2
                    assert U(Ec)-U(E)==eps*dx*(sum(a*b for a,b in zip(E,de))+sum(v*v for v in de)/2)
                    if alpha==0:assert U(E)==U(Ec)+U(de)
                    if alpha==2:assert U(Ec)>U(E)
                    c['energyIdentities']+=1
                J=[F(2*j-n,7) for j in range(n)];dt=F(1,3);rho0=[eps*x for x in mv(G,E)]
                for defect in [[F(0)]*n,[F(2*j-(n-1),11) for j in range(n)],[F(1)]*n]:
                    rhon=[r-dt*dj+dt*rr for r,dj,rr in zip(rho0,mv(G,J),defect)];En=[e-dt*j/eps for e,j in zip(E,J)];gn=[eps*a-b for a,b in zip(mv(G,En),rhon)]
                    assert gn==[-dt*x for x in defect];c['continuityDefects']+=1
                    if sum(rhon)==0:
                        _,de,_=project(En,rhon,eps,dx);Jeff=[j-eps*x/dt for j,x in zip(J,de)]
                        assert [(b-a)/dt+d for a,b,d in zip(rho0,rhon,mv(G,Jeff))]==[0]*n;c['effectiveCurrentChecks']+=1
                try:project(E,[F(1)]*n,eps,dx)
                except ValueError:c['incompatibleChargeRejections']+=1
                else:raise AssertionError('Incompatible charge accepted')
    return c

def references():
    return dict(initializations=[dict(eps=1,mu=mu,N=4,dx=1,k=math.pi) for mu in [1,4]],radial=[dict(initialGradient=.5,gradientRate=.25,chargeOffset=offset,permittivity=1,permeability=1,time=t,point=[1,-.5,.25],rho=3*(.5+.25*t)+offset,gauss=-offset) for offset in [0,.125] for t in [0,1,2]])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');p.add_argument('--references',action='store_true');a=p.parse_args()
    print(json.dumps(references() if a.references else verify(),sort_keys=True))
