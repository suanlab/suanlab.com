"""Independent exact normal systems / Fourier algebra and oriented quadrature.
Standard library only. No product module imports or third-party dependencies.
"""
from fractions import Fraction as F
import argparse,itertools,json,math
D=[[F(1),F(0)],[F(-1),F(1)],[F(0),F(-1)]]

def project(y,v):
    total=sum(y);den=sum(v)
    return [t-w*total/den for t,w in zip(y,v)]
def normal_fit(y,v):
    h=[[sum(D[e][i]*D[e][j]/v[e] for e in range(3)) for j in range(2)] for i in range(2)]
    g=[sum(D[e][i]*y[e]/v[e] for e in range(3)) for i in range(2)]
    det=h[0][0]*h[1][1]-h[0][1]*h[1][0]
    phi=[(g[0]*h[1][1]-h[0][1]*g[1])/det,(h[0][0]*g[1]-g[0]*h[1][0])/det]
    return [sum(row[i]*phi[i] for i in range(2)) for row in D],[[h[1][1]/det,-h[0][1]/det],[-h[1][0]/det,h[0][0]/det]]

def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def add(a,b):return [x+y for x,y in zip(a,b)]
def scale(a,s):return [s*x for x in a]

def A(x,L,b0,b1=1,n=1):
    k=2*math.pi*n/L
    return [-b0*x[1]/2,b0*x[0]/2+b1/k*math.sin(k*x[0]),0]
def B(x,L,b0,b1=1,n=1):return [0,0,b0+b1*math.cos(2*math.pi*n*x[0]/L)]
def integrate(fn,a,b,N=256):
    h=(b-a)/N
    return h/3*(fn(a)+fn(b)+sum((4 if i%2 else 2)*fn(a+i*h) for i in range(1,N)))
def circulation(L,b0,n):
    vertices=[[0,0,0],[L,0,0],[L,L,0],[0,L,0]];total=0
    for i,p in enumerate(vertices):
        q=vertices[(i+1)%4];delta=add(q,scale(p,-1))
        total+=integrate(lambda t:dot(A(add(p,scale(delta,t)),L,b0,1,n),delta),0,1)
    return total

def near(a,b,tol=3e-9):assert math.isfinite(a) and abs(a-b)<=tol*(1+abs(b)),(a,b)

def verify():
    c=dict(weightedNormalSystems=0,covarianceChecks=0,orientationChecks=0,gaugeShifts=0,correlationModels=0,fourierModes=0,fourierGaugeChecks=0,faceFluxChecks=0,cellMeanResiduals=0,finiteCurlChecks=0,boundaryJumps=0,constantPotentialLoops=0)
    variance_sets=[(1,1,1),(1,4,9),(1,1,16),(4,1,1),(F(1,4),1,4),(9,4,1)]
    ys=[[F(1)+F(a,10),F(2)+F(b,10),F(-3)+F(d,10)] for a,b,d in [(0,0,0),(1,-2,2),(-3,1,1),(2,2,2),(1,-1,0),(-1,-2,4)]]
    for raw in variance_sets:
        v=list(map(F,raw))
        for y in ys:
            fit=project(y,v);independent,cov=normal_fit(y,v);assert fit==independent and sum(fit)==0
            assert sum((a-b)**2/w for a,b,w in zip(y,fit,v))==sum(y)**2/sum(v);c['weightedNormalSystems']+=1
            for signs in itertools.product([-1,1],repeat=3):
                yy=[s*x for s,x in zip(signs,y)];loop=sum(s*x for s,x in zip(signs,yy));out=[yy[i]-v[i]*signs[i]*loop/sum(v) for i in range(3)]
                assert out==[s*x for s,x in zip(signs,fit)];c['orientationChecks']+=1
            phi=[F(0),fit[0],fit[0]+fit[1]]
            for offset in [F(-100),F(0),F(100)]:
                shifted=[x+offset for x in phi];assert [shifted[1]-shifted[0],shifted[2]-shifted[1],shifted[0]-shifted[2]]==fit
            c['gaugeShifts']+=1
        P=[[F(i==j)-v[i]/sum(v) for j in range(3)] for i in range(3)]
        C=[[sum(P[i][e]*v[e]*P[j][e] for e in range(3)) for j in range(3)] for i in range(3)]
        assert C==[[F(i==j)*v[i]-v[i]*v[j]/sum(v) for j in range(3)] for i in range(3)]
        _,cov=normal_fit(ys[0],v);assert C[2][2]==cov[1][1]==v[2]*(v[0]+v[1])/sum(v);c['covarianceChecks']+=1
    for s2 in map(F,[1,4,9]):
        for t2 in map(F,[0,1,4]):
            Sigma=[[s2*F(i==j)+t2 for j in range(3)] for i in range(3)];den=sum(map(sum,Sigma));assert den==3*s2+9*t2
            P=[[F(i==j)-F(1,3) for j in range(3)] for i in range(3)]
            C=[[sum(P[i][a]*Sigma[a][b]*P[j][b] for a in range(3) for b in range(3)) for j in range(3)] for i in range(3)]
            assert C==[[s2*(F(i==j)-F(1,3)) for j in range(3)] for i in range(3)];c['correlationModels']+=1
    for k in itertools.product([-1,0,1],repeat=3):
        if k==(0,0,0):continue
        for seed in [[1,0,0],[0,1,0],[0,0,1],[1,2,3]]:
            b=cross(k,seed);a=scale(cross(k,b),F(1,dot(k,k))) # Ahat=i a
            assert dot(k,a)==0 and scale(cross(k,a),-1)==b;c['fourierModes']+=1
            ap=add(a,scale(k,F(2,3)));assert scale(cross(k,ap),-1)==b
            assert dot(k,ap)==F(2,3)*dot(k,k);c['fourierGaugeChecks']+=1
    for L,b0,n in itertools.product([.5,1,2],[-1,0,.5,1],[1,2,3]):
        flux=integrate(lambda x:B([x,0,0],L,b0,1,n)[2]*L,0,L)
        near(flux,b0*L*L);near(circulation(L,b0,n),flux);c['faceFluxChecks']+=1
        points=[[L*(i+.5)/64,L*.23,L*.31] for i in range(64)]
        mean=sum(B(x,L,b0,1,n)[2] for x in points)/64
        residual=sum((B(x,L,b0,1,n)[2]-B(x,L,0,1,n)[2])**2 for x in points)/64
        near(mean,b0);near(residual,b0*b0);c['cellMeanResiduals']+=1
        x=[.13*L,.27*L,.31*L]
        for axis in [0,1]:
            shifted=list(x);shifted[axis]+=L;jump=add(A(shifted,L,b0,1,n),scale(A(x,L,b0,1,n),-1));expected=[0,b0*L/2,0] if axis==0 else [-b0*L/2,0,0]
            for a,b in zip(jump,expected):near(a,b)
            c['boundaryJumps']+=1
        h=L*1e-5
        def derivative(component,axis):
            vals=[]
            for delta in [-2,-1,1,2]:
                p=list(x);p[axis]+=delta*h;vals.append(A(p,L,b0,1,n)[component])
            return (vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h)
        curl=[derivative(2,1)-derivative(1,2),derivative(0,2)-derivative(2,0),derivative(1,0)-derivative(0,1)]
        for a,b in zip(curl,B(x,L,b0,1,n)):near(a,b)
        c['finiteCurlChecks']+=1
    for L,a0 in itertools.product([.5,1,2],[-.2,0,.7]):
        near(integrate(lambda t:a0,0,L),a0*L);c['constantPotentialLoops']+=1
    return c

def references():
    H=[[2,0,0],[0,4,0],[0,0,0]];linear=[0,1,0];vertices=[[0,0,0],[1,0,0],[0,1,0]]
    return dict(paths=[dict(start=vertices[e],end=vertices[(e+1)%3],control=[.25,.25,.2],difference=[1,2,-3][e],reference=C,hessian=H,linear=linear) for C in [-100,0,100] for e in range(3)],constants=[dict(axial=a,L=2,point=[.25,.3,.1]) for a in [-1e-6,0,2e-6]])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');p.add_argument('--references',action='store_true');args=p.parse_args()
    print(json.dumps(references() if args.references else verify(),sort_keys=True))
