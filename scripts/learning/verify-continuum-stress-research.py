#!/usr/bin/env python3
"""Independent finite stress-observation and planar-metric checks; standard library only.
Exact rational calculations and independent gradient/eigenvalue evaluations complement the proofs.
"""
from fractions import Fraction as F
from collections import Counter
import math,json


def inverse(matrix):
    n=len(matrix);a=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:raise ValueError('Rank-deficient observation matrix')
        a[j],a[pivot]=a[pivot],a[j];scale=a[j][j];a[j]=[v/scale for v in a[j]]
        for i in range(n):
            if i!=j:
                factor=a[i][j];a[i]=[v-factor*w for v,w in zip(a[i],a[j])]
    return [row[n:] for row in a]


def mv(a,v):return [sum(x*y for x,y in zip(row,v)) for row in a]
def transpose(a):return [list(row) for row in zip(*a)]
def product(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def eigenvalues(a):
    # Jacobi rotations of the actual 3x3 Gram matrix, independent of the special 2x2 formula.
    a=[list(row) for row in a]
    for _ in range(100):
        p,q=max(((i,j) for i in range(3) for j in range(i+1,3)),key=lambda ij:abs(a[ij[0]][ij[1]]))
        if abs(a[p][q])<1e-16:break
        tau=(a[q][q]-a[p][p])/(2*a[p][q]);t=math.copysign(1,tau)/(abs(tau)+math.sqrt(1+tau*tau));c=1/math.sqrt(1+t*t);s=t*c
        old=a[p][q];a[p][p]-=t*old;a[q][q]+=t*old;a[p][q]=a[q][p]=0.
        for k in range(3):
            if k not in (p,q):
                ap,aq=a[k][p],a[k][q];a[k][p]=a[p][k]=c*ap-s*aq;a[k][q]=a[q][k]=s*ap+c*aq
    return sorted(a[i][i] for i in range(3))


def conformal(alpha,length,x,y):
    if alpha==0:return [x,y]
    rx,ry=alpha*x/length,alpha*y/length
    return [length/alpha*(math.expm1(rx)*math.cos(ry)-2*math.sin(ry/2)**2),length/alpha*math.exp(rx)*math.sin(ry)]


def close(a,b,absolute=2e-10,relative=2e-9):
    assert abs(a-b)<=absolute+relative*abs(b),(a,b)


def verify():
    out=Counter();normals=[(F(1),F(0)),(F(3,5),F(4,5)),(F(0),F(1))]
    A=[[F(1),nx*nx-ny*ny,2*nx*ny] for nx,ny in normals];inv=inverse(A)
    for p in [F(-1),F(0),F(2)]:
        for d in [F(-2),F(0),F(1)]:
            for tau in [F(-1),F(0),F(3)]:
                state=[p,d,tau];sigma=[[p+d,tau],[tau,p-d]];data=[sum(n[i]*v for i,v in enumerate(mv(sigma,n))) for n in normals]
                assert data==mv(A,state) and mv(inv,data)==state;out['exactScalarStressRecoveries']+=1
                assert sum(v*v for row in sigma for v in row)==2*sum(v*v for v in state)
                for error in [[F(0)]*3,[F(1,10),F(0),F(-1,10)],[F(0),F(1,5),F(0)]]:
                    estimated=mv(inv,[x+e for x,e in zip(data,error)]);assert [x-y for x,y in zip(estimated,state)]==mv(inv,error);out['exactAdditiveNoisePropagations']+=1
                for nx,ny in normals:
                    tangent=[-ny,nx]
                    for beta in [F(-2),F(0),F(3)]:
                        changed=[[sigma[i][j]+beta*tangent[i]*tangent[j] for j in range(2)] for i in range(2)]
                        assert mv(changed,[nx,ny])==mv(sigma,[nx,ny]);out['exactOneTractionNullCases']+=1
    axes=[[F(1),F(1),F(0)],[F(1),F(0),F(1)],[F(1),F(-1),F(0)]];operator=inverse(axes);covariance=product(operator,transpose(operator))
    assert covariance==[[F(1,2),0,F(-1,2)],[0,F(1,2),0],[F(-1,2),0,F(3,2)]];out['exactNonDiagonalEstimatorCovariance']+=1
    try:inverse([[F(1),F(1),F(0)]]*3)
    except ValueError:out['exactRepeatedAxisRankRejection']+=1
    else:raise AssertionError('Repeated axes were accepted')
    for degrees in [5,15,30,45,60,80]:
        delta=degrees*math.pi/180;matrix=[[1.,math.cos(2*t),math.sin(2*t)] for t in [0.,delta,2*delta]];gram=product(transpose(matrix),matrix);actual=eigenvalues(gram)
        c=math.cos(2*delta);T=4+2*c*c;D=8*math.sin(delta)**4;large=(T+math.sqrt(T*T-4*D))/2;expected=sorted([large,D/large,2*math.sin(2*delta)**2])
        for value,reference in zip(actual,expected):close(value,reference,2e-14,2e-9)
        assert actual[0]>0;out['independentJacobiSpectrumCases']+=1
        condition=math.sqrt(actual[-1]/actual[0]);assert condition>=math.sqrt(2)-1e-12
        if degrees==60:close(condition,math.sqrt(2));out['balancedDesignAttainsLowerBound']+=1
    for alpha in [F(-1,2),F(0),F(1,10),F(3,10)]:
        for L in [F(1,2),F(1),F(2)]:
            for halfwidth in [L/4,L,2*L]:
                # Direct four oriented edges; the required angle one-form is affine on each edge.
                r=halfwidth;corners=[(-r,-r),(r,-r),(r,r),(-r,r),(-r,-r)];loop=F(0)
                for (x0,y0),(x1,y1) in zip(corners,corners[1:]):
                    x,y=(x0+x1)/2,(y0+y1)/2;loop+=-alpha*y/L**2*(x1-x0)+alpha*x/L**2*(y1-y0)
                assert loop==2*alpha/L**2*(2*r)**2;out['exactRadialAngleLoopCases']+=1
    for alpha in [0.,-.5,.2,1.]:
        for L in [.5,2.]:
            for x,y in [(-.2,.1),(0.,0.),(.3,-.4)]:
                step=L*1e-5;xp=conformal(alpha,L,x+step,y);xm=conformal(alpha,L,x-step,y);yp=conformal(alpha,L,x,y+step);ym=conformal(alpha,L,x,y-step)
                numeric=[[(xp[i]-xm[i])/(2*step),(yp[i]-ym[i])/(2*step)] for i in range(2)];scale=math.exp(alpha*x/L);angle=alpha*y/L;analytic=[[scale*math.cos(angle),-scale*math.sin(angle)],[scale*math.sin(angle),scale*math.cos(angle)]]
                for i in range(2):
                    for j in range(2):close(numeric[i][j],analytic[i][j],2e-10,2e-9)
                metric=product(transpose(numeric),numeric)
                for i in range(2):
                    for j in range(2):close(metric[i][j],scale*scale*(i==j),3e-10,3e-9)
                assert numeric[0][0]*numeric[1][1]-numeric[0][1]*numeric[1][0]>0;out['independentConformalGradientMetricCases']+=1
                if alpha:
                    first=conformal(alpha,L,x,y);second=conformal(alpha,L,x,y+2*math.pi*L/abs(alpha))
                    for a,b in zip(first,second):close(a,b,3e-12,3e-12)
                    out['locallyValidGloballyRepeatingMapCases']+=1
    return dict(sorted(out.items()))


if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
