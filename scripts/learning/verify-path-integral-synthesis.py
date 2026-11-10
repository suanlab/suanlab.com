"""Independent dense Fraction references for the four-chapter comparison."""
from fractions import Fraction as F
import json

def txt(x): return str(F(x))
def model(n,time,mass,hbar):
    time,mass,hbar=map(F,(time,mass,hbar));a,b=F(-1,3),F(2,3);step=time/n;s=mass/(hbar*step);d=n-1
    rows=[];constants=[]
    for edge in range(n):
        row=[F(0)]*d;constant=F(0)
        for index,sign in [(edge,-1),(edge+1,1)]:
            if index==0: constant+=sign*a
            elif index==n: constant+=sign*b
            else: row[index-1]+=sign
        rows.append(row);constants.append(constant)
    q=[[s*sum(row[i]*row[j] for row in rows) for j in range(d)] for i in range(d)]
    force=[-s*sum(row[i]*c for row,c in zip(rows,constants)) for i in range(d)]
    aug=[q[i]+[F(i==j) for j in range(d)] for i in range(d)];det=F(1)
    for k in range(d):
        pivot=aug[k][k];assert pivot>0;det*=pivot;aug[k]=[x/pivot for x in aug[k]]
        for i in range(d):
            if i==k: continue
            factor=aug[i][k];aug[i]=[x-factor*y for x,y in zip(aug[i],aug[k])]
    inv=[row[d:] for row in aug];mean=[a]+[sum(inv[i][j]*force[j] for j in range(d)) for i in range(d)]+[b]
    cov=[[inv[i-1][j-1] if 0<i<n and 0<j<n else F(0) for j in range(n+1)] for i in range(n+1)]
    action=s*sum((mean[i+1]-mean[i])**2 for i in range(n))/2
    return {'input':{'intervals':n,'time':txt(time),'mass':txt(mass),'hbar':txt(hbar),'start':txt(a),'end':txt(b),'omegaSquared':'0','probe':n//2},'diffusion':txt(hbar/(2*mass)),'precision':[[txt(x) for x in row] for row in q],'covariance':[[txt(x) for x in row] for row in cov],'mean':list(map(txt,mean)),'action':txt(action),'physicalAction':txt(action*hbar),'determinant':txt(det),'prefactorSquaredTimesPi':txt(s**n/(2*det))}
rows=[model(n,time,mass,hbar) for n in [2,3,7,32] for time in ['1','3/2'] for mass,hbar in [('1','1'),('3/2','1/2')]]
assert F(9,8)-F(1,2)*2*F(4,9)==F(49,72)
assert F(7,8)-F(1,2)*2*F(4,7)==F(17,56)
assert F(9,2)*F(1,2)+F(1,2)*F(1,2)**3==F(37,16)
assert F(9,2)+3*F(1,2)*F(1,2)**2==F(39,8)
print(json.dumps({'cases':rows,'harmonic':{'euclideanPrecision':'9/2','euclideanMean':'4/9','euclideanVariance':'2/9','euclideanAction':'49/72','realHessian':'7/2','realStationary':'4/7','realInverse':'2/7','realAction':'17/56'},'nonGaussian':{'stationary':'1/2','hessian':'39/8','localVariance':'8/39','quadraticMean':'37/72'}},indent=2))
