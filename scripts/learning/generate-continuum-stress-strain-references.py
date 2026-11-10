"""Independent quadrature of physical tractions and observations of mapped points."""
import argparse,json,random
from pathlib import Path
import mpmath as mp
mp.mp.dps=140
p=argparse.ArgumentParser();p.add_argument('--output',default='scripts/fixtures/continuum-stress-strain-independent.json');args=p.parse_args();path=Path(args.output)
if path.exists():raise SystemExit('Refusing to overwrite an existing reference')
rng=random.Random(240611)
M=mp.mpf
def serial(value):
 if isinstance(value,str):return value
 if isinstance(value,(list,tuple)):return[serial(v)for v in value]
 if isinstance(value,dict):return{k:serial(v)for k,v in value.items()}
 return float(value)
def sigma(s,x,y):return[[M(s['normalX'])+M(s['gradientX'])*x,M(s['shear'])+M(s['crossGradient'])*y],[M(s['shear'])+M(s['crossGradient'])*y,M(s['normalY'])+M(s['gradientY'])*y]]
def matvec(a,n):return[sum(a[i][j]*n[j]for j in range(2))for i in range(2)]
stress=[]
for i in range(48):
 s={k:rng.uniform(-3,3)for k in ['normalX','normalY','shear','gradientX','gradientY','crossGradient']}
 rect=dict(centerX=rng.uniform(-.8,.8),centerY=rng.uniform(-.8,.8),halfWidth=rng.uniform(.1,.9),halfHeight=rng.uniform(.1,.9),depth=rng.uniform(.3,1.7),bodyForce=[rng.uniform(-4,4),rng.uniform(-4,4)])
 x,y,angle=rng.uniform(-2,2),rng.uniform(-2,2),rng.uniform(-3,3);n=[mp.cos(M(angle)),mp.sin(M(angle))];tr=matvec(sigma(s,M(x),M(y)),n);normal=sum(a*b for a,b in zip(tr,n));shear=-n[1]*tr[0]+n[0]*tr[1]
 cx,cy,hx,hy,depth=map(M,[rect[k]for k in ['centerX','centerY','halfWidth','halfHeight','depth']]);faces=[]
 for name,normalvec,point,lo,hi in [('left',[-1,0],lambda z:[cx-hx,cy+z],-hy,hy),('right',[1,0],lambda z:[cx+hx,cy+z],-hy,hy),('bottom',[0,-1],lambda z:[cx+z,cy-hy],-hx,hx),('top',[0,1],lambda z:[cx+z,cy+hy],-hx,hx)]:
  def force(z):return matvec(sigma(s,*point(z)),normalvec)
  forces=[depth*mp.quad(lambda z:force(z)[j],[lo,hi])for j in range(2)]
  moment=depth*mp.quad(lambda z:(point(z)[0]-cx)*force(z)[1]-(point(z)[1]-cy)*force(z)[0],[lo,hi])
  faces.append(dict(name=name,force=forces,moment=moment))
 surface=[sum(f['force'][j]for f in faces)for j in range(2)]
 # Derivatives of the observed field, independently differentiated and integrated over physical volume.
 div=lambda xx,yy:[mp.diff(lambda z:sigma(s,z,yy)[0][0],xx)+mp.diff(lambda z:sigma(s,xx,z)[0][1],yy),mp.diff(lambda z:sigma(s,z,yy)[1][0],xx)+mp.diff(lambda z:sigma(s,xx,z)[1][1],yy)]
 nodes=[-mp.sqrt(M(3)/5),M(0),mp.sqrt(M(3)/5)];weights=[M(5)/9,M(8)/9,M(5)/9]
 volumeForce=[depth*hx*hy*sum(weights[a]*weights[b]*div(cx+hx*nodes[a],cy+hy*nodes[b])[j]for a in range(3)for b in range(3))for j in range(2)]
 stress.append(dict(stress=s,rectangle=rect,x=x,y=y,angle=angle,expected=serial(dict(traction=tr,normalStress=normal,signedShear=shear,faces=faces,surfaceForce=surface,surfaceMoment=sum(f['moment']for f in faces),volumeForce=volumeForce))))
deformation=[]
def point(d,X,Y):
 a=M(d['stretchX'])*X+M(d['shear'])*Y;b=M(d['stretchY'])*Y;c=mp.cos(M(d['rotation']));s=mp.sin(M(d['rotation']))
 return[c*a-s*b+M(d['translationX']),s*a+c*b+M(d['translationY'])]
for i in range(48):
 d=dict(stretchX=rng.uniform(.5,2),stretchY=rng.uniform(.5,2),shear=rng.uniform(-1,1),rotation=rng.uniform(-3,3),translationX=rng.uniform(-1,1),translationY=rng.uniform(-1,1));angle=rng.uniform(-3,3);origin=point(d,M(0),M(0));p1=point(d,M(1),M(0));p2=point(d,M(0),M(1));v1=[p1[j]-origin[j]for j in range(2)];v2=[p2[j]-origin[j]for j in range(2)]
 green=[[(sum(v1[j]**2 for j in range(2))-1)/2,sum(v1[j]*v2[j]for j in range(2))/2],[sum(v1[j]*v2[j]for j in range(2))/2,(sum(v2[j]**2 for j in range(2))-1)/2]]
 n=[mp.cos(M(angle)),mp.sin(M(angle))];mapped=point(d,*n);length=mp.sqrt(sum((mapped[j]-origin[j])**2 for j in range(2)));corners=[point(d,M(X),M(Y))for X,Y in [(-1,-1),(1,-1),(1,1),(-1,1)]];area=abs(sum(corners[j][0]*corners[(j+1)%4][1]-corners[j][1]*corners[(j+1)%4][0]for j in range(4)))/2
 F=[[v1[0],v2[0]],[v1[1],v2[1]]];eps=[[(F[a][b]+F[b][a])/2-(1 if a==b else 0)for b in range(2)]for a in range(2)]
 deformation.append(dict(model=d,angle=angle,expected=serial(dict(green=green,infinitesimal=eps,stretch=length,engineeringStrain=length-1,area=area,corners=corners))))
tiny=[]
for value in [1e-12,1e-20,1e-30]:
 theta=M(value);tiny.append(dict(angle=value,cosMinusOne=float(mp.cos(theta)-1),linearControlGreen=float(theta**2/2),linearControlStrain=float(mp.sqrt(1+theta**2)-1)))
path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(precision=140,stressCases=stress,deformationCases=deformation,tinyCases=tiny),indent=2)+'\n');print(json.dumps(dict(stressCases=len(stress),deformationCases=len(deformation),tinyCases=len(tiny))))
