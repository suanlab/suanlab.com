import type {Vec2} from './learning-simulations';
type V3 = [number,number,number];
const B=BigInt,Z=B(0),ONE=B(1);
type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat {
  if(d===Z)throw new Error('Zero rational denominator.');
  if(d<Z){n=-n;d=-d;}
  let a=n<Z?-n:n,b=d;
  while(b!==Z){const c=a%b;a=b;b=c;}
  return {n:n/a,d:d/a};
}
const R=(n:number,d=1)=>rat(B(n),B(d));
const add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d);
const neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b));
const mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n);
const sq=(a:Rat)=>mul(a,a);
const sum=(a:Rat[])=>a.reduce(add,R(0));
const dot=(a:Rat[],b:Rat[])=>sum(a.map((v,j)=>mul(v,b[j])));
const cross=(a:Rat[],b:Rat[])=>[sub(mul(a[1],b[2]),mul(a[2],b[1])),sub(mul(a[2],b[0]),mul(a[0],b[2])),sub(mul(a[0],b[1]),mul(a[1],b[0]))];
const mv=(A:Rat[][],v:Rat[])=>A.map(row=>dot(row,v));
const vmul=(s:Rat,v:Rat[])=>v.map(x=>mul(s,x));
const vadd=(a:Rat[],b:Rat[])=>a.map((x,j)=>add(x,b[j]));
const vsub=(a:Rat[],b:Rat[])=>a.map((x,j)=>sub(x,b[j]));
const zeros=()=>[R(0),R(0),R(0)];
const matrix=()=>Array.from({length:3},zeros);
function expose(a:Rat){return {numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const numeric=(a:Rat)=>expose(a).value,list=(v:Rat[])=>v.map(expose),rows=(a:Rat[][])=>a.map(list);
function codes(p:number[],ranges:readonly(readonly[number,number])[]) {
  if(p.length!==ranges.length)throw new Error('Use the stated rotation controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
function inertia(points:{mass:Rat;position:Rat[]}[]) {
  const A=matrix();
  for(const {mass,position:r}of points){const r2=dot(r,r);for(let i=0;i<3;i++)for(let j=0;j<3;j++)A[i][j]=add(A[i][j],mul(mass,sub(i===j?r2:R(0),mul(r[i],r[j]))));}
  return A;
}
function spectralXZ(A:Rat[][]) {
  // Both the COM and x-shifted tensor have only an xz offdiagonal block.
  const xx=numeric(A[0][0]),yy=numeric(A[1][1]),zz=numeric(A[2][2]),xz=numeric(A[0][2]),d=Math.hypot(xx-zz,2*xz),phi=Math.atan2(2*xz,xx-zz)/2;
  const blockDegenerate=sub(A[0][0],A[2][2]).n===Z&&A[0][2].n===Z;
  const yIsBlockEigenvalue=sub(mul(sub(A[0][0],A[1][1]),sub(A[2][2],A[1][1])),sq(A[0][2])).n===Z;
  const eigen=[{value:(xx+zz-d)/2,vector:[-Math.sin(phi),0,Math.cos(phi)]as V3},{value:yy,vector:[0,1,0]as V3},{value:(xx+zz+d)/2,vector:[Math.cos(phi),0,Math.sin(phi)]as V3}].sort((a,b)=>a.value-b.value);
  return {eigen,repeatedSpectrum:blockDegenerate||yIsBlockEigenvalue,blockDegenerate,
    residuals:eigen.map(({value,vector:v})=>Math.hypot(...A.map((row,j)=>row.reduce((s,x,k)=>s+numeric(x)*v[k],0)-value*v[j])))};
}
/** Instantaneous rigid rotation of six antipodal masses about a selected pivot.
 * [m1,m2,m3,2a,2b,2c,2shear,10wx,10wy,10wz,10pivotX,orientation*pi/12].
 * Pivot motion changes the velocity field v_i=omega cross(r_i-O), not just
 * the reference of an unchanged physical motion. Rz is a common fixed
 * coordinate rotation applied to points, pivot, omega and momentum together.
 */
export function pointMassInertia(p:number[]) {
  codes(p,[[1,4],[1,4],[1,4],[1,6],[1,6],[1,6],[-4,4],[-20,20],[-20,20],[-20,20],[-20,20],[-12,12]]);
  const [m1,m2,m3,ac,bc,cc,sc,wxc,wyc,wzc,dc,orientationCode]=p;
  const a=R(ac,2),b=R(bc,2),c=R(cc,2),shear=R(sc,2),omega=[R(wxc,10),R(wyc,10),R(wzc,10)],pivot=[R(dc,10),R(0),R(0)],mass=R(2*(m1+m2+m3));
  const points=[{mass:R(m1),position:[a,R(0),R(0)]},{mass:R(m1),position:[neg(a),R(0),R(0)]},{mass:R(m2),position:[R(0),b,R(0)]},{mass:R(m2),position:[R(0),neg(b),R(0)]},{mass:R(m3),position:[shear,R(0),c]},{mass:R(m3),position:[neg(shear),R(0),neg(c)]}];
  const relative=points.map(v=>({...v,position:vsub(v.position,pivot)})),IC=inertia(points),IO=inertia(relative),LC=mv(IC,omega),LO=mv(IO,omega),VC=cross(omega,pivot).map(neg),P=vmul(mass,VC),kC=div(dot(omega,LC),R(2)),kO=div(dot(omega,LO),R(2)),kTranslation=div(mul(mass,dot(VC,VC)),R(2));
  let pointK=R(0),pointL=zeros(),pointP=zeros(),center=zeros();
  const data=points.map((v,j)=>{const velocity=cross(omega,relative[j].position),momentum=vmul(v.mass,velocity);pointK=add(pointK,div(mul(v.mass,dot(velocity,velocity)),R(2)));pointL=vadd(pointL,cross(relative[j].position,momentum));pointP=vadd(pointP,momentum);center=vadd(center,vmul(v.mass,v.position));return {...v,relative:relative[j].position,velocity,momentum};});
  const expectedShift=pivot.map((x,i)=>pivot.map((y,j)=>mul(mass,sub(i===j?dot(pivot,pivot):R(0),mul(x,y))))),parallelAxisResidual=IO.map((row,i)=>row.map((v,j)=>sub(sub(v,IC[i][j]),expectedShift[i][j])));
  const axisSquared=dot(omega,omega),zeroOmega=axisSquared.n===Z,axisInertia=zeroOmega?null:div(dot(omega,LO),axisSquared),crossLW=cross(LO,omega),nonparallel=crossLW.some(v=>v.n!==Z),spectrum=spectralXZ(IO),comSpectrum=spectralXZ(IC);
  const orientation=orientationCode*Math.PI/12,cos=Math.cos(orientation),sin=Math.sin(orientation),rotate=(v:number[]):V3=>[cos*v[0]-sin*v[1],sin*v[0]+cos*v[1],v[2]],rotateR=(v:Rat[])=>rotate(v.map(numeric)),Rz=[[cos,-sin,0],[sin,cos,0],[0,0,1]],labTensor=Rz.map(row=>Rz.map(col=>row.reduce((s,x,i)=>s+x*IO[i].reduce((t,y,j)=>t+numeric(y)*col[j],0),0)));
  const labOmega=rotateR(omega),labMomentum=rotateR(LO),normalizedAxis=zeroOmega?null:labOmega.map(x=>x/Math.sqrt(numeric(axisSquared)))as V3;
  const angle=zeroOmega?null:Math.acos(Math.max(-1,Math.min(1,labOmega.reduce((v,x,j)=>v+x*labMomentum[j],0)/(Math.hypot(...labOmega)*Math.hypot(...labMomentum)))));
  const directionalInertia:Vec2[]=[];
  // Sweep directions in the body xz-plane, not the complete3D sphere.
  for(let j=0;j<=180;j++){const theta=Math.PI*j/180,n=[Math.cos(theta),0,Math.sin(theta)],I=n.reduce((v,x,k)=>v+x*IO[k].reduce((t,y,l)=>t+numeric(y)*n[l],0),0);directionalInertia.push([theta,I]);}
  return {mass:expose(mass),pairMasses:[m1,m2,m3],omega:list(omega),pivot:list(pivot),centerOfMass:list(center.map(v=>div(v,mass))),tensorCOM:rows(IC),tensorPivot:rows(IO),angularMomentumCOM:list(LC),angularMomentumPivot:list(LO),centerVelocity:list(VC),linearMomentum:list(P),kineticCOM:expose(kC),kineticPivot:expose(kO),translationKinetic:expose(kTranslation),kineticDecompositionResidual:expose(sub(sub(kO,kC),kTranslation)),pointKineticResidual:expose(sub(pointK,kO)),pointMomentumResidual:list(vsub(pointP,P)),pointAngularResidual:list(vsub(pointL,LO)),parallelAxisResidual:rows(parallelAxisResidual),originAngularResidual:list(vsub(LO,vsub(LC,cross(pivot,P)))),axisInertia:axisInertia?expose(axisInertia):null,zeroOmega,nonparallel,angleBetweenAngularAndOmega:angle,spectrum,comSpectrum,orientation,orientationCode,labTensor,labOmega,labAngularMomentum:labMomentum,labPivot:rotateR(pivot),normalizedAxis,
    points:data.map(v=>({mass:expose(v.mass),position:list(v.position),relative:list(v.relative),velocity:list(v.velocity),momentum:list(v.momentum),labPosition:rotateR(v.position),labVelocity:rotateR(v.velocity)})),directionalInertia,
    pivotChangesVelocityField:true as const,timeEvolutionSolved:false as const};
}

type Q4=[number,number,number,number];
const rotateBy=(R:number[][],v:number[]):V3=>R.map(row=>row.reduce((s,x,j)=>s+x*v[j],0))as V3;
export function bodyToLabRotation(raw:number[]) {
  if(raw.length!==4||raw.some(x=>!Number.isFinite(x)))throw new Error('Use a finite scalar-first quaternion.');
  const norm=Math.hypot(...raw);if(!Number.isFinite(norm)||norm===0)throw new Error('Quaternion norm must be finite and positive.');
  const [w,x,y,z]=raw.map(v=>v/norm);
  return [[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]];
}
function rk4(f:(y:number[])=>number[],y:number[],h:number) {
  const k1=f(y),k2=f(y.map((x,j)=>x+h*k1[j]/2)),k3=f(y.map((x,j)=>x+h*k2[j]/2)),k4=f(y.map((x,j)=>x+h*k3[j]));
  return y.map((x,j)=>x+h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6);
}
/** Torque-free rotation about fixed COM; body axes are principal axes.
 * Positive second moments A,B,C produce physical inertias B+C,C+A,A+B.
 * [A,B,C,10w1,10w2,10w3,initialZ*pi/12,10t].
 * Adaptively integrates Euler equations and unprojected scalar-first Hamilton
 * body-to-lab quaternion with RK4 step doubling/Richardson correction.
 */
export function torqueFreeEuler(p:number[]) {
  codes(p,[[1,5],[1,5],[1,5],[-20,20],[-20,20],[-20,20],[-12,12],[0,100]]);
  const [A,Bc,C,w1c,w2c,w3c,angleCode,tc]=p,I=[Bc+C,C+A,A+Bc],W0=[R(w1c,10),R(w2c,10),R(w3c,10)],w0=W0.map(numeric),theta=angleCode*Math.PI/12,T=tc/10,q0:Q4=[Math.cos(theta/2),0,0,Math.sin(theta/2)],K0=div(sum(W0.map((w,j)=>mul(R(I[j]),sq(w)))),R(2)),L20=sum(W0.map((w,j)=>sq(mul(R(I[j]),w)))),Lbody0=w0.map((x,j)=>x*I[j]),Llab0=rotateBy(bodyToLabRotation(q0),Lbody0),initial=[...w0,...q0];
  const rhs=(y:number[])=>{const [u,v,w,a,b,c,d]=y;return[(I[1]-I[2])*v*w/I[0],(I[2]-I[0])*w*u/I[1],(I[0]-I[1])*u*v/I[2],(-b*u-c*v-d*w)/2,(a*u+c*w-d*v)/2,(a*v+d*u-b*w)/2,(a*w+b*v-c*u)/2];};
  let state=[...initial],time=0,proposal=.01,accepted=0,rejected=0,attempts=0;
  const at=(target:number)=>{
    while(time<target){
      const h=Math.min(proposal,target-time);if(h<1e-14){time=target;break;}
      if(++attempts>40000)throw new Error('Torque-free integration exceeded its step budget.');
      const whole=rk4(rhs,state,h),half=rk4(rhs,rk4(rhs,state,h/2),h/2),error=half.map((x,j)=>(x-whole[j])/15),scaled=Math.max(...error.map((e,j)=>Math.abs(e)/(1e-12+2e-11*Math.max(Math.abs(state[j]),Math.abs(half[j])))));
      if(!Number.isFinite(scaled))throw new Error('Nonfinite torque-free integration error.');
      if(scaled<=1){state=half.map((x,j)=>x+error[j]);time=h===target-time?target:time+h;accepted++;}else rejected++;
      const factor=scaled===0?5:Math.max(.2,Math.min(5,.9*scaled**(-.2)));proposal=Math.max(1e-10,Math.min(.1,h*factor));
      if(scaled>1&&proposal===1e-10)throw new Error('Torque-free step size underflow.');
    }
    const bodyOmega=state.slice(0,3)as V3,rawQuaternion=state.slice(3)as Q4,norm=Math.hypot(...rawQuaternion),rotation=bodyToLabRotation(rawQuaternion),bodyMomentum=bodyOmega.map((x,j)=>I[j]*x)as V3,labMomentum=rotateBy(rotation,bodyMomentum),labOmega=rotateBy(rotation,bodyOmega),energy=bodyOmega.reduce((s,w,j)=>s+I[j]*w*w/2,0),normL2=bodyMomentum.reduce((s,x)=>s+x*x,0),derivative=rhs(state).slice(0,3)as V3;
    const orthogonalityResidual=Math.max(...rotation.flatMap((row,i)=>rotation.map((other,j)=>Math.abs(row.reduce((s,x,k)=>s+x*other[k],0)-(i===j?1:0)))));
    const determinant=rotation[0][0]*(rotation[1][1]*rotation[2][2]-rotation[1][2]*rotation[2][1])-rotation[0][1]*(rotation[1][0]*rotation[2][2]-rotation[1][2]*rotation[2][0])+rotation[0][2]*(rotation[1][0]*rotation[2][1]-rotation[1][1]*rotation[2][0]);
    return {time:target,bodyOmega,labOmega,bodyAngularMomentum:bodyMomentum,labAngularMomentum:labMomentum,rawQuaternion,quaternion:rawQuaternion.map(x=>x/norm)as Q4,rawQuaternionNormResidual:norm-1,rotation,orthogonalityResidual,determinantResidual:determinant-1,energy,angularMomentumNormSquared:normL2,energyResidual:energy-numeric(K0),angularNormResidual:normL2-numeric(L20),labAngularResidual:labMomentum.map((x,j)=>x-Llab0[j])as V3,bodyAngularAcceleration:derivative,orientationNormalizedForDisplay:true as const};
  };
  const times=new Set<number>([T]);for(let j=0;j<=400;j++)times.add(j/40);
  const states=[...times].sort((a,b)=>a-b).map(at),query=states.find(x=>x.time===T)!,curves:Vec2[][]=Array.from({length:10},()=>[]);
  for(const q of states){[...q.bodyOmega,...q.labAngularMomentum,q.energy,numeric(K0),q.angularMomentumNormSquared,numeric(L20)].forEach((v,j)=>curves[j].push([q.time,v]));}
  const stability=I.map((moment,j)=>{const k=(j+1)%3,l=(j+2)%3,rate=(I[k]-moment)*(moment-I[l])/(I[k]*I[l]);return {axis:j,moment,unitSpinGrowthRateSquared:rate,linearClass:rate>0?'unstable':rate<0?'oscillatory':'degenerate'};});
  return {secondMoments:[A,Bc,C],principalMoments:I,initialOmega:list(W0),initialOrientationAngle:theta,time:T,initialQuaternion:q0,initialEnergy:expose(K0),initialAngularMomentumNormSquared:expose(L20),initialLabAngularMomentum:Llab0,stationary:W0.every(w=>w.n===Z),spherical:I[0]===I[1]&&I[1]===I[2],axisymmetric:I[0]===I[1]||I[0]===I[2]||I[1]===I[2],strictTriangleInequalities:I.every((x,j)=>x<I[(j+1)%3]+I[(j+2)%3]),query,initial:states[0],curves,states,stability,solver:{method:'RK4 step doubling with Richardson correction',absoluteTolerance:1e-12,relativeTolerance:2e-11,acceptedSteps:accepted,rejectedSteps:rejected,quaternionProjectedDuringIntegration:false},externalTorque:[0,0,0]as V3};
}

/** Orthographic projection along (1,1,1): axes are orthonormal rows.
 * Physical3D lengths/angles must be computed before projection.
 */
export function projectRotationVector(v:readonly number[]):Vec2 {
  if(v.length!==3||!Array.from(v).every(Number.isFinite))throw new Error('Use a finite three-dimensional vector.');
  return [(v[0]-v[1])/Math.sqrt(2),(v[0]+v[1]-2*v[2])/Math.sqrt(6)];
}
