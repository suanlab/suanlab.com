import type {Vec2} from './learning-simulations';
const B=BigInt,Z=B(0),ONE=B(1);type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat{if(d===Z)throw new Error('Zero rational denominator.');if(d<Z){n=-n;d=-d;}let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}return{n:n/a,d:d/a};}
const R=(n:number)=>rat(B(n)),D=(n:number)=>rat(B(n),B(10)),add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b)),mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n);
function expose(a:Rat){return{numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const list=(a:Rat[])=>a.map(expose),dot=(a:Rat[],b:Rat[])=>a.reduce((v,x,j)=>add(v,mul(x,b[j])),R(0)),numeric=(a:Rat)=>expose(a).value;
function controls(p:number[],ranges:readonly(readonly[number,number])[]){if(p.length!==ranges.length)throw new Error('Use the stated classical-mechanics controls.');p.forEach((v,j)=>{if(!Number.isInteger(v)||v<ranges[j][0]||v>ranges[j][1])throw new Error('Use bounded integer control codes.');});}
const half=rat(ONE,B(2));
const cross=(a:Rat[],b:Rat[])=>sub(mul(a[0],b[1]),mul(a[1],b[0]));
const crossN=(a:number[],b:number[])=>a[0]*b[1]-a[1]*b[0];

const dotN=(a:number[],b:number[])=>a[0]*b[0]+a[1]*b[1];
const rotateQuarter=(v:number[])=>[-v[1],v[0]];

/** Isolated central harmonic interaction; the pair potential is stored only once. */
export function isolatedCentralPair(p:number[]) {
 controls(p,[[1,4],[1,4],[1,4],...Array.from({length:10},()=>[-20,20]as const),[0,100]]);
 const [m1,m2,k,x1,y1,x2,y2,vx1,vy1,vx2,vy2,dx,dy,tc]=p,M=m1+m2,mu=m1*m2/M,q10r=[D(x1),D(y1)],q20r=[D(x2),D(y2)],v10r=[D(vx1),D(vy1)],v20r=[D(vx2),D(vy2)],dr=[D(dx),D(dy)];
 const P0r=v10r.map((v,j)=>add(mul(R(m1),v),mul(R(m2),v20r[j]))),L0r=add(cross(q10r,v10r.map(v=>mul(R(m1),v))),cross(q20r,v20r.map(v=>mul(R(m2),v)))),LD0r=sub(L0r,cross(dr,P0r)),z0r=q10r.map((v,j)=>sub(v,q20r[j])),initialKr=mul(half,add(mul(R(m1),dot(v10r,v10r)),mul(R(m2),dot(v20r,v20r)))),initialVr=mul(half,mul(R(k),dot(z0r,z0r))),initialEr=add(initialKr,initialVr);
 const q10=q10r.map(numeric),q20=q20r.map(numeric),v10=v10r.map(numeric),v20=v20r.map(numeric),d=dr.map(numeric),P0=P0r.map(numeric),R0=q10.map((v,j)=>(m1*v+m2*q20[j])/M),V=P0.map(v=>v/M),z0=z0r.map(numeric),w0=v10.map((v,j)=>v-v20[j]),omega=Math.sqrt(k/mu),T=tc/10;
 const state=(t:number)=>{
  const c=Math.cos(omega*t),s=Math.sin(omega*t),center=R0.map((v,j)=>v+V[j]*t),z=z0.map((v,j)=>v*c+w0[j]*s/omega),w=z0.map((v,j)=>-omega*v*s+w0[j]*c),q1=t===0?[...q10]:center.map((v,j)=>v+m2*z[j]/M),q2=t===0?[...q20]:center.map((v,j)=>v-m1*z[j]/M),v1=t===0?[...v10]:V.map((v,j)=>v+m2*w[j]/M),v2=t===0?[...v20]:V.map((v,j)=>v-m1*w[j]/M),p1=v1.map(v=>m1*v),p2=v2.map(v=>m2*v),momentum=p1.map((v,j)=>v+p2[j]),force1=z.map(v=>-k*v),force2=force1.map(v=>-v),L=crossN(q1,p1)+crossN(q2,p2),LD=crossN(q1.map((v,j)=>v-d[j]),p1)+crossN(q2.map((v,j)=>v-d[j]),p2),kinetic=(m1*dotN(v1,v1)+m2*dotN(v2,v2))/2,potential=k*dotN(z,z)/2,energy=kinetic+potential,centerKinetic=dotN(momentum,momentum)/(2*M),relativeKinetic=mu*dotN(w,w)/2,relativeL=mu*crossN(z,w),centerL=crossN(center,momentum);
  return{time:t,position1:q1,position2:q2,velocity1:v1,velocity2:v2,momentum1:p1,momentum2:p2,momentum,force1,force2,center,relativePosition:z,relativeVelocity:w,angularMomentum:L,shiftedAngularMomentum:LD,totalTorque:crossN(q1,force1)+crossN(q2,force2),kinetic,potential,energy,centerKinetic,relativeKinetic,centerAngularMomentum:centerL,relativeAngularMomentum:relativeL,momentumResidual:momentum.map((v,j)=>v-P0[j]),angularResidual:L-numeric(L0r),shiftedAngularResidual:LD-numeric(LD0r),energyResidual:energy-numeric(initialEr),kineticDecompositionResidual:kinetic-centerKinetic-relativeKinetic,angularDecompositionResidual:L-centerL-relativeL,originShiftResidual:LD-(L-crossN(d,momentum))};
 };
 const query=state(T),curves:Vec2[][]=Array.from({length:11},()=>[]);
 for(let j=0;j<=200;j++){const r=state(j/20);curves[0].push(r.position1 as Vec2);curves[1].push(r.position2 as Vec2);curves[2].push(r.center as Vec2);[r.momentum[0],r.momentum1[0],r.momentum2[0],r.angularMomentum,r.shiftedAngularMomentum,r.kinetic,r.potential,r.energy].forEach((v,i)=>curves[i+3].push([j/20,v]));}
 return{mass1:m1,mass2:m2,stiffness:k,reducedMass:mu,omega,time:T,comparisonOrigin:d,initialMomentum:list(P0r),initialAngularMomentum:expose(L0r),initialShiftedAngularMomentum:expose(LD0r),initialKinetic:expose(initialKr),initialPotential:expose(initialVr),initialEnergy:expose(initialEr),query,curves};
}

function sinc(x:number){if(Math.abs(x)<1e-4){const z=x*x;return 1-z/6+z*z/120-z*z*z/5040;}return Math.sin(x)/x;}
/** External skew force rotates velocity without supplying kinetic work. */
export function gyroscopicConservation(p:number[]) {
 controls(p,[[1,4],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[0,100]]);
 const [mass,oc,xc,yc,vxc,vyc,tc]=p,Omega=oc/10,q0r=[D(xc),D(yc)],v0r=[D(vxc),D(vyc)],q0=q0r.map(numeric),v0=v0r.map(numeric),Jv0=rotateQuarter(v0),omr=D(oc),initialKr=mul(half,mul(R(mass),dot(v0r,v0r))),initialLr=mul(R(mass),cross(q0r,v0r)),initialModifiedr=sub(initialLr,mul(half,mul(R(mass),mul(omr,dot(q0r,q0r))))),Cr=oc===0?null:q0r.map((v,j)=>add(v,div(j===0?neg(v0r[1]):v0r[0],omr))),center=Cr?.map(numeric)??null,stationary=dot(v0r,v0r).n===Z,period=oc===0?null:2*Math.PI/Math.abs(Omega),T=tc/10;
 const state=(t:number)=>{
  let q:number[],v:number[];
  if(Omega===0){q=q0.map((x,j)=>x+v0[j]*t);v=[...v0];}
  else{const theta=Omega*t,c=Math.cos(theta),s=Math.sin(theta),A=t*sinc(theta),B=Omega*t*t*sinc(theta/2)**2/2;q=q0.map((x,j)=>x+A*v0[j]+B*Jv0[j]);v=v0.map((x,j)=>c*x+s*Jv0[j]);}
  if(t===0){q=[...q0];v=[...v0];}
  const momentum=v.map(v=>mass*v),force=rotateQuarter(v).map(v=>mass*Omega*v),kinetic=mass*dotN(v,v)/2,L=crossN(q,momentum),modified=L-mass*Omega*dotN(q,q)/2,torque=crossN(q,force),power=dotN(v,force),theta=Omega*t,impulse=v0.map((x,j)=>mass*(-2*Math.sin(theta/2)**2*x+Math.sin(theta)*Jv0[j])),qCenter=Omega===0?null:q.map((x,j)=>x+rotateQuarter(v)[j]/Omega);
  return{time:t,position:q,velocity:v,momentum,force,kinetic,angularMomentum:L,modifiedAngularMomentum:modified,torque,power,externalWork:0,externalImpulse:impulse,momentumBalanceResidual:momentum.map((v,j)=>v-mass*v0[j]-impulse[j]),energyResidual:kinetic-numeric(initialKr),modifiedResidual:modified-numeric(initialModifiedr),angularBalanceResidual:L-numeric(initialLr)-mass*Omega*(dotN(q,q)-dotN(q0,q0))/2,guidingCenter:qCenter,guidingCenterResidual:qCenter?.map((v,j)=>v-center![j])??null};
 };
 const query=state(T),curves:Vec2[][]=Array.from({length:6},()=>[]),spatialHorizon=period??10;
 for(let j=0;j<=200;j++){curves[0].push(state(spatialHorizon*j/200).position as Vec2);const r=state(j/20);[r.momentum[0],r.momentum[1],r.angularMomentum,r.modifiedAngularMomentum,r.kinetic].forEach((v,i)=>curves[i+1].push([j/20,v]));}
 return{mass,Omega,time:T,initialPosition:q0,initialVelocity:v0,initialKinetic:expose(initialKr),initialAngularMomentum:expose(initialLr),initialModifiedAngularMomentum:expose(initialModifiedr),guidingCenter: center,exactGuidingCenter:Cr?list(Cr):null,period,stationary,linearMomentumConserved:oc===0||stationary,mechanicalAngularConserved:oc===0||stationary||(Cr?.every(v=>v.n===Z)??false),query,curves};
}
