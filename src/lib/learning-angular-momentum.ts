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

/** Two fixed inertial reference points; changing either reference does not change motion. */
export function fixedOriginAngularMomentum(p:number[]) {
 controls(p,[[1,5],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[0,100]]);
 const [mc,qxc,qyc,vxc,vyc,fxc,fyc,oxc,oyc,dxc,dyc,tc]=p,M=R(mc),q0=[D(qxc),D(qyc)],v0=[D(vxc),D(vyc)],F=[D(fxc),D(fyc)],O=[D(oxc),D(oyc)],d=[D(dxc),D(dyc)],Q=O.map((v,j)=>add(v,d[j])),T=D(tc),a=F.map(v=>div(v,M)),slope=cross(v0,F);
 const state=(t:Rat)=>{
  const q=q0.map((v,j)=>add(add(v,mul(v0[j],t)),mul(half,mul(a[j],mul(t,t))))),v=v0.map((v,j)=>add(v,mul(a[j],t))),momentum=v.map(v=>mul(M,v)),r=q.map((v,j)=>sub(v,O[j])),rs=q.map((v,j)=>sub(v,Q[j]));
  return{q,v,momentum,r,rs,L:cross(r,momentum),shiftedL:cross(rs,momentum),torque:cross(r,F),shiftedTorque:cross(rs,F)};
 };
 const s=state(T),initial=state(R(0)),impulse=add(mul(initial.torque,T),mul(half,mul(slope,mul(T,T)))),shiftedImpulse=add(mul(initial.shiftedTorque,T),mul(half,mul(slope,mul(T,T)))),angularResidual=sub(sub(s.L,initial.L),impulse),shiftedAngularResidual=sub(sub(s.shiftedL,initial.shiftedL),shiftedImpulse),originLResidual=sub(s.shiftedL,sub(s.L,cross(d,s.momentum))),originTorqueResidual=sub(s.shiftedTorque,sub(s.torque,cross(d,F))),curves:Vec2[][]=[[],[],[],[],[]];
 for(let j=0;j<=200;j++){const t=rat(B(j),B(20)),r=state(t);curves[0].push(r.q.map(numeric)as Vec2);[r.L,r.shiftedL,r.torque,r.shiftedTorque].forEach((v,k)=>curves[k+1].push([j/20,numeric(v)]));}
 return{mass:mc,time:expose(T),initialPosition:list(q0),initialVelocity:list(v0),force:list(F),referenceOrigin:list(O),comparisonOrigin:list(Q),originShift:list(d),position:list(s.q),velocity:list(s.v),momentum:list(s.momentum),radius:list(s.r),shiftedRadius:list(s.rs),initialAngularMomentum:expose(initial.L),angularMomentum:expose(s.L),initialShiftedAngularMomentum:expose(initial.shiftedL),shiftedAngularMomentum:expose(s.shiftedL),initialTorque:expose(initial.torque),torque:expose(s.torque),initialShiftedTorque:expose(initial.shiftedTorque),shiftedTorque:expose(s.shiftedTorque),torqueSlope:expose(slope),angularImpulse:expose(impulse),shiftedAngularImpulse:expose(shiftedImpulse),angularResidual:expose(angularResidual),shiftedAngularResidual:expose(shiftedAngularResidual),originLResidual:expose(originLResidual),originTorqueResidual:expose(originTorqueResidual),conservedAboutReference:initial.torque.n===Z&&slope.n===Z,conservedAboutComparison:initial.shiftedTorque.n===Z&&slope.n===Z,zeroQueryTorque:s.torque.n===Z,curves};
}

/** Central harmonic force about the fixed origin, with a distinct fixed comparison point. */
export function centralHarmonicAngularMomentum(p:number[]) {
 controls(p,[[1,4],[1,4],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[0,100]]);
 const [mass,stiffness,qxc,qyc,vxc,vyc,dxc,dyc,tc]=p,q0r=[D(qxc),D(qyc)],v0r=[D(vxc),D(vyc)],q0=q0r.map(numeric),v0=v0r.map(numeric),d=[dxc/10,dyc/10],omega=Math.sqrt(stiffness/mass),T=tc/10,L0=cross(q0r,v0r),exactL=mul(R(mass),L0),Lref=numeric(exactL),initialEnergy=mass*(v0[0]**2+v0[1]**2)/2+stiffness*(q0[0]**2+q0[1]**2)/2,period=2*Math.PI/omega;
 const state=(t:number)=>{
  const c=Math.cos(omega*t),s=Math.sin(omega*t),q=t===0?[...q0]:q0.map((x,j)=>x*c+v0[j]*s/omega),v=t===0?[...v0]:q0.map((x,j)=>-omega*x*s+v0[j]*c),momentum=v.map(v=>mass*v),force=q.map(x=>-stiffness*x),r=q.map((x,j)=>x-d[j]),L=crossN(q,momentum),shiftedL=crossN(r,momentum),torque=crossN(q,force),shiftedTorque=crossN(r,force),energy=mass*(v[0]**2+v[1]**2)/2+stiffness*(q[0]**2+q[1]**2)/2,area=Lref*t/(2*mass),shiftedArea=area-crossN(d,q.map((x,j)=>x-q0[j]))/2;
  return{time:t,position:q,velocity:v,momentum,force,angularMomentum:L,shiftedAngularMomentum:shiftedL,torque,shiftedTorque,arealVelocity:L/(2*mass),shiftedArealVelocity:shiftedL/(2*mass),sweptArea:area,shiftedSweptArea:shiftedArea,energy,angularResidual:L-Lref,energyResidual:energy-initialEnergy,originLResidual:shiftedL-(L-crossN(d,momentum)),originTorqueResidual:shiftedTorque-(torque-crossN(d,force))};
 };
 const query=state(T),initial=state(0),curves:Vec2[][]=Array.from({length:7},()=>[]);
 for(let j=0;j<=200;j++){const orbit=state(period*j/200),r=state(j/20);curves[0].push(orbit.position as Vec2);[r.angularMomentum,r.shiftedAngularMomentum,r.torque,r.shiftedTorque,r.sweptArea,r.shiftedSweptArea].forEach((v,k)=>curves[k+1].push([j/20,v]));}
 const zeroPosition=dot(q0r,q0r).n===Z,zeroVelocity=dot(v0r,v0r).n===Z;
 return{mass,stiffness,omega,period,time:T,initialPosition:q0,initialVelocity:v0,comparisonOrigin:d,initialAngularMomentum:expose(exactL),exactArealVelocity:expose(div(exactL,R(2*mass))),initialEnergy,radialMotion:L0.n===Z,stationary:zeroPosition&&zeroVelocity,circularMotion:!zeroPosition&&dot(q0r,v0r).n===Z&&sub(mul(R(mass),dot(v0r,v0r)),mul(R(stiffness),dot(q0r,q0r))).n===Z,query,initial,curves};
}
