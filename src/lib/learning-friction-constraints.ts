import type {Vec2} from './learning-simulations';
const B=BigInt,Z=B(0),ONE=B(1);type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat{if(d===Z)throw new Error('Zero rational denominator.');if(d<Z){n=-n;d=-d;}let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}return{n:n/a,d:d/a};}
const R=(n:number)=>rat(B(n)),D=(n:number)=>rat(B(n),B(10)),add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b)),mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n);
function expose(a:Rat){return{numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const numeric=(a:Rat)=>expose(a).value;
function controls(p:number[],ranges:readonly(readonly[number,number])[]){if(p.length!==ranges.length)throw new Error('Use the stated classical-mechanics controls.');p.forEach((v,j)=>{if(!Number.isInteger(v)||v<ranges[j][0]||v>ranges[j][1])throw new Error('Use bounded integer control codes.');});}


const half=rat(ONE,B(2)),abs=(x:Rat)=>x.n<Z?neg(x):x,sign=(x:Rat)=>x.n<Z?-1:x.n>Z?1:0,compare=(a:Rat,b:Rat)=>{const d=a.n*b.d-b.n*a.d;return d<Z?-1:d>Z?1:0;};
function sampleTimes(event:Rat|null){const map=new Map<string,Rat>();for(let j=0;j<=200;j++){const t=rat(B(j),B(20));map.set(expose(t).label,t);}if(event&&compare(event,R(0))>0&&compare(event,R(10))<=0)map.set(expose(event).label,event);return[...map.values()].sort(compare);}

/** Event-exact Coulomb model with an explicit sticking convention at the static threshold. */
export function dryFrictionMotion(p:number[]) {
 controls(p,[[1,4],[0,10],[0,10],[-100,100],[-20,20],[0,100]]);
 const [mass,kc,dc,fc,vc,tc]=p,M=R(mass),muK=D(kc),muS=D(kc+dc),N=R(10*mass),fk=mul(muK,N),fs=mul(muS,N),F=D(fc),v0=D(vc),T=D(tc),initialK=mul(half,mul(M,mul(v0,v0))),canStick=compare(abs(F),fs)<=0;
 const restFriction=canStick?neg(F):mul(R(-sign(F)),fk),restA=div(add(F,restFriction),M),initialFriction=sign(v0)===0?restFriction:mul(R(-sign(v0)),fk),initialA=div(add(F,initialFriction),M),stop=sign(v0)!==0&&sign(v0)*sign(initialA)<0?div(neg(v0),initialA):null;
 const displacement=(v:Rat,a:Rat,t:Rat)=>add(mul(v,t),mul(half,mul(a,mul(t,t)))),stopX=stop?displacement(v0,initialA,stop):null;
 const raw=(t:Rat,beforeEvent=false)=>{
  let x:Rat,v:Rat,friction:Rat,a:Rat,distance:Rat,frictionImpulse:Rat,frictionWork:Rat,sticking:boolean;
  if(v0.n===Z){x=mul(half,mul(restA,mul(t,t)));v=mul(restA,t);friction=restFriction;a=restA;distance=abs(x);frictionImpulse=mul(friction,t);frictionWork=mul(friction,x);sticking=canStick;}
  else if(stop&&(compare(t,stop)>0||(compare(t,stop)===0&&!beforeEvent))){const dt=sub(t,stop),extra=mul(half,mul(restA,mul(dt,dt)));x=add(stopX!,extra);v=mul(restA,dt);friction=restFriction;a=restA;distance=add(abs(stopX!),abs(extra));frictionImpulse=add(mul(initialFriction,stop),mul(restFriction,dt));frictionWork=add(mul(initialFriction,stopX!),mul(restFriction,extra));sticking=canStick;}
  else{x=displacement(v0,initialA,t);v=add(v0,mul(initialA,t));friction=initialFriction;a=initialA;distance=abs(x);frictionImpulse=mul(friction,t);frictionWork=mul(friction,x);sticking=false;}
  const kinetic=mul(half,mul(M,mul(v,v))),heat=mul(fk,distance),driveWork=mul(F,x),netImpulse=add(mul(F,t),frictionImpulse),power=mul(friction,v);
  return{x,v,friction,a,distance,kinetic,heat,driveWork,frictionImpulse,frictionWork,frictionPower:power,heatRate:mul(fk,abs(v)),netImpulse,energyResidual:sub(add(sub(kinetic,initialK),heat),driveWork),frictionHeatResidual:add(frictionWork,heat),momentumResidual:sub(mul(M,sub(v,v0)),netImpulse),sticking,incipientSliding:!sticking&&v.n===Z};
 };
 const exposeState=(r:ReturnType<typeof raw>)=>({position:expose(r.x),velocity:expose(r.v),acceleration:expose(r.a),friction:expose(r.friction),distance:expose(r.distance),kinetic:expose(r.kinetic),heat:expose(r.heat),driveWork:expose(r.driveWork),frictionImpulse:expose(r.frictionImpulse),frictionWork:expose(r.frictionWork),frictionPower:expose(r.frictionPower),heatRate:expose(r.heatRate),netImpulse:expose(r.netImpulse),energyResidual:expose(r.energyResidual),frictionHeatResidual:expose(r.frictionHeatResidual),momentumResidual:expose(r.momentumResidual),sticking:r.sticking,incipientSliding:r.incipientSliding});
 const query=exposeState(raw(T)),curves:Vec2[][]=Array.from({length:7},()=>[]);
 for(const t of sampleTimes(stop)){const r=raw(t),tn=numeric(t);curves[0].push([tn,numeric(r.x)]);curves[1].push([tn,numeric(r.v)]);if(stop&&compare(t,stop)===0)curves[2].push([tn,numeric(raw(t,true).friction)]);curves[2].push([tn,numeric(r.friction)]);curves[3].push([tn,numeric(F)]);curves[4].push([tn,numeric(r.kinetic)]);curves[5].push([tn,numeric(r.heat)]);curves[6].push([tn,numeric(add(initialK,r.driveWork))]);}
 return{mass,gravity:10,time:expose(T),kineticCoefficient:expose(muK),staticCoefficient:expose(muS),normalLoad:expose(N),kineticLimit:expose(fk),staticLimit:expose(fs),drive:expose(F),initialVelocity:expose(v0),initialKinetic:expose(initialK),stopTime:stop?expose(stop):null,stopPosition:stopX?expose(stopX):null,sticksAfterStop:stop?canStick:null,eventReached:stop?compare(T,stop)>=0:false,query,curves};
}

/** Compatible motion on a fixed ideal circular guide with a constant signed tangential drive. */
export function circularGuideMotion(p:number[]) {
 controls(p,[[1,4],[1,6],[-20,20],[-24,24],[-20,20],[0,100]]);
 const [mass,rc,fc,angleCode,wc,tc]=p,M=R(mass),radius=rat(B(rc),B(2)),f=D(fc),w0=D(wc),T=D(tc),alpha=div(f,mul(M,radius)),inertia=mul(M,mul(radius,radius)),initialK=mul(half,mul(inertia,mul(w0,w0))),initialL=mul(inertia,w0),theta0=angleCode*Math.PI/12,Rn=numeric(radius),an=numeric(alpha),w0n=numeric(w0);
 const raw=(t:Rat)=>{const w=add(w0,mul(alpha,t)),delta=add(mul(w0,t),mul(half,mul(alpha,mul(t,t)))),normal=neg(mul(M,mul(radius,mul(w,w)))),kinetic=mul(half,mul(inertia,mul(w,w))),work=mul(f,mul(radius,delta)),L=mul(inertia,w),torque=mul(radius,f),theta=theta0+numeric(delta),er=[Math.cos(theta),Math.sin(theta)],et=[-er[1],er[0]],q=er.map(x=>Rn*x),v=et.map(x=>Rn*numeric(w)*x),reaction=er.map(x=>numeric(normal)*x),drive=et.map(x=>numeric(f)*x),a=er.map((x,j)=>-Rn*numeric(w)**2*x+Rn*an*et[j]);return{time:expose(t),omega:expose(w),angularDisplacement:expose(delta),normalReaction:expose(normal),kinetic:expose(kinetic),driveWork:expose(work),reactionWork:expose(R(0)),angularMomentum:expose(L),torque:expose(torque),angularImpulse:expose(mul(torque,t)),drivePower:expose(mul(f,mul(radius,w))),energyResidual:expose(sub(sub(kinetic,initialK),work)),angularResidual:expose(sub(sub(L,initialL),mul(torque,t))),angle:theta,position:q,velocity:v,reaction,driveForce:drive,acceleration:a,reactionPower:reaction[0]*v[0]+reaction[1]*v[1],radiusResidual:q[0]**2+q[1]**2-Rn**2,tangentResidual:q[0]*v[0]+q[1]*v[1],newtonResidual:a.map((x,j)=>mass*x-reaction[j]-drive[j])};};
 const turning=sign(w0)*sign(alpha)<0?div(neg(w0),alpha):null,query=raw(T),curves:Vec2[][]=Array.from({length:8},()=>[]),tn=numeric(T),maxW=Math.max(Math.abs(w0n),Math.abs(w0n+an*tn)),steps=Math.max(1,Math.ceil(maxW*tn/(Math.PI/64)));
 const pathTimes=Array.from({length:steps+1},(_,j)=>tn*j/steps);if(turning&&compare(turning,R(0))>0&&compare(turning,T)<0)pathTimes.push(numeric(turning));pathTimes.sort((a,b)=>a-b);
 for(const t of pathTimes){const theta=theta0+w0n*t+an*t*t/2;curves[0].push([Rn*Math.cos(theta),Rn*Math.sin(theta)]);}
 for(let j=0;j<=128;j++)curves[1].push([Rn*Math.cos(2*Math.PI*j/128),Rn*Math.sin(2*Math.PI*j/128)]);
 for(const t of sampleTimes(turning)){const r=raw(t),time=numeric(t);[r.normalReaction.value,numeric(f),r.kinetic.value,numeric(initialK)+r.driveWork.value,r.angularMomentum.value,numeric(initialL)+r.angularImpulse.value].forEach((v,j)=>curves[j+2].push([time,v]));}
 return{mass,radius:expose(radius),driveMagnitude:expose(f),initialAngleCode:angleCode,initialAngle:theta0,initialOmega:expose(w0),angularAcceleration:expose(alpha),initialKinetic:expose(initialK),initialAngularMomentum:expose(initialL),time:expose(T),turningTime:turning?expose(turning):null,query,curves};
}
