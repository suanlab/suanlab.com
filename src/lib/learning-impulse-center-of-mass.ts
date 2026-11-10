import type {Vec2} from './learning-simulations';
const B=BigInt,Z=B(0),ONE=B(1);type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat{if(d===Z)throw new Error('Zero rational denominator.');if(d<Z){n=-n;d=-d;}let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}return{n:n/a,d:d/a};}
const R=(n:number)=>rat(B(n)),D=(n:number)=>rat(B(n),B(10)),add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b)),mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n);
function expose(a:Rat){return{numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const numeric=(a:Rat)=>expose(a).value;
function controls(p:number[],ranges:readonly(readonly[number,number])[]){if(p.length!==ranges.length)throw new Error('Use the stated classical-mechanics controls.');p.forEach((v,j)=>{if(!Number.isInteger(v)||v<ranges[j][0]||v>ranges[j][1])throw new Error('Use bounded integer control codes.');});}
const half=rat(ONE,B(2));

/** Finite prescribed pulses. Values at the switch use the right-hand branch. */
export function finiteForcePulse(p:number[]) {
 controls(p,[[1,5],[-30,30],[-30,30],[1,50],[0,100]]);
 const [mass,fc,vc,dc,tc]=p,m=R(mass),F=D(fc),v0=D(vc),duration=D(dc),T=D(tc);
 const state=(t:Rat,ramp:boolean)=>{
  const u=t.n*duration.d<duration.n*t.d?t:duration,u2=mul(u,u);
  const J=ramp?div(mul(F,u2),duration):mul(F,u);
  const position=add(mul(v0,t),ramp?div(mul(F,sub(mul(u2,t),mul(rat(B(2),B(3)),mul(u2,u)))) ,mul(m,duration)):div(mul(F,sub(mul(u,t),mul(half,u2))),m));
  const velocity=add(v0,div(J,m)),kinetic=mul(half,mul(m,mul(velocity,velocity))),initialKinetic=mul(half,mul(m,mul(v0,v0)));
  const work=add(mul(v0,J),div(mul(J,J),mul(R(2),m))),force=t.n*duration.d>=duration.n*t.d?R(0):ramp?div(mul(R(2),mul(F,t)),duration):F;
  return{force:expose(force),impulse:expose(J),velocity:expose(velocity),position:expose(position),kinetic:expose(kinetic),work:expose(work),momentumResidual:expose(sub(mul(m,sub(velocity,v0)),J)),workResidual:expose(sub(sub(kinetic,initialKinetic),work))};
 };
 const flat=state(T,false),ramp=state(T,true),curves:Vec2[][]=[[],[],[],[]];
 for(let j=0;j<=200;j++){const t=rat(B(j),B(20)),a=state(t,false),b=state(t,true);curves[0].push([j/20,a.impulse.value]);curves[1].push([j/20,b.impulse.value]);curves[2].push([j/20,a.position.value]);curves[3].push([j/20,b.position.value]);}
 const tau=numeric(duration),f=numeric(F),forceCurves:Vec2[][]=[[[0,f],[tau,f],[tau,0],[10,0]],[[0,0],[tau,2*f],[tau,0],[10,0]]];
 return{mass,time:expose(T),duration:expose(duration),amplitude:expose(F),initialVelocity:expose(v0),flat,ramp,totalImpulse:expose(mul(F,duration)),positionGap:expose(sub(rat(B(flat.position.numerator),B(flat.position.denominator)),rat(B(ramp.position.numerator),B(ramp.position.denominator)))),finalPositionGap:expose(div(mul(F,mul(duration,duration)),mul(R(6),m))),pulseFinished:tc>=dc,curves,forceCurves};
}

/** Two point masses; +f and -f are a declared internal interaction pair. */
export function twoParticleCenterOfMass(p:number[]) {
 controls(p,[[1,5],[1,5],[-30,30],[-30,30],[-30,30],[-30,30],[-30,30],[-30,30],[-30,30],[0,50],[0,2]]);
 const [m1c,m2c,q1c,q2c,v1c,v2c,fc,e1c,e2c,tc,selectedSystem]=p;
 const m1=R(m1c),m2=R(m2c),M=add(m1,m2),mu=div(mul(m1,m2),M),q10=D(q1c),q20=D(q2c),v10=D(v1c),v20=D(v2c),f=D(fc),e1=D(e1c),e2=D(e2c),net=add(e1,e2),a1=div(add(f,e1),m1),a2=div(sub(e2,f),m2),T=D(tc);
 const law=(q:Rat,v:Rat,a:Rat,t:Rat)=>add(add(q,mul(v,t)),mul(half,mul(a,mul(t,t))));
 const state=(t:Rat)=>{const q1=law(q10,v10,a1,t),q2=law(q20,v20,a2,t),v1=add(v10,mul(a1,t)),v2=add(v20,mul(a2,t)),P=add(mul(m1,v1),mul(m2,v2)),K=mul(half,add(mul(m1,mul(v1,v1)),mul(m2,mul(v2,v2)))),potential=neg(mul(f,sub(q1,q2))),com=div(add(mul(m1,q1),mul(m2,q2)),M);return{q1,q2,v1,v2,P,K,potential,com};};
 const r=state(T),initial=state(R(0)),dq1=sub(r.q1,q10),dq2=sub(r.q2,q20),internalWork=mul(f,sub(dq1,dq2)),externalWork=add(mul(e1,dq1),mul(e2,dq2)),relativeK=mul(half,mul(mu,mul(sub(r.v1,r.v2),sub(r.v1,r.v2)))),comK=div(mul(r.P,r.P),mul(R(2),M));
 const selectedMomentum=selectedSystem===0?r.P:selectedSystem===1?mul(m1,r.v1):mul(m2,r.v2),selectedInitialMomentum=selectedSystem===0?initial.P:selectedSystem===1?mul(m1,v10):mul(m2,v20),selectedExternalForce=selectedSystem===0?net:selectedSystem===1?add(f,e1):sub(e2,f);
 const raw={position1:r.q1,position2:r.q2,velocity1:r.v1,velocity2:r.v2,acceleration1:a1,acceleration2:a2,totalMomentum:r.P,initialMomentum:initial.P,externalForce:net,centerOfMass:r.com,centerVelocity:div(r.P,M),centerAcceleration:div(net,M),kinetic:r.K,initialKinetic:initial.K,potential:r.potential,initialPotential:initial.potential,relativeKinetic:relativeK,centerKinetic:comK,internalWork,externalWork,selectedMomentum,selectedInitialMomentum,selectedExternalForce,selectedImpulse:mul(selectedExternalForce,T),selectedMomentumResidual:sub(sub(selectedMomentum,selectedInitialMomentum),mul(selectedExternalForce,T)),momentumResidual:sub(sub(r.P,initial.P),mul(net,T)),kineticDecompositionResidual:sub(r.K,add(relativeK,comK)),workResidual:sub(sub(r.K,initial.K),add(internalWork,externalWork)),mechanicalResidual:sub(sub(add(r.K,r.potential),add(initial.K,initial.potential)),externalWork)};
 const exact=Object.fromEntries(Object.entries(raw).map(([k,v])=>[k,expose(v)])) as {[K in keyof typeof raw]:ReturnType<typeof expose>};
 const curves:Vec2[][]=[[],[],[],[],[]];
 for(let j=0;j<=100;j++){const t=rat(B(j),B(20)),s=state(t);[s.q1,s.q2,s.com,s.P,add(initial.P,mul(net,t))].forEach((v,i)=>curves[i].push([j/20,numeric(v)]));}
 return{...exact,mass1:m1c,mass2:m2c,time:expose(T),interactionForce:expose(f),externalForce1:expose(e1),externalForce2:expose(e2),selectedSystem,momentumConserved:net.n===Z,selectedMomentumConserved:selectedExternalForce.n===Z,curves};
}
