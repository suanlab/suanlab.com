import type {Vec2} from './learning-simulations';

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
const add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b));
const mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n),square=(a:Rat)=>mul(a,a);
const sign=(a:Rat)=>a.n<Z?-1:a.n>Z?1:0,abs=(a:Rat)=>a.n<Z?neg(a):a;
const compare=(a:Rat,b:Rat)=>sign(sub(a,b));
const half=R(1,2);
function expose(a:Rat){return {numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const number=(a:Rat)=>expose(a).value,list=(v:Rat[])=>v.map(expose);
function controls(p:number[],ranges:readonly(readonly[number,number])[]) {
  if(p.length!==ranges.length)throw new Error('Use the stated rigid-contact controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
const displacement=(v:Rat,a:Rat,t:Rat)=>add(mul(v,t),mul(half,mul(a,square(t))));
/** Exact total travel, including a velocity reversal inside the interval. */
function travel(v:Rat,a:Rat,t:Rat) {
  const dx=displacement(v,a,t),turn=sign(v)*sign(a)<0?div(neg(v),a):null;
  if(turn&&compare(turn,R(0))>0&&compare(turn,t)<0){const x=displacement(v,a,turn);return add(abs(x),abs(sub(dx,x)));}
  return abs(dx);
}

/** Single instantaneous frictionless normal impact of a tilted uniform rod.
 * [mass,2*horizontalLever,2*COMheight,10*Vx,10*Vy,10*CCWomega,10*restitution].
 * Geometry is frozen: lower endpoint(a,0),COM(0,h),upper endpoint(-a,2h).
 * h>0 makes the lower endpoint the only initial floor contact.
 */
export function frictionlessRodImpact(p:number[]) {
  controls(p,[[1,4],[0,6],[1,6],[-20,20],[-20,20],[-20,20],[0,10]]);
  const [mass,ac,hc,vxc,vyc,wc,ec]=p,M=R(mass),a=R(ac,2),h=R(hc,2),vx=R(vxc,10),vy=R(vyc,10),omega=R(wc,10),e=R(ec,10);
  const inertia=div(mul(M,add(square(a),square(h))),R(3)),W=add(div(R(1),M),div(square(a),inertia)),incomingNormal=add(vy,mul(a,omega)),active=sign(incomingNormal)<0,impulse=active?div(neg(mul(add(R(1),e),incomingNormal)),W):R(0);
  const afterVy=add(vy,div(impulse,M)),afterOmega=add(omega,div(mul(a,impulse),inertia));
  const state=(Vy:Rat,w:Rat)=>{
    const momentum=[mul(M,vx),mul(M,Vy)],spin=mul(inertia,w),orbital=sub(neg(mul(a,momentum[1])),mul(h,momentum[0])),translation=mul(half,mul(M,add(square(vx),square(Vy)))),rotation=mul(half,mul(inertia,square(w)));
    return {velocity:list([vx,Vy]),omega:expose(w),contactVelocity:list([add(vx,mul(h,w)),add(Vy,mul(a,w))]),momentum:list(momentum),spinAngularMomentum:expose(spin),orbitalAngularMomentumAboutContact:expose(orbital),angularMomentumAboutContact:expose(add(spin,orbital)),kineticTranslation:expose(translation),kineticRotation:expose(rotation),kinetic:expose(add(translation,rotation))};
  };
  const before=state(vy,omega),after=state(afterVy,afterOmega),kinetic=(Vy:Rat,w:Rat)=>add(mul(half,mul(M,add(square(vx),square(Vy)))),mul(half,mul(inertia,square(w)))),loss=sub(kinetic(vy,omega),kinetic(afterVy,afterOmega));
  const predictedLoss=active?div(mul(sub(R(1),square(e)),square(incomingNormal)),mul(R(2),W)):R(0),normalAfter=add(afterVy,mul(a,afterOmega));
  const L=(Vy:Rat,w:Rat)=>sub(sub(mul(inertia,w),mul(a,mul(M,Vy))),mul(h,mul(M,vx)));
  const geometry={center:[0,number(h)]as Vec2,contact:[number(a),0]as Vec2,upperEndpoint:[-number(a),2*number(h)]as Vec2};
  return {mass,horizontalLever:expose(a),height:expose(h),length:2*Math.hypot(number(a),number(h)),inertia:expose(inertia),restitution:expose(e),inverseEffectiveNormalMass:expose(W),effectiveNormalMass:expose(div(R(1),W)),normalImpulse:expose(impulse),tangentialImpulse:expose(R(0)),active,nonclosing:!active,before,after,energyLoss:expose(loss),predictedEnergyLoss:expose(predictedLoss),energyLossResidual:expose(sub(loss,predictedLoss)),normalMomentumResidual:expose(sub(mul(M,sub(afterVy,vy)),impulse)),spinImpulseResidual:expose(sub(mul(inertia,sub(afterOmega,omega)),mul(a,impulse))),contactAngularResidual:expose(sub(L(afterVy,afterOmega),L(vy,omega))),restitutionResidual:active?expose(add(normalAfter,mul(e,incomingNormal))):null,geometry,instantaneousOnly:true as const,frictionless:true as const,persistentContactSolved:false as const};
}

/** Continuous wheel with clockwise Omega and slip u=V-R*Omega.
 * [mass,2R,10muK,10staticMargin,10kappa,10F,10tauCW,10V0,10Omega0,10t].
 * Fixed COM heightR, normal loadmg,g10; I=kappa*m*R² with0<kappa<=1.
 * Constant drive/torque; exact switching when relative contact slip vanishes.
 */
export function drivenWheelContact(p:number[]) {
  controls(p,[[1,4],[1,6],[0,10],[0,10],[1,10],[-100,100],[-100,100],[-20,20],[-20,20],[0,100]]);
  const [mass,rc,kc,dc,kappaCode,fc,tc,v0c,w0c,timeCode]=p,M=R(mass),radius=R(rc,2),muK=R(kc,10),muS=R(kc+dc,10),kappa=R(kappaCode,10),inertia=mul(kappa,mul(M,square(radius))),normal=R(10*mass),fk=mul(muK,normal),fs=mul(muS,normal),drive=R(fc,10),torque=R(tc,10),v0=R(v0c,10),w0=R(w0c,10),T=R(timeCode,10);
  const slip0=sub(v0,mul(radius,w0)),freeSlipAcceleration=sub(div(drive,M),div(mul(radius,torque),inertia)),coupling=add(div(R(1),M),div(square(radius),inertia)),required=div(neg(freeSlipAcceleration),coupling),canRoll=compare(abs(required),fs)<=0;
  const restFriction=canRoll?required:mul(R(-sign(freeSlipAcceleration)),fk),initialFriction=sign(slip0)===0?restFriction:mul(R(-sign(slip0)),fk);
  const acceleration=(friction:Rat)=>({linear:div(add(drive,friction),M),angular:div(sub(torque,mul(radius,friction)),inertia),slip:add(freeSlipAcceleration,mul(coupling,friction))});
  const initialA=acceleration(initialFriction),restA=acceleration(restFriction),stop=sign(slip0)*sign(initialA.slip)<0?div(neg(slip0),initialA.slip):null;
  const stopX=stop?displacement(v0,initialA.linear,stop):null,stopPhi=stop?displacement(w0,initialA.angular,stop):null,stopV=stop?add(v0,mul(initialA.linear,stop)):null,stopOmega=stop?add(w0,mul(initialA.angular,stop)):null;
  const initialTranslation=mul(half,mul(M,square(v0))),initialRotation=mul(half,mul(inertia,square(w0))),initialKinetic=add(initialTranslation,initialRotation);
  const raw=(t:Rat,beforeEvent=false)=>{
    const reached=stop&&(compare(t,stop)>0||(compare(t,stop)===0&&!beforeEvent));
    const durations:{time:Rat;friction:Rat;linear:Rat;angular:Rat;slipA:Rat;V:Rat;Omega:Rat;u:Rat}[]=reached?[
      {time:stop!,friction:initialFriction,linear:initialA.linear,angular:initialA.angular,slipA:initialA.slip,V:v0,Omega:w0,u:slip0},
      {time:sub(t,stop!),friction:restFriction,linear:restA.linear,angular:restA.angular,slipA:restA.slip,V:stopV!,Omega:stopOmega!,u:R(0)},
    ]:[{time:t,friction:initialFriction,linear:initialA.linear,angular:initialA.angular,slipA:initialA.slip,V:v0,Omega:w0,u:slip0}];
    let x=R(0),phi=R(0),distance=R(0),slipDistance=R(0),frictionImpulse=R(0),frictionTranslationWork=R(0),frictionRotationWork=R(0);
    for(const d of durations){const dx=displacement(d.V,d.linear,d.time),dp=displacement(d.Omega,d.angular,d.time);x=add(x,dx);phi=add(phi,dp);distance=add(distance,travel(d.V,d.linear,d.time));slipDistance=add(slipDistance,travel(d.u,d.slipA,d.time));frictionImpulse=add(frictionImpulse,mul(d.friction,d.time));frictionTranslationWork=add(frictionTranslationWork,mul(d.friction,dx));frictionRotationWork=sub(frictionRotationWork,mul(radius,mul(d.friction,dp)));}
    const last=durations[durations.length-1],V=add(last.V,mul(last.linear,last.time)),Omega=add(last.Omega,mul(last.angular,last.time)),u=sub(V,mul(radius,Omega)),translation=mul(half,mul(M,square(V))),rotation=mul(half,mul(inertia,square(Omega))),K=add(translation,rotation),Q=mul(fk,slipDistance),forceWork=mul(drive,x),torqueWork=mul(torque,phi),work=add(forceWork,torqueWork),contactWork=add(frictionTranslationWork,frictionRotationWork),rolling=(slip0.n===Z||!!reached)&&canRoll,friction=last.friction;
    const linearImpulse=add(mul(drive,t),frictionImpulse),angularImpulse=sub(mul(torque,t),mul(radius,frictionImpulse)),contactPower=mul(friction,u);
    return {x,phi,V,Omega,u,friction,linearAcceleration:last.linear,angularAcceleration:last.angular,slipAcceleration:last.slipA,distance,slipDistance,signedSlip:sub(x,mul(radius,phi)),translation,rotation,K,Q,forceWork,torqueWork,work,frictionImpulse,frictionTranslationWork,frictionRotationWork,contactWork,contactPower,heatRate:mul(fk,abs(u)),linearImpulse,angularImpulse,energyResidual:sub(add(sub(K,initialKinetic),Q),work),contactHeatResidual:add(contactWork,Q),linearImpulseResidual:sub(mul(M,sub(V,v0)),linearImpulse),angularImpulseResidual:sub(mul(inertia,sub(Omega,w0)),angularImpulse),rolling,incipientSliding:!rolling&&u.n===Z};
  };
  const exposeState=(s:ReturnType<typeof raw>)=>({position:expose(s.x),clockwiseAngle:expose(s.phi),velocity:expose(s.V),clockwiseOmega:expose(s.Omega),slipVelocity:expose(s.u),friction:expose(s.friction),linearAcceleration:expose(s.linearAcceleration),angularAcceleration:expose(s.angularAcceleration),slipAcceleration:expose(s.slipAcceleration),COMdistance:expose(s.distance),slipDistance:expose(s.slipDistance),signedSlipDisplacement:expose(s.signedSlip),kineticTranslation:expose(s.translation),kineticRotation:expose(s.rotation),kinetic:expose(s.K),heat:expose(s.Q),forceWork:expose(s.forceWork),torqueWork:expose(s.torqueWork),externalWork:expose(s.work),frictionImpulse:expose(s.frictionImpulse),frictionTranslationWork:expose(s.frictionTranslationWork),frictionRotationWork:expose(s.frictionRotationWork),contactWork:expose(s.contactWork),contactPower:expose(s.contactPower),heatRate:expose(s.heatRate),linearImpulse:expose(s.linearImpulse),clockwiseAngularImpulse:expose(s.angularImpulse),energyResidual:expose(s.energyResidual),contactHeatResidual:expose(s.contactHeatResidual),linearImpulseResidual:expose(s.linearImpulseResidual),angularImpulseResidual:expose(s.angularImpulseResidual),rolling:s.rolling,incipientSliding:s.incipientSliding});
  const samples=new Map<string,Rat>();for(let j=0;j<=200;j++){const t=R(j,20);samples.set(expose(t).label,t);}samples.set(expose(T).label,T);if(stop&&compare(stop,R(0))>0&&compare(stop,R(10))<=0)samples.set(expose(stop).label,stop);
  const curves:Vec2[][]=Array.from({length:11},()=>[]);
  for(const t of [...samples.values()].sort(compare)){const s=raw(t),time=number(t);[s.x,s.V,mul(radius,s.Omega),s.u].forEach((v,j)=>curves[j].push([time,number(v)]));if(stop&&compare(t,stop)===0)curves[4].push([time,number(raw(t,true).friction)]);curves[4].push([time,number(s.friction)]);[s.translation,s.rotation,s.K,s.Q,add(initialKinetic,s.work),s.phi].forEach((v,j)=>curves[j+5].push([time,number(v)]));}
  const query=exposeState(raw(T)),rn=number(radius),angle=query.clockwiseAngle.value;
  return {mass,gravity:10,radius:expose(radius),kappa:expose(kappa),inertia:expose(inertia),kineticCoefficient:expose(muK),staticCoefficient:expose(muS),normalLoad:expose(normal),kineticLimit:expose(fk),staticLimit:expose(fs),drive:expose(drive),clockwiseTorque:expose(torque),initialVelocity:expose(v0),initialClockwiseOmega:expose(w0),initialSlip:expose(slip0),initialKinetic:expose(initialKinetic),requiredRollingFriction:expose(required),rollingFeasible:canRoll,time:expose(T),zeroSlipEventTime:stop?expose(stop):null,eventReached:stop?compare(T,stop)>=0:false,eventPosition:stopX?expose(stopX):null,eventClockwiseAngle:stopPhi?expose(stopPhi):null,eventVelocity:stopV?expose(stopV):null,eventClockwiseOmega:stopOmega?expose(stopOmega):null,rollsAfterEvent:stop?canRoll:null,query,curves,queryGeometry:{center:[query.position.value,rn]as Vec2,contact:[query.position.value,0]as Vec2,materialMark:[query.position.value+rn*Math.cos(angle),rn-rn*Math.sin(angle)]as Vec2},normalImpactSolved:false as const,clockwisePositive:true as const};
}
