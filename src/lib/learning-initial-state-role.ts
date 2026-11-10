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
const mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),sq=(a:Rat)=>mul(a,a),pow=(a:Rat,n:number)=>{let y=R(1);for(let j=0;j<n;j++)y=mul(y,a);return y;};
const compare=(a:Rat,b:Rat)=>{const d=a.n*b.d-b.n*a.d;return d<Z?-1:d>Z?1:0;};
function expose(a:Rat){return {numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const numeric=(a:Rat)=>expose(a).value;
function controls(p:number[],ranges:readonly(readonly[number,number])[]){
  if(p.length!==ranges.length)throw new Error('Use the stated initial-state controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
type State={position:number;velocity:number;kinetic:number;potential:number;energy:number;energyResidual:number;scaledEnergyResidual:number};
type M2=[[number,number],[number,number]];
const mv=(A:M2,v:Vec2):Vec2=>[A[0][0]*v[0]+A[0][1]*v[1],A[1][0]*v[0]+A[1][1]*v[1]];
const norm=(x:Vec2)=>Math.hypot(...x),minus=(a:Vec2,b:Vec2):Vec2=>[a[0]-b[0],a[1]-b[1]];

/** Newton m*x''=k*x. k>=0 is destabilizing stiffness, not a restoring spring.
 * [m,k,10xA,10vA,10xB,10vB,10t]. State norm uses (x,T*v),T=1s.
 * Exact initial energies; floating point exp/sqrt, flow, states and diagnostics.
 */
export function invertedSpringInitialState(p:number[]){
  controls(p,[[1,4],[0,4],[-20,20],[-20,20],[-20,20],[-20,20],[0,50]]);
  const [mass,stiffness,xa,va,xb,vb,tc]=p,time=tc/10,lambda=Math.sqrt(stiffness/mass),referenceTime=1;
  const initialA:Vec2=[xa/10,va/10],initialB:Vec2=[xb/10,vb/10];
  const initialEnergy=(x:number,v:number)=>expose(sub(mul(R(mass,2),sq(R(v,10))),mul(R(stiffness,2),sq(R(x,10)))));
  const energyA=initialEnergy(xa,va),energyB=initialEnergy(xb,vb);
  const flow=(t:number):M2=>lambda===0?[[1,t/referenceTime],[0,1]]:[[Math.cosh(lambda*t),Math.sinh(lambda*t)/(lambda*referenceTime)],[referenceTime*lambda*Math.sinh(lambda*t),Math.cosh(lambda*t)]];
  const state=(z:Vec2,t:number,E:number):State=>{
    let position:number,velocity:number;
    if(lambda===0){position=z[0]+z[1]*t;velocity=z[1];}
    else {const growing=(z[0]+z[1]/lambda)/2,decaying=(z[0]-z[1]/lambda)/2,up=growing*Math.exp(lambda*t),down=decaying*Math.exp(-lambda*t);position=up+down;velocity=lambda*(up-down);}
    const kinetic=mass*velocity*velocity/2,potential=-stiffness*position*position/2,energy=kinetic+potential,residual=energy-E;
    return {position,velocity,kinetic,potential,energy,energyResidual:residual,scaledEnergyResidual:residual/Math.max(1,kinetic,Math.abs(potential),Math.abs(E))};
  };
  const qA=state(initialA,time,energyA.value),qB=state(initialB,time,energyB.value),matrix=flow(time),inverse=flow(-time);
  // det M=1 analytically. Singular values are evaluated using that identity;
  // the independently computed floating point determinant remains diagnostic.
  const trace=matrix.flat().reduce((s,x)=>s+x*x,0),sigmaMax=Math.sqrt((trace+Math.sqrt(Math.max(0,(trace-2)*(trace+2))))/2),sigmaMin=1/sigmaMax;
  const initialDifference=minus(initialB,initialA),initialDistance=norm(initialDifference),queryDifference=minus([qB.position,qB.velocity],[qA.position,qA.velocity]),queryDistance=norm(queryDifference),predictedDifference=mv(matrix,initialDifference);
  const observed:Vec2=[qA.position,qA.velocity],roundedObservationA=observed.map(x=>Math.round(x*1000)/1000)as Vec2,recoveredInitialA=mv(inverse,observed),roundedRecoveredInitialA=mv(inverse,roundedObservationA);
  const observationErrorNorm=norm(minus(roundedObservationA,observed)),reconstructionErrorNorm=norm(minus(recoveredInitialA,initialA)),roundedReconstructionErrorNorm=norm(minus(roundedRecoveredInitialA,initialA));
  const sampleTimes=Array.from({length:201},(_,j)=>j/40);if(!sampleTimes.includes(time))sampleTimes.push(time);sampleTimes.sort((a,b)=>a-b);
  // xA,xB,vA,vB,Euclidean state distance,operator upper bound,E_A,E_B.
  const curves:Vec2[][]=Array.from({length:8},()=>[]),phaseCurves:Vec2[][]=[[],[]];
  for(const t of sampleTimes){const a=state(initialA,t,energyA.value),b=state(initialB,t,energyB.value),M=flow(t),tr=M.flat().reduce((s,x)=>s+x*x,0),s=Math.sqrt((tr+Math.sqrt(Math.max(0,(tr-2)*(tr+2))))/2),distance=norm([b.position-a.position,b.velocity-a.velocity]);[a.position,b.position,a.velocity,b.velocity,distance,s*initialDistance,a.energy,b.energy].forEach((value,j)=>curves[j].push([t,value]));phaseCurves[0].push([a.position,referenceTime*a.velocity]);phaseCurves[1].push([b.position,referenceTime*b.velocity]);}
  return {mass,stiffness,lambda,referenceTime,time,initialA,initialB,initialEnergyA:energyA,initialEnergyB:energyB,queryA:qA,queryB:qB,matrix,inverse,sigmaMax,sigmaMin,numericalDeterminant:matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0],analyticDeterminant:1,initialDifference,queryDifference,predictedDifference,initialDistance,queryDistance,amplification:initialDistance===0?null:queryDistance/initialDistance,operatorBound:sigmaMax*initialDistance,flowDifferenceResidual:norm(minus(queryDifference,predictedDifference)),roundedObservationA,recoveredInitialA,roundedRecoveredInitialA,observationErrorNorm,reconstructionErrorNorm,roundedReconstructionErrorNorm,roundedRecoveryBound:sigmaMax*observationErrorNorm,roundingBoxBound:sigmaMax*Math.SQRT2*.0005,sameCompleteInitialState:xa===xb&&va===vb,sameInitialPosition:xa===xb,sameInitialEnergy:energyA.label===energyB.label,locallyLipschitz:true as const,freeLimit:stiffness===0,modeA:lambda===0?null:{growing:(initialA[0]+initialA[1]/lambda)/2,decaying:(initialA[0]-initialA[1]/lambda)/2},modeB:lambda===0?null:{growing:(initialB[0]+initialB[1]/lambda)/2,decaying:(initialB[0]-initialB[1]/lambda)/2},curves,phaseCurves};
}

/** Explicit nonunique C² solution family, not an ODE selection algorithm.
 * F(x)=12*m*c*sqrt(max(x,0)); c units sqrt(m)/s².
 * [mass,2c,10waitA,10waitB,10t]. All share x(0)=v(0)=0.
 * A genuine stationary solution is retained separately for every finite horizon.
 */
export function delayedNewtonRelease(p:number[]){
  controls(p,[[1,4],[0,4],[0,50],[0,50],[0,100]]);
  const [mass,cc,ac,bc,tc]=p,M=R(mass),c=R(cc,2),waitA=R(ac,10),waitB=R(bc,10),time=R(tc,10);
  const state=(t:Rat,wait:Rat|null)=>{
    const elapsed=wait&&compare(t,wait)>0?sub(t,wait):R(0),c2=sq(c),x=mul(c2,pow(elapsed,4)),v=mul(R(4),mul(c2,pow(elapsed,3))),a=mul(R(12),mul(c2,sq(elapsed))),force=mul(M,a),K=mul(R(8),mul(M,mul(pow(c,4),pow(elapsed,6))));
    // On these nonnegative solutions sqrt(x)=c*s² exactly. General F(x) is
    // not claimed rational for arbitrary probe positions.
    const rootX=mul(c,sq(elapsed)),forceFromLaw=mul(R(12),mul(M,mul(c,rootX))),U=neg(mul(R(8),mul(M,mul(c,mul(x,rootX)))));
    return {position:expose(x),velocity:expose(v),acceleration:expose(a),force:expose(force),kinetic:expose(K),potential:expose(U),energy:expose(add(K,U)),newtonResidual:expose(sub(mul(M,a),forceFromLaw)),energyResidual:expose(add(K,U)),elapsedAfterWait:expose(elapsed),nonzeroMotion:x.n!==Z};
  };
  const queryA=state(time,waitA),queryB=state(time,waitB),stationary=state(time,null);
  const sampleTimes=new Map<string,Rat>();for(let j=0;j<=200;j++){const t=R(j,20);sampleTimes.set(expose(t).label,t);}for(const t of [waitA,waitB,time])sampleTimes.set(expose(t).label,t);
  // x_A,x_B,x_stationary; v_A,v_B,v_stationary; a_A,a_B,a_stationary;
  // F_A,F_B,F_stationary; K_A,U_A,E_A; K_B,U_B,E_B.
  const curves:Vec2[][]=Array.from({length:18},()=>[]),phaseCurves:Vec2[][]=[[],[],[]];
  for(const t of [...sampleTimes.values()].sort(compare)){const states=[state(t,waitA),state(t,waitB),state(t,null)],n=numeric(t);for(const[j,key]of (['position','velocity','acceleration','force']as const).entries())states.forEach((s,i)=>curves[3*j+i].push([n,s[key].value]));for(let i=0;i<2;i++)for(const[j,key]of (['kinetic','potential','energy']as const).entries())curves[12+3*i+j].push([n,states[i][key].value]);states.forEach((s,i)=>phaseCurves[i].push([s.position.value,s.velocity.value]));}
  const originSlopeProbes=[1,2,4,8,16].map(j=>({position:expose(R(1,j*j)),force:expose(mul(R(12,j),mul(M,c))),forcePerPosition:expose(mul(R(12*j),mul(M,c)))}));
  return {mass,coefficient:expose(c),waitA:expose(waitA),waitB:expose(waitB),time:expose(time),queryA,queryB,stationary,sameCompleteInitialState:true as const,nonuniqueConstructed:cc>0,locallyLipschitzAtOrigin:cc===0,finiteWaitFamiliesDistinct:cc>0&&ac!==bc,distinctAtQuery:(queryA.position.label!==queryB.position.label||queryA.velocity.label!==queryB.velocity.label),stationaryIsInfiniteWait:true as const,originSlopeProbes,curves,phaseCurves};
}
