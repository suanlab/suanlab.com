/** Two phase oscillators with total K split as K/2 per directed edge.
 * Exact flow, lifted phase and explicit wrapping/turn-count conventions.
 * Unregistered numerical foundation; no Worker/UI completion claim. */
import {exact} from './learning-entropy-production-math';
import {f,zero,one,two,add,sub,mul,div,neg,floor,ceil,point,le,plus,minus,scale,product,square,quotient,sqrt,atan,piRange,rotation,pairFlowMatrix,matrixVector,observation,type R,type F} from './learning-pair-phase-math';
import { isPairPhaseParameters, pairPhaseParameterKeys, type PairPhaseParameters } from './learning-pair-phase-input';
export { isPairPhaseParameters, pairPhaseParameterKeys } from './learning-pair-phase-input';
export type { PairPhaseParameters } from './learning-pair-phase-input';
const sign=(x:F)=>x.n>BigInt(0)?1:x.n<BigInt(0)?-1:0;
const twoPi=scale(piRange,two),abs=(x:F)=>x.n<BigInt(0)?neg(x):x;
function signedCount(lower:bigint,upper:bigint,reason:string){if(lower>upper)throw new RangeError('Inverted integer count enclosure');return{status:lower===upper?'exact-integer-count' as const:'unresolved-count' as const,signedCount:lower===upper?Number(lower):null,lower:Number(lower),upper:Number(upper),reason};}
function coherence(cos:R){const r=scale(plus(point(one),cos),f(BigInt(1),BigInt(2)));return{lower:r.lower.n<BigInt(0)?zero:r.lower,upper:le(one,r.upper)?one:r.upper};}
export function pairPhaseLockingModel(input:PairPhaseParameters){
 if(!isPairPhaseParameters(input))throw new RangeError('Thirteen owned integers; denominators positive,|omega1/2|<=8,K0..16,|meanPhase0|<=8,time0..4s,records1..64, nonzero projective half-angle pair; magnitudes<=1e9');
 const parameters=Object.freeze(Object.fromEntries(pairPhaseParameterKeys.map(k=>[k,input[k]+0]))) as unknown as Readonly<PairPhaseParameters>,p=parameters;
 const q=(s:string)=>f(BigInt(p[(s+'Numerator') as keyof PairPhaseParameters]),BigInt(p[(s+'Denominator') as keyof PairPhaseParameters]));
 const w1=q('omega1'),w2=q('omega2'),K=q('coupling'),delta=sub(w1,w2),mean=div(add(w1,w2),two),phi0=q('meanPhase'),T=q('time'),disc=sub(mul(K,K),mul(delta,delta));
 let ai=BigInt(p.initialHalfA),bi=BigInt(p.initialHalfB);if(bi<BigInt(0)||(bi===BigInt(0)&&ai<BigInt(0))){ai=-ai;bi=-bi;}
 const magnitude=(ai<0?-ai:ai)>(bi<0?-bi:bi)?(ai<0?-ai:ai):(bi<0?-bi:bi),a=f(ai,magnitude),b=f(bi,magnitude),norm=add(mul(a,a),mul(b,b));
 const cos0=div(sub(mul(b,b),mul(a,a)),norm),sin0=div(mul(two,mul(a,b)),norm),initialPhase=bi===BigInt(0)?{...piRange}:scale(atan(point(div(a,b))),two),initialSpeed=sub(delta,mul(K,sin0)),fixed=initialSpeed.n===BigInt(0),neutral=K.n===BigInt(0)&&delta.n===BigInt(0),critical=disc.n===BigInt(0)&&!neutral;
 const regime=neutral?'neutral-all-relative-phases':critical?'critical-one-sided-bottleneck':disc.n>BigInt(0)?'locking-fixed-phases':'drifting-relative-phase';
 const spectral=neutral||critical?point(zero):sqrt(point(abs(disc))),asymptotic=disc.n<BigInt(0)?scale(spectral,f(BigInt(sign(delta)))):point(zero),direction=sign(initialSpeed);
 const fixedPhases:{kind:string;phase:R;cos:R;sin:R;linearGrowthRate:R}[]=[];
 if(disc.n>BigInt(0)){
 const s=div(delta,K),c=scale(spectral,div(one,K)),stable=scale(atan(quotient(point(s),c)),one);
 const unstable=s.n<BigInt(0)?minus(scale(piRange,f(BigInt(-1))),stable):minus(piRange,stable);
 fixedPhases.push({kind:'stable',phase:stable,cos:c,sin:point(s),linearGrowthRate:scale(spectral,f(BigInt(-1)))},{kind:'unstable',phase:unstable,cos:scale(c,f(BigInt(-1))),sin:point(s),linearGrowthRate:spectral});
 }else if(critical)fixedPhases.push({kind:'critical-semistable',phase:scale(piRange,f(BigInt(sign(delta)),BigInt(2))),cos:point(zero),sin:point(f(BigInt(sign(delta)))),linearGrowthRate:point(zero)});
 const initialFixedType=!fixed?null:neutral?'neutral':critical?'critical-semistable':cos0.n>BigInt(0)?'stable':'unstable';
 const speedBound=add(K,abs(delta)),gridN=fixed||K.n===BigInt(0)||T.n===BigInt(0)?0:Math.max(1,Number(ceil(mul(two,mul(speedBound,T)).n,mul(two,mul(speedBound,T)).d)));
 const cache=new Map<string,{v:R[];M:ReturnType<typeof pairFlowMatrix>}>();
 const flow=(time:F)=>{const key=`${time.n}/${time.d}`,old=cache.get(key);if(old)return old;const M=pairFlowMatrix(K,delta,time),value={M,v:matrixVector(M,[b,a])};cache.set(key,value);return value;};
 const increment=(u:R[],v:R[])=>{const dot=plus(product(u[0],v[0]),product(u[1],v[1])),cross=minus(product(u[0],v[1]),product(u[1],v[0]));return scale(atan(quotient(cross,dot)),two);};
 const grid:R[]=[point(zero)];if(gridN){let prev=flow(zero).v;for(let k=1;k<=gridN;k++){const v=flow(mul(T,f(BigInt(k),BigInt(gridN)))).v;grid.push(plus(grid[k-1],increment(prev,v)));prev=v;}}
 const history=Array.from({length:p.sampleSteps+1},(_,step)=>{
 const time=mul(T,f(BigInt(step),BigInt(p.sampleSteps))),zeroTime=time.n===BigInt(0);
 let change:R,cos:R,sin:R,pole=false,aligned=false,halfVector:R[]=[point(b),point(a)];
 if(fixed||zeroTime){change=point(zero);cos=point(cos0);sin=point(sin0);pole=cos0.n===-cos0.d&&sin0.n===BigInt(0);}
 else if(K.n===BigInt(0)){change=point(mul(delta,time));const half=rotation(scale(change,f(BigInt(1),BigInt(2))));halfVector=[minus(scale(half.cos,b),scale(half.sin,a)),plus(scale(half.sin,b),scale(half.cos,a))];const turn=rotation(change);cos=minus(scale(turn.cos,cos0),scale(turn.sin,sin0));sin=plus(scale(turn.sin,cos0),scale(turn.cos,sin0));}
 else{
 const i=Math.floor(step*gridN/p.sampleSteps),left=flow(mul(T,f(BigInt(i),BigInt(gridN)))).v,current=flow(time).v;halfVector=current;change=step*gridN%p.sampleSteps===0?grid[i]:plus(grid[i],increment(left,current));
 const denominator=plus(square(current[0]),square(current[1]));cos=quotient(minus(square(current[0]),square(current[1])),denominator);sin=quotient(scale(product(current[0],current[1]),two),denominator);
 if(critical){const exactB=add(b,mul(div(sub(mul(K,b),mul(delta,a)),two),time));pole=exactB.n===BigInt(0);const exactA=add(a,mul(div(sub(mul(delta,b),mul(K,a)),two),time));aligned=exactA.n===BigInt(0);if(pole){cos=point(f(BigInt(-1)));sin=point(zero);}}
 }
 let phase=plus(initialPhase,change),branch:ReturnType<typeof signedCount>;
 if(fixed||zeroTime)branch=signedCount(BigInt(0),BigInt(0),'initial-principal-phase-in-minus-pi-open-to-pi-closed');
 else if(aligned){const r=quotient(phase,twoPi),lo=ceil(r.lower.n,r.lower.d),hi=floor(r.upper.n,r.upper.d);branch=signedCount(lo,hi,'exact-critical-homogeneous-alignment');if(lo===hi){phase=scale(twoPi,f(lo));change=minus(phase,initialPhase);}}
 else if(pole){
 const r=quotient(phase,piRange),lo=ceil(sub(r.lower,one).n,sub(r.lower,one).d*BigInt(2)),hi=floor(sub(r.upper,one).n,sub(r.upper,one).d*BigInt(2));branch=signedCount(lo,hi,'exact-critical-homogeneous-pole');
 if(lo===hi){phase=scale(piRange,f(BigInt(2)*lo+BigInt(1)));change=minus(phase,initialPhase);}
 }else{const r=quotient(minus(phase,piRange),twoPi);branch=signedCount(ceil(r.lower.n,r.lower.d),ceil(r.upper.n,r.upper.d),'principal-phase-in-minus-pi-open-to-pi-closed');}
 const wrapped=branch.signedCount===null?null:aligned?point(zero):pole?{...piRange}:minus(phase,scale(twoPi,f(BigInt(branch.signedCount))));
 let turns:ReturnType<typeof signedCount>;
 if(disc.n>=BigInt(0)||zeroTime)turns=signedCount(BigInt(0),BigInt(0),zeroTime?'zero-observation-time':'no-complete-turn-before-an-autonomous-fixed-phase');
 else{const r=quotient(scale(spectral,time),twoPi),lo=floor(r.lower.n,r.lower.d),hi=floor(r.upper.n,r.upper.d);turns=sign(delta)>0?signedCount(lo,hi,'whole-drift-periods-since-initial-phase'):signedCount(-hi,-lo,'whole-drift-periods-since-initial-phase');}
 const velocity=fixed?point(zero):minus(point(delta),scale(sin,K)),phi=add(phi0,mul(mean,time)),theta1=plus(point(phi),scale(phase,f(BigInt(1),BigInt(2)))),theta2=minus(point(phi),scale(phase,f(BigInt(1),BigInt(2)))),meanRotation=rotation(point(phi)),halfNorm=sqrt(plus(square(halfVector[0]),square(halfVector[1]))),hc=quotient(halfVector[0],halfNorm),hs=quotient(halfVector[1],halfNorm),r1={cos:minus(product(meanRotation.cos,hc),product(meanRotation.sin,hs)),sin:plus(product(meanRotation.sin,hc),product(meanRotation.cos,hs))},r2={cos:plus(product(meanRotation.cos,hc),product(meanRotation.sin,hs)),sin:minus(product(meanRotation.sin,hc),product(meanRotation.cos,hs))},coh=coherence(cos);
 return{step,time:exact(time),relativeCos:observation(cos),relativeSin:observation(sin),relativePhase:observation(phase),phaseChange:observation(change),wrappedRelativePhase:wrapped?observation(wrapped):null,phaseBranch:branch,completedRelativeTurns:turns,relativeAngularVelocity:observation(velocity),finiteWindowRelativeFrequency:zeroTime?null:observation(scale(change,div(one,time))),meanPhase:exact(phi),phase1:observation(theta1),phase2:observation(theta2),cos1:observation(r1.cos),sin1:observation(r1.sin),cos2:observation(r2.cos),sin2:observation(r2.sin),angularFrequency1:observation(plus(point(mean),scale(velocity,f(BigInt(1),BigInt(2))))),angularFrequency2:observation(minus(point(mean),scale(velocity,f(BigInt(1),BigInt(2))))),firstHarmonicSquared:observation(coh),firstHarmonic:observation(sqrt(coh)),secondHarmonicSquared:observation(square(cos)),relativeCircleResidual:observation(minus(plus(square(cos),square(sin)),point(one))),meanFrequencyConservationResidual:exact(zero)};
 });
 const M=flow(T).M;
 return{parameters,precisionBits:256,guardBits:32,regime,detuning:exact(delta),totalCoupling:exact(K),edgeCoupling:exact(div(K,two)),discriminant:exact(disc),meanAngularFrequency:exact(mean),initialRelativeCos:exact(cos0),initialRelativeSin:exact(sin0),initialRelativePhase:observation(initialPhase),initialRelativeAngularVelocity:exact(initialSpeed),asymptoticRelativeFrequency:observation(asymptotic),localLockingDecayRate:disc.n>BigInt(0)?observation(spectral):null,localLockingRelaxationTime:disc.n>BigInt(0)?observation(quotient(point(one),spectral)):null,driftPeriod:disc.n<BigInt(0)?observation(quotient(twoPi,spectral)):null,fixedPhases:fixedPhases.map(v=>({kind:v.kind,phase:observation(v.phase),cos:observation(v.cos),sin:observation(v.sin),linearGrowthRate:observation(v.linearGrowthRate)})),classification:{initialRelativePhaseFixed:fixed,initialFixedType,neutralFamily:neutral,criticalExactly:critical,phaseDirection:direction,asymptoticallyEqualFrequencies:disc.n>=BigInt(0),stableLockingExists:disc.n>BigInt(0),sampledNoise:false,zeroCoupling:K.n===BigInt(0)},liftPartition:{subintervals:gridN,maxRelativeIncrementBound:exact(gridN?div(mul(speedBound,T),f(BigInt(gridN))):zero),method:'exact-flow-local-angle-sum-independent-of-reader-record-count'},flowMatrix:M.map(row=>row.map(observation)),flowDeterminantResidual:observation(minus(minus(product(M[0][0],M[1][1]),product(M[0][1],M[1][0])),point(one))),history,current:history.at(-1)!,
 approximationContract:'Deterministic two-oscillator phase model with K/2 on each directed edge: deltaDot=Delta-K sin(delta), K>=0. Exact rational regime/fixed-phase evidence; homogeneous half-angle real2x2 exponential via nilpotent identity or directed signed64-term Taylor scaling/squaring with analytic norm-tail bound. Lift sums local directed atan increments on a partition with |delta increment|<=1/2rad, independent of display records; this is not ODE time stepping. Initial delta lies in(-pi,pi]; initial half-angle (a,b) and its nonzero scalar multiples describe the same relative point. Phase-branch index uses the same half-open convention; exactly known critical poles retain that convention. Completed net2pi turns since the initial phase are a separate signed count, derived from whole drift periods; unresolved count bounds are not rounded. Individual mean phase is supplied independently. Frequency locking, phase alignment, stable locking and finite-window drift are distinct. No amplitude dynamics, noise, thermodynamic phase-transition, Worker, reader-registration or whole-topic completion claim.',
 };
}
