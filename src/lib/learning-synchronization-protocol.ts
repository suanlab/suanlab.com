/** Browser-safe transport validation. Numerical models are type-only imports. */
import {isHopfAmplitudeParameters,hopfAmplitudeKeys,type HopfAmplitudeParameters} from './learning-hopf-amplitude-input';
import {isPairPhaseParameters,pairPhaseParameterKeys,type PairPhaseParameters} from './learning-pair-phase-input';
import {isFiniteKuramotoParameters,finiteKuramotoKeys,type FiniteKuramotoParameters} from './learning-finite-kuramoto-input';
import {isMasterObservation} from './learning-master-equation-protocol';
import {parseNumberJsonDraft} from './learning-json-number-input';
import type {hopfAmplitudeModel} from './learning-hopf-amplitude';
import type {pairPhaseLockingModel} from './learning-pair-phase-locking';
import type {finiteKuramotoModel} from './learning-finite-kuramoto';
export const synchronizationKinds=['hopf-amplitude-critical-relaxation','two-oscillator-locking-phase-slips','finite-kuramoto-collective-order'] as const;
export type SynchronizationInput=
 |{kind:typeof synchronizationKinds[0];parameters:HopfAmplitudeParameters}
 |{kind:typeof synchronizationKinds[1];parameters:PairPhaseParameters}
 |{kind:typeof synchronizationKinds[2];parameters:FiniteKuramotoParameters};
export type SynchronizationSuccess={version:1;requestId:number;ok:true;inputKey:string}&(
 |{kind:typeof synchronizationKinds[0];input:Extract<SynchronizationInput,{kind:typeof synchronizationKinds[0]}>;result:ReturnType<typeof hopfAmplitudeModel>}
 |{kind:typeof synchronizationKinds[1];input:Extract<SynchronizationInput,{kind:typeof synchronizationKinds[1]}>;result:ReturnType<typeof pairPhaseLockingModel>}
 |{kind:typeof synchronizationKinds[2];input:Extract<SynchronizationInput,{kind:typeof synchronizationKinds[2]}>;result:ReturnType<typeof finiteKuramotoModel>});
export type SynchronizationResponse=SynchronizationSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
type Check=(x:unknown)=>boolean;
export const synchronizationRecord=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype;
const shape=(x:unknown,spec:Record<string,Check>):x is Record<string,unknown>=>synchronizationRecord(x)&&Reflect.ownKeys(x).length===Object.keys(spec).length&&Object.entries(spec).every(([k,check])=>Object.hasOwn(x,k)&&check(x[k]));
const literal=(v:unknown):Check=>x=>x===v,choice=(...v:unknown[]):Check=>x=>v.includes(x),bool:Check=x=>typeof x==='boolean',finite:Check=x=>typeof x==='number'&&Number.isFinite(x);
const integer=(x:unknown,min=0,max=Number.MAX_SAFE_INTEGER):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
const array=(n:number,check:Check):Check=>x=>Array.isArray(x)&&Object.getPrototypeOf(x)===Array.prototype&&x.length===n&&Reflect.ownKeys(x).length===n+1&&Array.from({length:n},(_,i)=>Object.hasOwn(x,i)&&check(x[i])).every(Boolean);
const fields=(keys:string,check:Check)=>Object.fromEntries(keys.split(' ').map(k=>[k,check]));
const nullable=(check:Check):Check=>x=>x===null||check(x),text:Check=x=>typeof x==='string'&&x.length>0&&x.length<12000;
const same=(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b);
export function isSynchronizationInput(x:unknown):x is SynchronizationInput{
 return shape(x,{kind:choice(...synchronizationKinds),parameters:synchronizationRecord})&&(x.kind===synchronizationKinds[0]?isHopfAmplitudeParameters(x.parameters):x.kind===synchronizationKinds[1]?isPairPhaseParameters(x.parameters):isFiniteKuramotoParameters(x.parameters));
}
export function snapshotSynchronizationInput(x:SynchronizationInput):SynchronizationInput{
 if(!isSynchronizationInput(x))throw new RangeError('Invalid synchronization preparation');
 const keys=x.kind===synchronizationKinds[0]?hopfAmplitudeKeys:x.kind===synchronizationKinds[1]?pairPhaseParameterKeys:finiteKuramotoKeys;
 const p=x.parameters as unknown as Record<string,number|readonly number[]>;
 return Object.freeze({kind:x.kind,parameters:Object.freeze(Object.fromEntries(keys.map(k=>[k,Array.isArray(p[k])?Object.freeze((p[k] as number[]).map(v=>v+0)):(p[k] as number)+0])))}) as unknown as SynchronizationInput;
}
export const synchronizationInputKey=(x:SynchronizationInput)=>JSON.stringify(snapshotSynchronizationInput(x));
export const isSynchronizationRequestEnvelope=(x:unknown):x is{version:1;requestId:number;input:unknown}=>shape(x,{version:literal(1),requestId:x=>integer(x,1),input:()=>true});
export function parseSynchronizationInputDraft(raw:string):SynchronizationInput|null{
 try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['parameters']);return isSynchronizationInput(x)?snapshotSynchronizationInput(x):null;}catch{return null;}
}
export function parseSynchronizationStoredInput(raw:string):SynchronizationInput|null{
 try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['version','parameters']);return shape(x,{version:literal(1),value:isSynchronizationInput})?snapshotSynchronizationInput(x.value as SynchronizationInput):null;}catch{return null;}
}
/** Validates the complete returned structure and its input attribution without
 * executing the numerical model on the main thread. Independent model tests
 * establish numerical accuracy; this validator checks the transport contract. */
export function isSynchronizationSuccessForInput(value:unknown,input:SynchronizationInput,id:number):value is SynchronizationSuccess{
 if(!integer(id,1)||!isSynchronizationInput(input)||!shape(value,{version:literal(1),requestId:literal(id),ok:literal(true),kind:literal(input.kind),input:isSynchronizationInput,inputKey:literal(synchronizationInputKey(input)),result:synchronizationRecord})||synchronizationInputKey(value.input as SynchronizationInput)!==synchronizationInputKey(input))return false;
 const r=value.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
 const q:Check=x=>{if(!synchronizationRecord(x))return false;if(cache.has(x))return cache.get(x)!;const valid=isMasterObservation(x);cache.set(x,valid);return valid;};
 const exact:Check=x=>isMasterObservation(x,true),positiveInteger:Check=x=>integer(x,1),count:Check=x=>integer(x),signedInteger:Check=x=>integer(x,-Number.MAX_SAFE_INTEGER);
 const p=input.parameters;
 const common={parameters:(x:unknown)=>(input.kind===synchronizationKinds[0]?isHopfAmplitudeParameters(x):input.kind===synchronizationKinds[1]?isPairPhaseParameters(x):isFiniteKuramotoParameters(x))&&same(x,snapshotSynchronizationInput(input).parameters),approximationContract:text};
 let row:Check;
 if(input.kind===synchronizationKinds[0]){
  row=x=>shape(x,{step:count,time:exact,...fields('squaredAmplitude amplitude integratedSquaredAmplitude x y squaredAmplitudeDerivative sensitivityToInitialSquaredAmplitude coordinateRadiusResidual',q),...fields('phaseShift instantaneousAngularFrequency logAmplitudeBalanceResidual',nullable(q)),algebraicLogAmplitudeBalanceResidual:nullable(exact),phaseDefined:bool});
  if(!shape(r,{...common,precisionBits:literal(256),initialSquaredAmplitude:exact,regime:choice('stable-origin','critical-algebraic-attraction','unstable-origin-and-attracting-circle'),classification:x=>shape(x,{positiveSaturation:literal(true),...fields('originInitial stationaryAmplitude criticalParameterExactlyZero ordinaryHopfCrossingAtMuZero positiveMuCircleExists positiveMuCircleIsPeriodicOrbit positiveMuCircleIsEquilibria phaseIsDefined',bool),sampledNoise:literal(false)}),...fields('originLinearAmplitudeGrowthRate originLinearSquaredAmplitudeGrowthRate',exact),...fields('attractingCircleSquaredAmplitude attractingCircleAngularFrequency attractingCircleRadialDecayRate originLinearAmplitudeRelaxationTime attractingCircleRadialRelaxationTime criticalSquaredAmplitudeHalfTime criticalAmplitudeHalfTime periodDividedByTwoPi',nullable(exact)),history:array(p.sampleSteps+1,row),current:row}))return false;
  const c=r.classification as Record<string,unknown>,hp=input.parameters,origin=hp.xNumerator===0&&hp.yNumerator===0,positive=hp.muNumerator>0,critical=hp.muNumerator===0;
  if(c.originInitial!==origin||c.phaseIsDefined===origin||c.criticalParameterExactlyZero!==critical||c.positiveMuCircleExists!==positive||c.ordinaryHopfCrossingAtMuZero!==(hp.omegaNumerator!==0))return false;
  const hn=(n:number)=>BigInt(n),rhoN=hn(hp.xNumerator)**BigInt(2)*hn(hp.yDenominator)**BigInt(2)+hn(hp.yNumerator)**BigInt(2)*hn(hp.xDenominator)**BigInt(2),rhoD=hn(hp.xDenominator)**BigInt(2)*hn(hp.yDenominator)**BigInt(2);
  const stationary=origin||hn(hp.gNumerator)*rhoN*hn(hp.muDenominator)===hn(hp.muNumerator)*hn(hp.gDenominator)*rhoD;
  const zeroCircleFrequency=hn(hp.omegaNumerator)*hn(hp.shearDenominator)*hn(hp.muDenominator)*hn(hp.gNumerator)===hn(hp.shearNumerator)*hn(hp.muNumerator)*hn(hp.omegaDenominator)*hn(hp.gDenominator);
  if(c.stationaryAmplitude!==stationary||c.positiveMuCircleIsEquilibria!==(positive&&zeroCircleFrequency)||c.positiveMuCircleIsPeriodicOrbit!==(positive&&!zeroCircleFrequency))return false;
  if((r.history as Record<string,unknown>[]).some(h=>h.phaseDefined===origin||(h.phaseShift===null)!==origin||(h.instantaneousAngularFrequency===null)!==origin))return false;
  const regime=positive?'unstable-origin-and-attracting-circle':critical?'critical-algebraic-attraction':'stable-origin';if(r.regime!==regime)return false;
 }else if(input.kind===synchronizationKinds[1]){
  const turns:Check=x=>shape(x,{status:choice('exact-integer-count','unresolved-count'),signedCount:nullable(signedInteger),lower:signedInteger,upper:signedInteger,reason:text})&&(x.lower as number)<=(x.upper as number)&&(x.status==='exact-integer-count'?x.lower===x.upper&&x.signedCount===x.lower:x.lower!==x.upper&&x.signedCount===null);
  row=x=>shape(x,{step:count,time:exact,...fields('relativeCos relativeSin relativePhase phaseChange relativeAngularVelocity phase1 phase2 cos1 sin1 cos2 sin2 angularFrequency1 angularFrequency2 firstHarmonicSquared firstHarmonic secondHarmonicSquared relativeCircleResidual',q),...fields('wrappedRelativePhase finiteWindowRelativeFrequency',nullable(q)),phaseBranch:turns,completedRelativeTurns:turns,meanPhase:exact,meanFrequencyConservationResidual:exact});
  const fixedPhase:Check=x=>shape(x,{kind:choice('stable','unstable','critical-semistable'),...fields('phase cos sin linearGrowthRate',q)});
  if(!shape(r,{...common,precisionBits:literal(256),guardBits:literal(32),regime:choice('neutral-all-relative-phases','critical-one-sided-bottleneck','locking-fixed-phases','drifting-relative-phase'),...fields('detuning totalCoupling edgeCoupling discriminant meanAngularFrequency initialRelativeCos initialRelativeSin initialRelativeAngularVelocity',exact),...fields('initialRelativePhase asymptoticRelativeFrequency flowDeterminantResidual',q),...fields('localLockingDecayRate localLockingRelaxationTime driftPeriod',nullable(q)),fixedPhases:x=>Array.isArray(x)&&x.length<=2&&array(x.length,fixedPhase)(x),classification:x=>shape(x,{...fields('initialRelativePhaseFixed neutralFamily criticalExactly asymptoticallyEqualFrequencies stableLockingExists zeroCoupling',bool),initialFixedType:choice('stable','unstable','critical-semistable','neutral',null),phaseDirection:choice(-1,0,1),sampledNoise:literal(false)}),liftPartition:x=>shape(x,{subintervals:count,maxRelativeIncrementBound:exact,method:literal('exact-flow-local-angle-sum-independent-of-reader-record-count')}),flowMatrix:array(2,array(2,q)),history:array(p.sampleSteps+1,row),current:row}))return false;

  const pp=input.parameters,bn=(x:number)=>BigInt(x),dn=bn(pp.omega1Numerator)*bn(pp.omega2Denominator)-bn(pp.omega2Numerator)*bn(pp.omega1Denominator),dd=bn(pp.omega1Denominator)*bn(pp.omega2Denominator),kn=bn(pp.couplingNumerator),kd=bn(pp.couplingDenominator),disc=kn*kn*dd*dd-dn*dn*kd*kd,a=bn(pp.initialHalfA),b=bn(pp.initialHalfB),speed=dn*(a*a+b*b)*kd-kn*BigInt(2)*a*b*dd,neutral=kn===BigInt(0)&&dn===BigInt(0),critical=disc===BigInt(0)&&!neutral,fixed=speed===BigInt(0);
  const regime=neutral?'neutral-all-relative-phases':critical?'critical-one-sided-bottleneck':disc>BigInt(0)?'locking-fixed-phases':'drifting-relative-phase';
  const expected={initialRelativePhaseFixed:fixed,initialFixedType:!fixed?null:neutral?'neutral':critical?'critical-semistable':b*b-a*a>BigInt(0)?'stable':'unstable',neutralFamily:neutral,criticalExactly:critical,phaseDirection:speed>BigInt(0)?1:speed<BigInt(0)?-1:0,asymptoticallyEqualFrequencies:disc>=BigInt(0),stableLockingExists:disc>BigInt(0),sampledNoise:false,zeroCoupling:kn===BigInt(0)};
  if(!Object.entries(expected).every(([k,v])=>(r.classification as Record<string,unknown>)[k]===v)||r.regime!==regime||(r.fixedPhases as unknown[]).length!==(disc>BigInt(0)?2:critical?1:0)||(r.driftPeriod===null)!==(disc>=BigInt(0))||(r.localLockingDecayRate===null)!==(disc<=BigInt(0))||(r.localLockingRelaxationTime===null)!==(disc<=BigInt(0)))return false;
 }else{
  const fp=input.parameters,N=fp.omegaNumerators.length;
  const vec=array(N,finite),nonnegative:Check=x=>finite(x)&&(x as number)>=0;
  const order:Check=x=>shape(x,{...fields('real imaginary',finite),...fields('magnitude squaredMagnitude rawNumericalMagnitude rawNumericalSquaredMagnitude',nonnegative),meanDirection:nullable(finite),status:choice('analytic-zero','analytic-unit','binary64-observation'),rawMagnitudeOutsideUnitInterval:bool})&&(x.status!=='analytic-zero'||x.magnitude===0&&x.squaredMagnitude===0&&x.real===0&&x.imaginary===0&&x.meanDirection===null)&&(x.status!=='analytic-unit'||x.magnitude===1&&x.squaredMagnitude===1)&&(x.magnitude!==0||x.meanDirection===null);
  const harmonic:Check=x=>synchronizationRecord(x)&&integer(x.harmonic,1,N)&&shape(x,{harmonic:positiveInteger,...fields('real imaginary',finite),...fields('magnitude squaredMagnitude rawNumericalMagnitude rawNumericalSquaredMagnitude',nonnegative),meanDirection:nullable(finite),status:choice('analytic-zero','analytic-unit','binary64-observation'),rawMagnitudeOutsideUnitInterval:bool});
  row=x=>shape(x,{step:count,gridIndex:x=>integer(x,0,fp.integrationSteps),time:nonnegative,timeExact:exact,...fields('phases phaseChanges velocities cosines sines rawRk4FinestPhases',vec),phaseChangeRepresentation:choice('analytic-linear-increment-rounded-to-binary64','binary64-difference'),finiteWindowFrequencies:nullable(vec),firstHarmonic:order,secondHarmonic:order,balancedGroupHarmonic:nullable(harmonic),...fields('meanUnwrappedPhase observedMeanFrequency meanFrequencyResidual meanPhaseBalanceResidual couplingPotential instantaneousPotentialDerivative',finite),...fields('instantaneousFrequencySpread rk4DepartureFromPrimary coarseToMediumPhaseDifference mediumToFinePhaseDifference',nonnegative),finiteWindowFrequencySpread:nullable(nonnegative),negativeSquaredCouplingRate:nullable(finite),potentialDescentIdentityApplies:bool,orderIdentityReason:choice(null,'rotating-balanced-roots-of-unity','initial-balanced-roots-of-unity')});
  if(!shape(r,{...common,populationSize:literal(N),trajectoryMethod:choice('analytic-invariant-profile-rotation','analytic-uncoupled-rotation','binary64-rk4-finest-grid'),omega:x=>vec(x)&&same(x,fp.omegaNumerators.map(n=>n/fp.omegaDenominator)),initialPhases:x=>vec(x)&&same(x,fp.phasePiNumerators.map(n=>Math.PI*(n/fp.phasePiDenominator))),naturalFrequencySpread:nonnegative,meanAngularFrequencyIdentity:exact,edgeCoupling:exact,classification:x=>shape(x,{singleOscillator:literal(N===1),...fields('identicalNaturalFrequencies stationaryRelativeConfigurationIdentity frequencyEqualityIdentityApplies',bool),...fields('initialBalancedRootGroups invariantRootGroups',nullable(x=>integer(x,1,N))),...fields('stabilityInferredFromOrderOrWindow finitePopulationCriticalValueInferred sampledNoise',literal(false))}),numerics:x=>shape(x,{method:literal('classical-rk4-with-pairwise-antisymmetric-rhs'),baseSteps:literal(fp.integrationSteps),stepCounts:x=>same(x,fp.timeNumerator===0?[0,0,0]:[fp.integrationSteps,2*fp.integrationSteps,4*fp.integrationSteps])&&array(3,count)(x),...fields('finestTimeStep couplingTimesFinestStep maxCoarseToMediumPhaseDifference maxMediumToFinePhaseDifference formalRichardsonErrorEstimate',nonnegative),observedRefinementRatio:nullable(nonnegative),errorEstimateIsCertified:literal(false),differenceTrend:choice('differences-rounded-to-zero','finest-difference-rounded-to-zero','decreasing','increasing','equal-nonzero'),maxUnmodifiedRkDepartureFromAnalyticControl:nullable(nonnegative),displayRecordRule:text}),continuumReference:x=>shape(x,{model:literal('infinite-population-lorentzian-ott-antonsen-manifold'),...fields('width criticalCoupling linearOrderGrowthRateAtZero cubicSaturation',exact),regime:choice('above-onset','below-onset','critical'),zeroOrderBranchExists:literal(true),zeroBranchStability:choice('unstable','attracting','nonhyperbolic-algebraic'),...fields('nonzeroOrderSquared radialDecayRateAtNonzeroOrder radialRelaxationTimeAtNonzeroOrder orderRelaxationTimeBelow',nullable(exact)),nonzeroOrder:nullable(q),fittedFromFiniteFrequencies:literal(false),finitePopulationCriticalValueClaim:literal(false),conditions:text}),history:array(p.sampleSteps+1,row),current:row}))return false;

  const identical=fp.omegaNumerators.every(n=>n===fp.omegaNumerators[0]),c=r.classification as Record<string,unknown>,ref=r.continuumReference as Record<string,unknown>;
  const kn=BigInt(fp.couplingNumerator),kd=BigInt(fp.couplingDenominator),gn=BigInt(fp.referenceWidthNumerator),gd=BigInt(fp.referenceWidthDenominator),mu=kn*gd-BigInt(2)*gn*kd,above=mu>BigInt(0),below=mu<BigInt(0);
  if(c.identicalNaturalFrequencies!==identical||ref.regime!==(above?'above-onset':below?'below-onset':'critical')||ref.zeroBranchStability!==(above?'unstable':below?'attracting':'nonhyperbolic-algebraic'))return false;
  for(const field of ['nonzeroOrderSquared','nonzeroOrder','radialDecayRateAtNonzeroOrder','radialRelaxationTimeAtNonzeroOrder'])if((ref[field]!==null)!==above)return false;
  if((ref.orderRelaxationTimeBelow!==null)!==below)return false;
  for(const h of r.history as Record<string,unknown>[])if(h.potentialDescentIdentityApplies!==identical||(h.negativeSquaredCouplingRate!==null)!==identical)return false;
 }
 const history=r.history as Record<string,unknown>[];
 if(!same(r.current,history.at(-1)))return false;
 for(let i=0;i<history.length;i++){
  const h=history[i];if(h.step!==i)return false;
  if(input.kind===synchronizationKinds[2]){
   const pp=input.parameters,index=Math.floor(i*pp.integrationSteps/pp.sampleSteps),time=(pp.timeNumerator/pp.timeDenominator)*index/pp.integrationSteps;
   if(h.gridIndex!==index||h.time!==time||((h.timeExact as {value:unknown}).value!==time&&Math.abs(((h.timeExact as {value:number}).value)-time)>1e-14))return false;
   if((time===0)!==(h.finiteWindowFrequencies===null)||(time===0)!==(h.finiteWindowFrequencySpread===null))return false;
  }else if((h.time as {value:unknown}).value!==p.timeNumerator*i/(p.timeDenominator*p.sampleSteps)&&Math.abs((h.time as {value:number}).value-p.timeNumerator*i/(p.timeDenominator*p.sampleSteps))>1e-14)return false;
 }
 return true;
}
