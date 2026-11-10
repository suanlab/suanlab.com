/** Reader-safe transport and preparation checks. Numerical imports are types only. */
import {isMarkovCoarseParameters,markovCoarseKeys,type MarkovCoarseParameters} from './learning-markov-coarse-input';
import {isLinearMemoryParameters,linearMemoryKeys,type LinearMemoryParameters} from './learning-linear-memory-input';
import {isMasterObservation} from './learning-master-equation-protocol';
import {binaryRatioObservation} from './learning-binary-products';
import {parseNumberJsonDraft} from './learning-json-number-input';
import type {markovCoarseModel} from './learning-markov-coarse';
import type {linearMemoryModel} from './learning-linear-memory';
import type {PartitionObservation as Q} from './learning-partition-exact-core';
export const coarseKinds=['markov-partition-lumpability-memory','linear-hidden-memory-markov-limit'] as const;
export type CoarseInput={kind:typeof coarseKinds[0];parameters:MarkovCoarseParameters}|{kind:typeof coarseKinds[1];parameters:LinearMemoryParameters};
export type CoarseSuccess={version:1;requestId:number;ok:true;inputKey:string}&(
 {kind:typeof coarseKinds[0];input:Extract<CoarseInput,{kind:typeof coarseKinds[0]}>;result:ReturnType<typeof markovCoarseModel>}
 |{kind:typeof coarseKinds[1];input:Extract<CoarseInput,{kind:typeof coarseKinds[1]}>;result:ReturnType<typeof linearMemoryModel>});
export type CoarseResponse=CoarseSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
type Check=(x:unknown)=>boolean;
export const coarseRecord=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype;
const shape=(x:unknown,spec:Record<string,Check>):x is Record<string,unknown>=>coarseRecord(x)&&Reflect.ownKeys(x).length===Object.keys(spec).length&&Object.entries(spec).every(([k,c])=>Object.hasOwn(x,k)&&c(x[k]));
const literal=(v:unknown):Check=>x=>x===v,choice=(...v:unknown[]):Check=>x=>v.includes(x),bool:Check=x=>typeof x==='boolean';
const integer=(x:unknown,min=0,max=Number.MAX_SAFE_INTEGER):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
const array=(n:number,c:Check):Check=>x=>Array.isArray(x)&&Object.getPrototypeOf(x)===Array.prototype&&x.length===n&&Reflect.ownKeys(x).length===n+1&&Array.from({length:n},(_,i)=>Object.hasOwn(x,i)&&c(x[i])).every(Boolean);
const fields=(names:string,c:Check)=>Object.fromEntries(names.split(' ').map(k=>[k,c])),text:Check=x=>typeof x==='string'&&x.length>0&&x.length<12000,same=(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b);
export function isCoarseInput(x:unknown):x is CoarseInput{return shape(x,{kind:choice(...coarseKinds),parameters:coarseRecord})&&(x.kind===coarseKinds[0]?isMarkovCoarseParameters(x.parameters):isLinearMemoryParameters(x.parameters));}
export function snapshotCoarseInput(value:CoarseInput):CoarseInput{
 if(!isCoarseInput(value))throw new RangeError('Invalid coarse-graining preparation');const keys=value.kind===coarseKinds[0]?markovCoarseKeys:linearMemoryKeys,p=value.parameters as unknown as Record<string,number|readonly number[]>;
 return Object.freeze({kind:value.kind,parameters:Object.freeze(Object.fromEntries(keys.map(k=>[k,Array.isArray(p[k])?Object.freeze((p[k] as number[]).map(v=>v+0)):(p[k] as number)+0])))}) as unknown as CoarseInput;
}
export const coarseInputKey=(x:CoarseInput)=>JSON.stringify(snapshotCoarseInput(x));
export const isCoarseRequestEnvelope=(x:unknown):x is{version:1;requestId:number;input:unknown}=>shape(x,{version:literal(1),requestId:x=>integer(x,1),input:()=>true});
export function parseCoarseInputDraft(raw:string):CoarseInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['parameters']);return isCoarseInput(x)?snapshotCoarseInput(x):null;}catch{return null;}}
export function parseCoarseStoredInput(raw:string):CoarseInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['version','parameters']);return shape(x,{version:literal(1),value:isCoarseInput})?snapshotCoarseInput(x.value as CoarseInput):null;}catch{return null;}}
// Cheap exact preparation algebra only: no powers, exponentials or trajectories.
type F={n:bigint;d:bigint};const f=(n:bigint,d=BigInt(1)):F=>({n,d}),zero=f(BigInt(0)),one=f(BigInt(1)),two=f(BigInt(2)),add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),eq=(a:F,b:F)=>a.n*b.d===b.n*a.d,le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,sgn=(a:F)=>a.n>BigInt(0)?1:a.n<BigInt(0)?-1:0;
const lower=(q:Q)=>f(BigInt(q.bounds!.lower.numerator.coefficient),BigInt(q.bounds!.lower.denominator.coefficient)),upper=(q:Q)=>f(BigInt(q.bounds!.upper.numerator.coefficient),BigInt(q.bounds!.upper.denominator.coefficient));
const matches=(q:unknown,v:F)=>{const x=q as Q;return eq(lower(x),v)&&eq(upper(x),v);};
// Directed fixed-point endpoints enclose non-dyadic initial data such as 2/3.
const encloses=(q:unknown,v:F)=>{const x=q as Q;return le(lower(x),v)&&le(v,upper(x));};
const sum=(v:F[])=>v.reduce(add,zero),matmul=(a:F[][],b:F[][])=>a.map(row=>b[0].map((_,j)=>sum(row.map((v,k)=>mul(v,b[k][j])))));
const containsZero=(q:unknown)=>{const x=q as Q;return lower(x).n<=BigInt(0)&&upper(x).n>=BigInt(0);};
/** A format/attribution validator, not a proof of an arbitrary sender's dynamics.
 * Independent model tests establish the mathematical comparisons. */
export function isCoarseSuccessForInput(value:unknown,input:CoarseInput,id:number):value is CoarseSuccess{
 if(!integer(id,1)||!isCoarseInput(input)||!shape(value,{version:literal(1),requestId:literal(id),ok:literal(true),kind:literal(input.kind),input:isCoarseInput,inputKey:literal(coarseInputKey(input)),result:coarseRecord})||coarseInputKey(value.input as CoarseInput)!==coarseInputKey(input))return false;
 const r=value.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
 const q:Check=x=>{
  if(!coarseRecord(x))return false;if(cache.has(x))return cache.get(x)!;let valid=isMasterObservation(x);
  if(valid){const v=x as unknown as Q,a=lower(v),b=upper(v);valid=le(a,b);if(valid&&v.resolution!=='unresolved-at-precision-ceiling'){const mid=div(add(a,b),two),projected=binaryRatioObservation({coefficient:mid.n,exponent:0},{coefficient:mid.d,exponent:0});valid=Object.is(v.value,projected.value)||v.value===projected.value;}}
  cache.set(x,valid);return valid;
 };
 const exact:Check=x=>q(x)&&isMasterObservation(x,true),structural:Check=x=>exact(x)&&(x as Q).resolution==='structural-zero',residual:Check=x=>q(x)&&containsZero(x),vec=(n:number,c:Check=q)=>array(n,c),matrix=(rows:number,cols:number,c:Check=q)=>array(rows,vec(cols,c));
 const common={parameters:(x:unknown)=>(input.kind===coarseKinds[0]?isMarkovCoarseParameters(x):isLinearMemoryParameters(x))&&same(x,snapshotCoarseInput(input).parameters),approximationContract:text};
 let records:Record<string,unknown>[];
 if(input.kind===coarseKinds[0]){
  const p=input.parameters,row:Check=x=>shape(x,{step:x=>integer(x,0,p.steps),...fields('micro hidden',vec(3,exact)),...fields('macro naiveMarkov initialAwareTruncated instantaneousTerm initialHiddenTerm fullMemoryTerm omittedMemoryOnExactHistory exactNextMacro reconstructedNextMacro witnessFromState0 witnessFromState1',vec(2,exact)),...fields('naiveErrorTV truncatedErrorHalfL1 witnessDistanceTV',exact),truncatedIsProbability:bool,reconstructionResidual:vec(2,structural),...fields('microMassResidual macroMassResidual',structural)});
  if(!shape(r,{...common,...fields('transition hiddenProjection hiddenPropagation',matrix(3,3,exact)),aggregation:matrix(2,3,exact),lifting:matrix(3,2,exact),instantaneousMacroTransition:matrix(2,2,exact),resolvedToHidden:matrix(3,2,exact),...fields('hiddenEigenvalue lumpabilityExitDefect',exact),classification:x=>shape(x,{...fields('strongLumpableAllInitialLaws initialHiddenExactlyZero allRetainedLagsForDisplayedHistory geometricHiddenModeStrictlyDecays hiddenModeAlternates',bool),...fields('kernelsAreProbabilities physicalNoiseSpecified continuousTimeModel',literal(false))}),memoryKernels:array(p.steps,x=>shape(x,{lag:x=>integer(x,0,31),matrix:matrix(2,2,exact)})),history:array(p.steps+1,row),current:row}))return false;
  const qn=(n:number)=>f(BigInt(n),BigInt(p.transitionDenominator)),a=qn(p.weight01),b=qn(p.weight02),c=qn(p.weight10),d=qn(p.weight12),e=qn(p.weight20),g=qn(p.weight21),alpha=f(BigInt(p.liftNumerator),BigInt(p.liftDenominator));
  const T=[[sub(sub(one,a),b),c,e],[a,sub(sub(one,c),d),g],[b,d,sub(sub(one,e),g)]],C=[[one,one,zero],[zero,zero,one]],R=[[alpha,zero],[sub(one,alpha),zero],[zero,one]],RC=matmul(R,C),H=RC.map((row,i)=>row.map((v,j)=>sub(i===j?one:zero,v))),A=matmul(matmul(C,T),R),E=matmul(H,T),D=matmul(E,R);
  for(const [name,m]of Object.entries({transition:T,aggregation:C,lifting:R,hiddenProjection:H,instantaneousMacroTransition:A,hiddenPropagation:E,resolvedToHidden:D}))if(!m.every((row,i)=>row.every((v,j)=>matches((r[name] as Q[][])[i][j],v))))return false;
  const lambda=sum(E.map((row,i)=>row[i])),initial=p.initialWeights.map(n=>f(BigInt(n))),total=sum(initial),initialLaw=initial.map(v=>div(v,total)),hidden0=sub(initialLaw[0],mul(alpha,add(initialLaw[0],initialLaw[1])));
  const expected={strongLumpableAllInitialLaws:eq(b,d),initialHiddenExactlyZero:hidden0.n===BigInt(0),allRetainedLagsForDisplayedHistory:p.memoryDepth>=Math.max(0,p.steps-1),geometricHiddenModeStrictlyDecays:(lambda.n<BigInt(0)?-lambda.n:lambda.n)<lambda.d,hiddenModeAlternates:lambda.n<BigInt(0),kernelsAreProbabilities:false,physicalNoiseSpecified:false,continuousTimeModel:false};
  if(!matches(r.hiddenEigenvalue,lambda)||!matches(r.lumpabilityExitDefect,sub(b,d))||!Object.entries(expected).every(([k,v])=>(r.classification as Record<string,unknown>)[k]===v))return false;
  if(!(r.memoryKernels as {lag:number}[]).every((k,i)=>k.lag===i))return false;
  records=r.history as Record<string,unknown>[];
  const probability=(v:Q[])=>v.every(q=>lower(q).n>=BigInt(0))&&eq(sum(v.map(lower)),one);
  for(const h of records){const micro=h.micro as Q[],macro=h.macro as Q[],naive=h.naiveMarkov as Q[],truncated=h.initialAwareTruncated as Q[];
   if(!probability(micro)||!probability(macro)||!probability(naive)||h.truncatedIsProbability!==probability(truncated)||!matches(macro[0],add(lower(micro[0]),lower(micro[1])))||!matches(macro[1],lower(micro[2])))return false;
   for(const key of ['naiveErrorTV','truncatedErrorHalfL1','witnessDistanceTV'])if(lower(h[key] as Q).n<BigInt(0))return false;
  }
  if(!(records[0].micro as Q[]).every((q,i)=>matches(q,initialLaw[i])))return false;
 }else{
  const p=input.parameters,qn=(s:string)=>f(BigInt(p[(s+'Numerator')as keyof LinearMemoryParameters]),BigInt(p[(s+'Denominator')as keyof LinearMemoryParameters])),a=qn('a'),b=qn('b'),c=qn('c'),d=qn('d'),eps=qn('epsilon'),x0=qn('x'),y0=qn('y'),T=qn('time'),bc=mul(b,c),detN=sub(mul(a,d),bc),fast=div(d,eps),tau=div(add(a,fast),two),diff=div(sub(fast,a),two),eta=add(mul(diff,diff),div(bc,eps)),gain=div(c,d),mismatch=sub(y0,mul(gain,x0)),forcing=mul(b,y0),stationary=sub(mul(b,y0),mul(a,x0)).n===BigInt(0)&&sub(mul(c,x0),mul(d,y0)).n===BigInt(0);
  const row:Check=x=>shape(x,{step:x=>integer(x,0,p.sampleSteps),time:exact,flowMethod:choice('exact-zero-time','coalescence-series','real-decaying-modes','damped-rotation'),...fields('x y markovX quasiSteadyHidden observedError hiddenError instantaneousTerm initialHiddenTerm homogeneousHidden memoryIntegral memoryKernelAtLag directObservedDerivative reconstructedObservedDerivative quasiSteadyMismatch markovLocalDefect',q),...fields('memoryEquationResidual markovDefectResidual',residual),initialHiddenAnalyticSign:literal(sgn(forcing)),initialHiddenLogAbsolute:forcing.n===BigInt(0)?literal(null):q,...fields('initialHiddenMagnitudeUnresolvedAtWorkingPrecision initialHiddenRoundedValueUnavailable',bool),convolutionOfObservedState:bc.n===BigInt(0)?literal(null):q,hiddenReconstructionResidual:bc.n===BigInt(0)?literal(null):residual,flowMatrix:matrix(2,2)});
  if(!shape(r,{...common,precisionBits:literal(512),driftMatrix:matrix(2,2,exact),...fields('trace determinant centerDecayRate centerDiscriminant hiddenRelaxationRate hiddenRelaxationTime quasiSteadyGain markovDriftRate initialQuasiSteadyMismatch',exact),markovRelaxationTime:detN.n===BigInt(0)?literal(null):exact,classification:x=>shape(x,{...fields('asymptoticallyStable marginalZeroMode oscillatoryModes repeatedEigenvalueExactly initialStationaryExactly initialOrigin memoryKernelExactlyZero observedStateAutonomousForAllInitialConditions initialQuasiSteadyExactly',bool),...fields('physicalNoiseSpecified physicalEnergyOrEntropySpecified',literal(false))}),history:array(p.sampleSteps+1,row),current:row}))return false;
  const staticValues={trace:neg(add(a,fast)),determinant:div(detN,eps),centerDecayRate:tau,centerDiscriminant:eta,hiddenRelaxationRate:fast,hiddenRelaxationTime:div(eps,d),quasiSteadyGain:gain,markovDriftRate:neg(div(detN,d)),initialQuasiSteadyMismatch:mismatch};
  if(!Object.entries(staticValues).every(([k,v])=>matches(r[k],v))||(detN.n!==BigInt(0)&&!matches(r.markovRelaxationTime,div(d,detN))))return false;
  const drift=[[neg(a),b],[div(c,eps),neg(fast)]];if(!drift.every((row,i)=>row.every((v,j)=>matches((r.driftMatrix as Q[][])[i][j],v))))return false;
  const expected={asymptoticallyStable:detN.n>BigInt(0),marginalZeroMode:detN.n===BigInt(0),oscillatoryModes:eta.n<BigInt(0),repeatedEigenvalueExactly:eta.n===BigInt(0),initialStationaryExactly:stationary,initialOrigin:x0.n===BigInt(0)&&y0.n===BigInt(0),memoryKernelExactlyZero:bc.n===BigInt(0),observedStateAutonomousForAllInitialConditions:b.n===BigInt(0),initialQuasiSteadyExactly:mismatch.n===BigInt(0),physicalNoiseSpecified:false,physicalEnergyOrEntropySpecified:false};
  if(!Object.entries(expected).every(([k,v])=>(r.classification as Record<string,unknown>)[k]===v))return false;
  records=r.history as Record<string,unknown>[];
  for(const [i,h]of records.entries()){
   const time=mul(T,f(BigInt(i),BigInt(p.sampleSteps))),u=mul(eta,mul(time,time)),absU=u.n<BigInt(0)?neg(u):u,method=time.n===BigInt(0)?'exact-zero-time':le(absU,f(BigInt(1),BigInt(16)))?'coalescence-series':eta.n>BigInt(0)?'real-decaying-modes':'damped-rotation';
   if(!matches(h.time,time)||h.flowMethod!==method||h.initialHiddenRoundedValueUnavailable!==((h.initialHiddenTerm as Q).value===null)||h.initialHiddenMagnitudeUnresolvedAtWorkingPrecision!==(forcing.n!==BigInt(0)&&containsZero(h.initialHiddenTerm)))return false;
   if((i===0||stationary)&&(!encloses(h.x,x0)||!encloses(h.y,y0)))return false;
   if(bc.n===BigInt(0)&&((h.memoryIntegral as Q).resolution!=='structural-zero'||(h.memoryKernelAtLag as Q).resolution!=='structural-zero'))return false;
   if(forcing.n===BigInt(0)&&(h.initialHiddenTerm as Q).resolution!=='structural-zero')return false;
  }
 }
 return records.every((row,i)=>row.step===i)&&same(r.current,records.at(-1));
}
