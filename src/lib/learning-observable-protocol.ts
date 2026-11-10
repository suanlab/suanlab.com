/** Reader-safe transport. Calculation engines below are type-only imports. */
import {hermitianKeys,sequentialKeys,isHermitianParameters,isSequentialObservationParameters,type HermitianParameters,type SequentialObservationParameters} from './learning-quantum-observable-input';
import {isMasterObservation} from './learning-master-equation-protocol';
import {binaryRatioObservation} from './learning-binary-products';
import {rationalGreatestCommonDivisor as gcd} from './learning-rational-gcd';
import {parseNumberJsonDraft} from './learning-json-number-input';
import type {hermitianSpectrumModel} from './learning-hermitian-spectrum';
import type {sequentialObservationModel} from './learning-sequential-observation';
import type {spectralMeasurement} from './learning-observable-matrix';
import type {PartitionObservation as Q} from './learning-partition-exact-core';
export const observableKinds=['hermitian-spectrum-born-expectation','degenerate-projectors-sequential-observation'] as const;
export type ObservableInput={kind:typeof observableKinds[0];parameters:HermitianParameters}|{kind:typeof observableKinds[1];parameters:SequentialObservationParameters};
export type ObservableSuccess={version:1;requestId:number;ok:true;inputKey:string}&({kind:typeof observableKinds[0];input:Extract<ObservableInput,{kind:typeof observableKinds[0]}>;result:ReturnType<typeof hermitianSpectrumModel>}|{kind:typeof observableKinds[1];input:Extract<ObservableInput,{kind:typeof observableKinds[1]}>;result:ReturnType<typeof sequentialObservationModel>});
export type ObservableResponse=ObservableSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const observableRecord=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype;
type Check=(x:unknown)=>boolean;
const shape=(x:unknown,s:Record<string,Check>):x is Record<string,unknown>=>observableRecord(x)&&Reflect.ownKeys(x).length===Object.keys(s).length&&Object.entries(s).every(([k,c])=>Object.hasOwn(x,k)&&c(x[k]));
const literal=(v:unknown):Check=>x=>x===v,choice=(...v:unknown[]):Check=>x=>v.includes(x),bool:Check=x=>typeof x==='boolean',integer=(x:unknown,min=0,max=Number.MAX_SAFE_INTEGER):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
const array=(n:number,c:Check):Check=>x=>Array.isArray(x)&&Object.getPrototypeOf(x)===Array.prototype&&x.length===n&&Reflect.ownKeys(x).length===n+1&&Array.from({length:n},(_,i)=>Object.hasOwn(x,i)&&c(x[i])).every(Boolean);
const text:Check=x=>typeof x==='string'&&x.length>0&&x.length<12000;
export function isObservableInput(x:unknown):x is ObservableInput{return shape(x,{kind:choice(...observableKinds),parameters:observableRecord})&&(x.kind===observableKinds[0]?isHermitianParameters(x.parameters):isSequentialObservationParameters(x.parameters));}
export function snapshotObservableInput(x:ObservableInput):ObservableInput{if(!isObservableInput(x))throw new RangeError('Invalid observable preparation');const keys=x.kind===observableKinds[0]?hermitianKeys:sequentialKeys,p=x.parameters as Record<string,number>;return Object.freeze({kind:x.kind,parameters:Object.freeze(Object.fromEntries(keys.map(k=>[k,p[k]+0])))}) as ObservableInput;}
export const observableInputKey=(x:ObservableInput)=>JSON.stringify(snapshotObservableInput(x));
export const isObservableRequestEnvelope=(x:unknown):x is{version:1;requestId:number;input:unknown}=>shape(x,{version:literal(1),requestId:x=>integer(x,1),input:()=>true});
export function parseObservableInputDraft(raw:string):ObservableInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['parameters']);return isObservableInput(x)?snapshotObservableInput(x):null;}catch{return null;}}
export function parseObservableStoredInput(raw:string):ObservableInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['version','parameters']);return shape(x,{version:literal(1),value:isObservableInput})?snapshotObservableInput(x.value as ObservableInput):null;}catch{return null;}}
// Bounded scalar consistency checks, never matrix projection or engine evaluation.
type F={n:bigint;d:bigint};
const f=(n:bigint,d=BigInt(1)):F=>{if(d<BigInt(0)){n=-n;d=-d;}const g=gcd(n,d);return{n:n/g,d:d/g};},zero=f(BigInt(0)),one=f(BigInt(1)),add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),sub=(a:F,b:F)=>f(a.n*b.d-b.n*a.d,a.d*b.d),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),eq=(a:F,b:F)=>a.n*b.d===b.n*a.d,le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
const lo=(q:Q)=>f(BigInt(q.bounds!.lower.numerator.coefficient),BigInt(q.bounds!.lower.denominator.coefficient)),hi=(q:Q)=>f(BigInt(q.bounds!.upper.numerator.coefficient),BigInt(q.bounds!.upper.denominator.coefficient)),sum=(xs:F[])=>xs.reduce(add,zero),matches=(q:Q,x:F)=>eq(lo(q),x)&&eq(hi(q),x);
type Measurement=ReturnType<typeof spectralMeasurement>['observation'];
type Sequence=ReturnType<typeof sequentialObservationModel>['aThenB'];
/** Validates complete wire shapes, preparation attribution and scalar accounts.
 * It does not prove the mathematics of an arbitrary sender; independent symbolic
 * references verify the calculation engines. No synchronous fallback exists. */
export function isObservableSuccessForInput(value:unknown,input:ObservableInput,id:number):value is ObservableSuccess{
 if(!integer(id,1)||!isObservableInput(input)||!shape(value,{version:literal(1),requestId:literal(id),ok:literal(true),kind:literal(input.kind),input:isObservableInput,inputKey:literal(observableInputKey(input)),result:observableRecord})||observableInputKey(value.input as ObservableInput)!==observableInputKey(input))return false;
 const r=value.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
 const q:Check=x=>{
  if(!observableRecord(x))return false;if(cache.has(x))return cache.get(x)!;
  const bounds=x.bounds;
  let ok=observableRecord(bounds)&&['lower','upper'].every(k=>{const b=bounds[k];return observableRecord(b)&&['numerator','denominator'].every(k=>{const t=b[k];return observableRecord(t)&&typeof t.coefficient==='string'&&t.coefficient.length<=4096;});})&&isMasterObservation(x);
  if(ok){const v=x as unknown as Q;ok=le(lo(v),hi(v));if(ok&&v.resolution!=='unresolved-at-precision-ceiling'){const midpoint=div(add(lo(v),hi(v)),f(BigInt(2))),drawing=binaryRatioObservation({coefficient:midpoint.n,exponent:0},{coefficient:midpoint.d,exponent:0});ok=v.value===drawing.value&&(v.logMagnitude===null?drawing.logMagnitude===null:typeof drawing.logMagnitude==='number'&&Math.abs(v.logMagnitude-drawing.logMagnitude)<=1e-12*Math.max(1,Math.abs(drawing.logMagnitude)));}}
  cache.set(x,ok);return ok;
 };
 const exact:Check=x=>q(x)&&isMasterObservation(x,true),z:Check=x=>exact(x)&&(x as Q).sign===0,prob:Check=x=>exact(x)&&le(zero,lo(x as Q))&&le(lo(x as Q),one),positive:Check=x=>exact(x)&&lo(x as Q).n>BigInt(0),nonnegative:Check=x=>exact(x)&&lo(x as Q).n>=BigInt(0),complex=(check:Check=q):Check=>x=>shape(x,{real:check,imaginary:check}),matrix=(n:number,check:Check=exact)=>array(n,array(n,complex(check))),nullable=(check:Check):Check=>x=>x===null||check(x);
 function measurementSchema(n:number,m:number):Record<string,Check>{return{
  matrix:matrix(n),outcomes:array(m,x=>shape(x,{eigenvalue:exact,rank:x=>integer(x,1,n),projector:matrix(n),projectedNumerators:array(n,complex(exact)),probability:prob,conditionalAmplitudes:nullable(array(n,complex())),expectationContribution:exact})),mean:exact,secondMoment:nonnegative,variance:nonnegative,nonselectiveDensity:matrix(n),nonselectivePurity:prob,
  residuals:x=>shape(x,{hermiticity:matrix(n,z),projectorIdempotence:array(m,matrix(n,z)),projectorHermiticity:array(m,matrix(n,z)),projectorOrthogonality:array(m,array(m,nullable(matrix(n,z)))),projectorCompleteness:matrix(n,z),probabilityMass:z,directMean:complex(z),directSecondMoment:z,nonselectiveTrace:complex(z)})};}
 const p=input.parameters as Record<string,number>,norm=(prefix:string,n:number)=>sum(Array.from({length:n},(_,i)=>add(f(BigInt(p[`${prefix}Real${i}`])**BigInt(2)),f(BigInt(p[`${prefix}Imag${i}`])**BigInt(2))))),N=norm('state',input.kind===observableKinds[0]?2:3);
 const rational=(prefix:string)=>f(BigInt(p[prefix+'Numerator']),BigInt(p[prefix+'Denominator']));
 const spec=(positive:F,negative:F,n:number)=>eq(positive,negative)?[{value:positive,rank:n}]:[{value:positive,rank:1},{value:negative,rank:n-1}].sort((a,b)=>le(a.value,b.value)?-1:1);
 const common={parameters:(x:unknown)=>(input.kind===observableKinds[0]?isHermitianParameters(x):isSequentialObservationParameters(x))&&JSON.stringify(x)===JSON.stringify(snapshotObservableInput(input).parameters),precisionBits:literal(512),rawNormSquared:positive,stateAmplitudes:array(input.kind===observableKinds[0]?2:3,complex()),approximationContract:text};
 function consistentMeasurement(m:Measurement,n:number,specs:{value:F;rank:number}[]){
  if(!matches(r.rawNormSquared as Q,N)||!matches(m.mean,sum(m.outcomes.map(b=>mul(lo(b.eigenvalue),lo(b.probability)))))||!matches(m.secondMoment,sum(m.outcomes.map(b=>mul(mul(lo(b.eigenvalue),lo(b.eigenvalue)),lo(b.probability)))))||!matches(m.variance,sub(lo(m.secondMoment),mul(lo(m.mean),lo(m.mean))))||!eq(sum(m.outcomes.map(b=>lo(b.probability))),one))return false;
  if(!eq(sum(m.nonselectiveDensity.map((row,i)=>lo(row[i].real))),one)||!m.nonselectiveDensity.every((row,i)=>matches(row[i].imaginary,zero)))return false;
  for(const[i,b]of m.outcomes.entries()){
   if(!matches(b.eigenvalue,specs[i].value)||b.rank!==specs[i].rank||(lo(b.probability).n===BigInt(0)?b.conditionalAmplitudes!==null:b.conditionalAmplitudes===null)||!matches(b.expectationContribution,mul(lo(b.eigenvalue),lo(b.probability))))return false;
   const mass=sum(b.projectedNumerators.map(v=>add(mul(lo(v.real),lo(v.real)),mul(lo(v.imaginary),lo(v.imaginary)))));
   if(!matches(b.probability,div(mass,N))||!eq(sum(b.projector.map((row,j)=>lo(row[j].real))),f(BigInt(b.rank))))return false;
   for(let j=0;j<specs.length;j++)if((i===j)!==(m.residuals.projectorOrthogonality[i][j]===null))return false;
  }
  return m.matrix.length===n;
 }
 if(input.kind===observableKinds[0]){
  const center=rational('center'),gap=rational('gap'),s=spec(add(center,gap),sub(center,gap),2);
  if(!shape(r,{...common,dimension:literal(2),...measurementSchema(2,s.length),passiveBasis:x=>shape(x,{rawNormSquared:positive,columns:array(2,array(2,complex())),coordinateNumerators:array(2,complex(exact)),coordinates:array(2,complex()),matrix:matrix(2),mean:complex(exact),meanResidual:complex(z),matrixReconstructionResidual:matrix(2,z)}),nonHermitianCounterexample:x=>shape(x,{matrix:matrix(2),expectation:complex(exact),hermiticityDefect:matrix(2)}),classification:x=>shape(x,{spectralOutcomes:literal(s.length),fullyDegenerate:literal(s.length===1),varianceIsExactlyZero:bool,eigenstateForThisObservable:bool,negativeGapReordered:literal(gap.n<BigInt(0)),finiteDimensionOnly:literal(true),physicalTimeSpecified:literal(false)})}))return false;
  const x=r as unknown as ReturnType<typeof hermitianSpectrumModel>;
  return consistentMeasurement(x,2,s)&&matches(x.passiveBasis.rawNormSquared,norm('basis',2))&&matches(x.passiveBasis.mean.real,lo(x.mean))&&matches(x.passiveBasis.mean.imaginary,zero)&&x.classification.varianceIsExactlyZero===(lo(x.variance).n===BigInt(0))&&x.classification.eigenstateForThisObservable===x.classification.varianceIsExactlyZero;
 }
 const sa=spec(rational('aPositive'),rational('aNegative'),3),sb=spec(rational('bPositive'),rational('bNegative'),3),fine=[{value:rational('aPositive'),rank:1},{value:rational('aNegative'),rank:1},{value:rational('aNegative'),rank:1}];
 const sequenceSchema=(m:number,k:number):Check=>x=>shape(x,{jointProbabilities:array(m,array(k,prob)),firstMarginal:array(m,prob),secondMarginal:array(k,prob),conditionalSecondProbabilities:array(m,nullable(array(k,prob))),residuals:x=>shape(x,{totalMass:z,firstBornDifferences:array(m,z),secondDensityDifferences:array(k,complex(z))})});
 if(!shape(r,{...common,dimension:literal(3),a:x=>shape(x,measurementSchema(3,sa.length)),b:x=>shape(x,measurementSchema(3,sb.length)),refinedA:x=>shape(x,{...measurementSchema(3,3),labels:array(3,text),selectedStandardAxis:x=>integer(x,0,2),distinctInstrumentEvenWhenEigenvaluesCoincide:literal(true)}),aThenB:sequenceSchema(sa.length,sb.length),bThenA:sequenceSchema(sb.length,sa.length),refinedAThenB:sequenceSchema(3,sb.length),commutator:matrix(3),commutatorSquaredNorm:nonnegative,originalDensity:matrix(3),coarseMinusRefinedDensity:matrix(3),coarseRefinementSquaredDistance:nonnegative,secondMarginalCoarseMinusRefined:array(sb.length,exact),classification:x=>shape(x,{aFullyDegenerate:literal(sa.length===1),bFullyDegenerate:literal(sb.length===1),aSpectralOutcomeCount:literal(sa.length),bSpectralOutcomeCount:literal(sb.length),observablesCommute:bool,coarseAndRefinedNonselectiveStateEqual:bool,coarsePreservesInitialState:bool,refinementIsSeparateInstrument:literal(true),physicalTimeSpecified:literal(false),arbitraryMixedStateInput:literal(false)})}))return false;
 const x=r as unknown as ReturnType<typeof sequentialObservationModel>;
 const selectedAxis=['axisAReal1','axisAImag1','axisAReal2','axisAImag2'].every(k=>p[k]===0)?1:0;
 if(!consistentMeasurement(x.a,3,sa)||!consistentMeasurement(x.b,3,sb)||!consistentMeasurement(x.refinedA,3,fine)||JSON.stringify(x.refinedA.labels)!=='["P","S","T"]'||x.refinedA.selectedStandardAxis!==selectedAxis)return false;
 function consistentSequence(s:Sequence,first:Measurement){
  if(!eq(sum(s.firstMarginal.map(lo)),one))return false;
  for(const[i,row]of s.jointProbabilities.entries()){
   const mass=sum(row.map(lo));if(!matches(s.firstMarginal[i],mass)||!matches(first.outcomes[i].probability,mass))return false;
   const conditional=s.conditionalSecondProbabilities[i];if(mass.n===BigInt(0)?conditional!==null:conditional===null)return false;
   if(conditional&&!conditional.every((q,j)=>matches(q,div(lo(row[j]),mass))))return false;
  }
  return s.secondMarginal.every((q,j)=>matches(q,sum(s.jointProbabilities.map(row=>lo(row[j])))));
 }
 if(!consistentSequence(x.aThenB,x.a)||!consistentSequence(x.bThenA,x.b)||!consistentSequence(x.refinedAThenB,x.refinedA))return false;
 const matrixZero=(a:ReturnType<typeof sequentialObservationModel>['commutator'])=>a.every(row=>row.every(v=>matches(v.real,zero)&&matches(v.imaginary,zero)));
 const squareNorm=(a:ReturnType<typeof sequentialObservationModel>['commutator'])=>sum(a.flat().map(v=>add(mul(lo(v.real),lo(v.real)),mul(lo(v.imaginary),lo(v.imaginary)))));
 if(!matches(x.commutatorSquaredNorm,squareNorm(x.commutator))||!matches(x.coarseRefinementSquaredDistance,squareNorm(x.coarseMinusRefinedDensity))||x.classification.observablesCommute!==matrixZero(x.commutator)||x.classification.coarseAndRefinedNonselectiveStateEqual!==matrixZero(x.coarseMinusRefinedDensity))return false;
 const sameInitial=x.a.nonselectiveDensity.every((row,i)=>row.every((v,j)=>eq(lo(v.real),lo(x.originalDensity[i][j].real))&&eq(lo(v.imaginary),lo(x.originalDensity[i][j].imaginary))));
 if(x.classification.coarsePreservesInitialState!==sameInitial||!x.coarseMinusRefinedDensity.every((row,i)=>row.every((v,j)=>matches(v.real,sub(lo(x.a.nonselectiveDensity[i][j].real),lo(x.refinedA.nonselectiveDensity[i][j].real)))&&matches(v.imaginary,sub(lo(x.a.nonselectiveDensity[i][j].imaginary),lo(x.refinedA.nonselectiveDensity[i][j].imaginary))))))return false;
 return x.secondMarginalCoarseMinusRefined.every((q,i)=>matches(q,sub(lo(x.aThenB.secondMarginal[i]),lo(x.refinedAThenB.secondMarginal[i]))));
}
