/** Reader-safe owned wire schemas and bounded attribution; no numerical engine import. */
import {isDensityEnsembleParameters,snapshotDensityEnsembleParameters,type DensityEnsembleParameters} from './learning-density-ensemble-input';
import {isDensitySpectrumParameters,snapshotDensitySpectrumParameters,type DensitySpectrumParameters} from './learning-density-spectrum-input';
import {isMasterObservation} from './learning-master-equation-protocol';
import {binaryRatioObservation} from './learning-binary-products';
import {rationalGreatestCommonDivisor as gcd} from './learning-rational-gcd';
import {parseNumberJsonDraft} from './learning-json-number-input';
import type {densityEnsembleModel} from './learning-density-ensemble';
import type {densitySpectrumModel} from './learning-density-spectrum';
import type {PartitionObservation as Q} from './learning-partition-exact-core';
export const densityKinds=['finite-ensemble-density-gram-tomography','qubit-density-spectrum-entropy-dephasing'] as const;
export type DensityInput={kind:typeof densityKinds[0];parameters:DensityEnsembleParameters}|{kind:typeof densityKinds[1];parameters:DensitySpectrumParameters};
export type DensitySuccess={version:1;requestId:number;ok:true;inputKey:string}&({kind:typeof densityKinds[0];input:Extract<DensityInput,{kind:typeof densityKinds[0]}>;result:ReturnType<typeof densityEnsembleModel>}|{kind:typeof densityKinds[1];input:Extract<DensityInput,{kind:typeof densityKinds[1]}>;result:ReturnType<typeof densitySpectrumModel>});
export type DensityResponse=DensitySuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const densityRecord=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype;
type Check=(x:unknown)=>boolean;
const shape=(x:unknown,s:Record<string,Check>):x is Record<string,unknown>=>densityRecord(x)&&Reflect.ownKeys(x).length===Object.keys(s).length&&Object.entries(s).every(([k,c])=>Object.hasOwn(x,k)&&c(x[k]));
const literal=(v:unknown):Check=>x=>x===v,choice=(...v:unknown[]):Check=>x=>v.includes(x),bool:Check=x=>typeof x==='boolean';
const integer=(x:unknown,min=0,max=Number.MAX_SAFE_INTEGER):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
const array=(n:number,c:Check):Check=>x=>Array.isArray(x)&&Object.getPrototypeOf(x)===Array.prototype&&x.length===n&&Reflect.ownKeys(x).length===n+1&&Array.from({length:n},(_,i)=>Object.hasOwn(x,i)&&c(x[i])).every(Boolean);
const text:Check=x=>typeof x==='string'&&x.length>0&&x.length<12000,nullable=(c:Check):Check=>x=>x===null||c(x);
const fields=(names:string,c:Check)=>Object.fromEntries(names.split(' ').map(k=>[k,c]));
export function isDensityInput(x:unknown):x is DensityInput{return shape(x,{kind:choice(...densityKinds),parameters:densityRecord})&&(x.kind===densityKinds[0]?isDensityEnsembleParameters(x.parameters):isDensitySpectrumParameters(x.parameters));}
export function snapshotDensityInput(x:DensityInput):DensityInput{if(!isDensityInput(x))throw new RangeError('Invalid density preparation');return Object.freeze(x.kind===densityKinds[0]?{kind:x.kind,parameters:snapshotDensityEnsembleParameters(x.parameters)}:{kind:x.kind,parameters:snapshotDensitySpectrumParameters(x.parameters)});}
export const densityInputKey=(x:DensityInput)=>JSON.stringify(snapshotDensityInput(x));
export const isDensityRequestEnvelope=(x:unknown):x is{version:1;requestId:number;input:unknown}=>shape(x,{version:literal(1),requestId:x=>integer(x,1),input:()=>true});
export function parseDensityInputDraft(raw:string):DensityInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['parameters']);return isDensityInput(x)?snapshotDensityInput(x):null;}catch{return null;}}
export function parseDensityStoredInput(raw:string):DensityInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const x=parseNumberJsonDraft(raw,['version','parameters']);return shape(x,{version:literal(1),value:isDensityInput})?snapshotDensityInput(x.value as DensityInput):null;}catch{return null;}}
type F={n:bigint;d:bigint};
const f=(n:bigint,d=BigInt(1)):F=>{if(d<BigInt(0)){n=-n;d=-d;}const g=gcd(n,d);return{n:n/g,d:d/g};};
const zero=f(BigInt(0)),one=f(BigInt(1)),two=f(BigInt(2)),int=(x:number)=>f(BigInt(x));
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),eq=(a:F,b:F)=>a.n*b.d===b.n*a.d,le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,sum=(xs:F[])=>xs.reduce(add,zero),square=(x:F)=>mul(x,x);
const lo=(q:Q)=>f(BigInt(q.bounds!.lower.numerator.coefficient),BigInt(q.bounds!.lower.denominator.coefficient)),hi=(q:Q)=>f(BigInt(q.bounds!.upper.numerator.coefficient),BigInt(q.bounds!.upper.denominator.coefficient));
const matches=(q:Q,x:F)=>eq(lo(q),x)&&eq(hi(q),x);
/** Validate complete shape, bounded certificates and input-specific scalar accounts.
 * Independent references establish the engine's mathematics. These inexpensive
 * accounts do not recompute logarithms, diagonalize matrices or prove an arbitrary
 * sender truthful. Invalid frames never trigger a synchronous calculation fallback.
 */
export function isDensitySuccessForInput(value:unknown,input:DensityInput,id:number):value is DensitySuccess{
 if(!integer(id,1)||!isDensityInput(input)||!shape(value,{version:literal(1),requestId:literal(id),ok:literal(true),kind:literal(input.kind),input:isDensityInput,inputKey:literal(densityInputKey(input)),result:densityRecord})||densityInputKey(value.input as DensityInput)!==densityInputKey(input))return false;
 const r=value.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
 const term=(positive=false):Check=>x=>shape(x,{coefficient:v=>typeof v==='string'&&v.length<=4096&&(positive?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(v)&&v!=='-0',exponent:literal(0)});
 const fraction:Check=x=>shape(x,{numerator:term(),denominator:term(true)});
 const q:Check=x=>{
  if(!densityRecord(x))return false;if(cache.has(x))return cache.get(x)!;
  let ok=shape(x,{value:v=>v===null||typeof v==='number'&&Number.isFinite(v),sign:choice(-1,0,1,null),logMagnitude:v=>v===null||typeof v==='number'&&Number.isFinite(v),logResolution:choice('finite','not-applicable','unresolved'),resolution:choice('structural-zero','exact-rational','interval-resolved','unresolved-at-precision-ceiling'),bounds:v=>shape(v,{lower:fraction,upper:fraction})})&&isMasterObservation(x);
  if(ok){const v=x as unknown as Q,a=lo(v),b=hi(v),abs=(v:F)=>v.n<BigInt(0)?neg(v):v,scale=[one,abs(a),abs(b)].reduce((best,v)=>le(best,v)?v:best,one);ok=le(a,b)&&le(mul(sub(b,a),f(BigInt(10)**BigInt(125))),scale);
   if(ok&&v.resolution!=='unresolved-at-precision-ceiling'){const mid=div(add(a,b),two),drawing=binaryRatioObservation({coefficient:mid.n,exponent:0},{coefficient:mid.d,exponent:0});ok=v.value===drawing.value&&(v.logMagnitude===null?drawing.logMagnitude===null:typeof drawing.logMagnitude==='number'&&Math.abs(v.logMagnitude-drawing.logMagnitude)<=1e-12*Math.max(1,Math.abs(drawing.logMagnitude)));}}
  cache.set(x,ok);return ok;
 };
 const exact:Check=x=>q(x)&&isMasterObservation(x,true),z:Check=x=>exact(x)&&(x as Q).sign===0;
 const nonnegative:Check=x=>exact(x)&&le(zero,lo(x as Q)),positive:Check=x=>nonnegative(x)&&lo(x as Q).n>BigInt(0),nonnegativeInterval:Check=x=>q(x)&&le(zero,lo(x as Q));
 const prob:Check=x=>nonnegativeInterval(x)&&le(hi(x as Q),one),exactProb:Check=x=>exact(x)&&prob(x),contains=(target:F):Check=>x=>q(x)&&le(lo(x as Q),target)&&le(target,hi(x as Q)),residual=contains(zero);
 const complex=(c:Check=q):Check=>x=>shape(x,{real:c,imaginary:c}),matrix=(d:number,c:Check=q)=>array(d,array(d,complex(c)));
 const p=input.parameters as unknown as Record<string,number>,parameters:Check=x=>(input.kind===densityKinds[0]?isDensityEnsembleParameters(x):isDensitySpectrumParameters(x))&&Object.entries(p).every(([k,v])=>(x as Record<string,number>)[k]===v);
 const common={parameters,precisionBits:literal(512),arithmeticWorkBits:literal(768),approximationContract:text};
 if(input.kind===densityKinds[0]){
  const d=p.dimension,mat=(c:Check=q)=>matrix(d,c),norm=(prefix:string)=>sum(Array.from({length:d},(_,j)=>add(square(int(p[`${prefix}Real${j}`])),square(int(p[`${prefix}Imag${j}`]))))),norms=Array.from({length:4},(_,i)=>norm(`component${i}`)),W=sum(Array.from({length:4},(_,i)=>int(p[`weight${i}`]))),weights=Array.from({length:4},(_,i)=>div(int(p[`weight${i}`]),W));
  const sets=(n:number)=>Array.from({length:2**n-1},(_,i)=>Array.from({length:n},(_,j)=>j).filter(j=>((i+1)&2**j)!==0));
  const minors=(n:number):Check=>v=>array(2**n-1,x=>shape(x,{indices:y=>Array.isArray(y)&&array(y.length,z=>integer(z,0,n-1))(y),determinant:y=>shape(y,{real:nonnegative,imaginary:z})}))(v)&&(v as {indices:number[]}[]).every((x,i)=>JSON.stringify(x.indices)===JSON.stringify(sets(n)[i]));
  const pairCount=d*(d-1)/2,branchCount=p.gapNumerator===0||d===1?1:2;
  const component:Check=x=>shape(x,{index:v=>integer(v,0,3),occupied:bool,probability:exactProb,rawNormSquared:nonnegative,normalizedAmplitudes:nullable(array(d,complex())),pureDensity:nullable(mat(exact)),observableMean:nullable(exact),observableVariance:nullable(nonnegative)});
  const outcome:Check=x=>shape(x,{eigenvalue:exact,rank:v=>integer(v,1,d),projector:mat(exact),probability:exactProb,conditionalDensity:nullable(mat(exact)),conditionalPurity:nullable(exactProb)});
  if(!shape(r,{...common,dimension:literal(d),ensembleSize:literal(4),totalRawWeight:positive,weights:array(4,exactProb),components:array(4,component),density:mat(exact),standardProbabilities:array(d,exactProb),purity:exactProb,linearEntropy:exactProb,purityAboveMaximallyMixed:nonnegative,rank:v=>integer(v,1,d),principalMinors:minors(d),characteristicPolynomial:v=>shape(v,{coefficientOrder:literal('descending powers of lambda'),density:array(d+1,exact),gram:array(5,exact)}),pairFidelities:array(4,array(4,nullable(exactProb))),gramSquaredMagnitudes:array(4,array(4,exactProb)),purityFromOverlaps:exactProb,gramRank:v=>integer(v,1,d),gramPrincipalMinors:minors(4),weightedColumns:array(4,array(d,complex())),gramMatrix:matrix(4),observable:v=>shape(v,{matrix:mat(exact),mean:exact,secondMoment:nonnegative,variance:nonnegative,withinEnsembleVariance:nonnegative,betweenEnsembleMeanVariance:nonnegative,outcomes:array(branchCount,outcome),nonselectiveDensity:mat(exact),nonselectivePurity:exactProb,purityLost:nonnegative,disturbanceSquaredHilbertSchmidt:nonnegative}),probeComparison:v=>shape(v,{pureDensity:mat(exact),overlapProbability:exactProb,squaredHilbertSchmidtDifference:nonnegative,basisProbabilities:array(d,exactProb)}),tomography:v=>shape(v,{pairs:array(pairCount,x=>shape(x,{i:y=>integer(y,0,d-1),j:y=>integer(y,0,d-1),probabilities:y=>shape(y,fields('xPlus xMinus yPlus yMinus outside',exactProb))})),reconstructedDensity:mat(exact)}),standardBasisDephasing:v=>shape(v,{density:mat(exact),purity:exactProb,squaredOffDiagonalCoherence:nonnegative}),qubitBloch:d===2?array(3,exact):literal(null),exactResiduals:v=>shape(v,{hermiticity:mat(z),trace:complex(z),ensembleWeight:z,purityGram:z,gramElementaryCoefficients:array(5,z),tomography:mat(z),probabilityMass:z,measurementMass:z,spectralMean:z,spectralSecondMoment:z,totalVariance:z,pinchingPurity:z,idempotenceWhenPure:nullable(mat(z))}),directedResiduals:v=>shape(v,{densityFactorization:mat(residual),gramTrace:complex(residual),gramPurity:complex(residual)}),classification:v=>shape(v,{pureState:bool,rankOne:bool,occupiedComponents:y=>integer(y,1,4),allOccupiedComponentsSameRay:bool,maximallyMixed:bool,finiteDimensionOnly:literal(true),entropyCalculated:literal(false),generalSpectrumNumericallyDiagonalized:literal(false),physicalTimeSpecified:literal(false),tensorProductInputSupplied:literal(false),partialTraceSupplied:literal(false),measurementInstrument:literal('Luders-spectral-projectors')})}))return false;
  const x=r as unknown as ReturnType<typeof densityEnsembleModel>,occupied=weights.map((w,i)=>w.n>BigInt(0)?i:-1).filter(i=>i>=0);
  if(!matches(x.totalRawWeight,W)||!x.weights.every((w,i)=>matches(w,weights[i]))||!x.components.every((v,i)=>v.index===i&&v.occupied===(weights[i].n>BigInt(0))&&matches(v.probability,weights[i])&&matches(v.rawNormSquared,norms[i])&&(v.normalizedAmplitudes===null)===(norms[i].n===BigInt(0))&&(v.pureDensity===null)===(norms[i].n===BigInt(0))&&(v.observableMean===null)===(norms[i].n===BigInt(0))&&(v.observableVariance===null)===(norms[i].n===BigInt(0))))return false;
  // Attribute every density entry to the requested raw ensemble without square roots.
  for(let i=0;i<d;i++)for(let j=0;j<d;j++){
   const real=sum(occupied.map(k=>mul(div(weights[k],norms[k]),add(mul(int(p[`component${k}Real${i}`]),int(p[`component${k}Real${j}`])),mul(int(p[`component${k}Imag${i}`]),int(p[`component${k}Imag${j}`]))))));
   const imaginary=sum(occupied.map(k=>mul(div(weights[k],norms[k]),sub(mul(int(p[`component${k}Imag${i}`]),int(p[`component${k}Real${j}`])),mul(int(p[`component${k}Real${i}`]),int(p[`component${k}Imag${j}`]))))));
   if(!matches(x.density[i][j].real,real)||!matches(x.density[i][j].imaginary,imaginary)||!matches(x.tomography.reconstructedDensity[i][j].real,real)||!matches(x.tomography.reconstructedDensity[i][j].imaginary,imaginary)||!matches(x.standardBasisDephasing.density[i][j].real,i===j?real:zero)||!matches(x.standardBasisDephasing.density[i][j].imaginary,zero))return false;
  }
  const purity=sum(x.density.flat().map(v=>add(square(lo(v.real)),square(lo(v.imaginary))))),pinched=sum(x.standardProbabilities.map(lo).map(square));
  const rank=(m:typeof x.principalMinors)=>Math.max(...m.filter(v=>v.determinant.real.sign===1).map(v=>v.indices.length));
  if(!matches(x.purity,purity)||!matches(x.purityFromOverlaps,purity)||!matches(x.linearEntropy,sub(one,purity))||!matches(x.purityAboveMaximallyMixed,sub(purity,div(one,int(d))))||!x.standardProbabilities.every((v,i)=>matches(v,lo(x.density[i][i].real)))||!matches(x.standardBasisDephasing.purity,pinched)||!matches(x.standardBasisDephasing.squaredOffDiagonalCoherence,sub(purity,pinched))||x.rank!==rank(x.principalMinors)||x.gramRank!==rank(x.gramPrincipalMinors)||x.rank!==x.gramRank)return false;
  const pure=eq(purity,one),maxMixed=eq(purity,div(one,int(d)));
  if(x.classification.pureState!==pure||x.classification.rankOne!==(x.rank===1)||x.classification.maximallyMixed!==maxMixed||x.classification.occupiedComponents!==occupied.length||(x.exactResiduals.idempotenceWhenPure===null)!==!pure||x.classification.allOccupiedComponentsSameRay!==pure)return false;
  if(!x.pairFidelities.every((row,i)=>row.every((v,j)=>(v===null)===(norms[i].n===BigInt(0)||norms[j].n===BigInt(0)))))return false;
  const a=f(BigInt(p.centerNumerator),BigInt(p.centerDenominator)),b=f(BigInt(p.gapNumerator),BigInt(p.gapDenominator));
  const branches=(p.gapNumerator===0?[{e:a,rank:d}]:d===1?[{e:add(a,b),rank:1}]:[{e:add(a,b),rank:1},{e:sub(a,b),rank:d-1}].sort((a,b)=>le(a.e,b.e)?-1:1));
  if(!x.observable.outcomes.every((v,i)=>matches(v.eigenvalue,branches[i].e)&&v.rank===branches[i].rank&&(v.conditionalDensity===null)===(v.probability.sign===0)&&(v.conditionalPurity===null)===(v.probability.sign===0))||!eq(sum(x.observable.outcomes.map(v=>lo(v.probability))),one))return false;
  const probeN=norm('probe'),probeEntry=(i:number,j:number)=>({real:div(add(mul(int(p[`probeReal${i}`]),int(p[`probeReal${j}`])),mul(int(p[`probeImag${i}`]),int(p[`probeImag${j}`]))),probeN),imaginary:div(sub(mul(int(p[`probeImag${i}`]),int(p[`probeReal${j}`])),mul(int(p[`probeReal${i}`]),int(p[`probeImag${j}`]))),probeN)});
  for(let i=0;i<d;i++)for(let j=0;j<d;j++){
   const v=probeEntry(i,j),probe=x.probeComparison.pureDensity[i][j],operator=x.observable.matrix[i][j],base=i===j?sub(a,b):zero;
   if(!matches(probe.real,v.real)||!matches(probe.imaginary,v.imaginary)||!matches(operator.real,add(base,mul(mul(two,b),v.real)))||!matches(operator.imaginary,mul(mul(two,b),v.imaginary)))return false;
   for(let k=0;k<4;k++)if(norms[k].n!==BigInt(0)){
    const raw=(part:'Real'|'Imag',at:number)=>int(p[`component${k}${part}${at}`]),entry=x.components[k].pureDensity![i][j];
    if(!matches(entry.real,div(add(mul(raw('Real',i),raw('Real',j)),mul(raw('Imag',i),raw('Imag',j))),norms[k]))||!matches(entry.imaginary,div(sub(mul(raw('Imag',i),raw('Real',j)),mul(raw('Real',i),raw('Imag',j))),norms[k])))return false;
   }
  }
  const overlap=sum(x.density.flatMap((row,i)=>row.map((v,j)=>sub(mul(lo(v.real),probeEntry(j,i).real),mul(lo(v.imaginary),probeEntry(j,i).imaginary)))));
  if(!matches(x.probeComparison.overlapProbability,overlap)||!matches(x.probeComparison.squaredHilbertSchmidtDifference,sub(add(purity,one),mul(two,overlap)))||!x.probeComparison.basisProbabilities.every((v,i)=>matches(v,probeEntry(i,i).real)))return false;
  if(!matches(x.observable.mean,add(sub(a,b),mul(mul(two,b),overlap)))||!matches(x.observable.secondMoment,add(square(sub(a,b)),mul(mul(int(4),mul(a,b)),overlap))))return false;
  if(!x.observable.outcomes.every(v=>matches(v.probability,branches.length===1?one:eq(lo(v.eigenvalue),add(a,b))?overlap:sub(one,overlap))))return false;
  const o=x.observable;if(!matches(o.variance,sub(lo(o.secondMoment),square(lo(o.mean))))||!matches(o.variance,add(lo(o.withinEnsembleVariance),lo(o.betweenEnsembleMeanVariance)))||!matches(o.mean,sum(o.outcomes.map(v=>mul(lo(v.eigenvalue),lo(v.probability)))))||!matches(o.secondMoment,sum(o.outcomes.map(v=>mul(square(lo(v.eigenvalue)),lo(v.probability)))))||!matches(o.purityLost,sub(purity,lo(o.nonselectivePurity))))return false;
  let pair=0;for(let i=0;i<d;i++)for(let j=i+1;j<d;j++){
   const row=x.tomography.pairs[pair++],diagonal=add(lo(x.standardProbabilities[i]),lo(x.standardProbabilities[j])),half=div(diagonal,two),entry=x.density[i][j];
   if(row.i!==i||row.j!==j||!matches(row.probabilities.xPlus,add(half,lo(entry.real)))||!matches(row.probabilities.xMinus,sub(half,lo(entry.real)))||!matches(row.probabilities.yPlus,sub(half,lo(entry.imaginary)))||!matches(row.probabilities.yMinus,add(half,lo(entry.imaginary)))||!matches(row.probabilities.outside,sub(one,diagonal)))return false;
  }
  const cp=x.characteristicPolynomial;if(!cp.density.every((v,i)=>matches(cp.gram[i],lo(v)))||!cp.gram.slice(d+1).every(v=>v.sign===0)||!matches(cp.density[0],one)||!matches(cp.density[1],neg(one)))return false;
  const elementary=(minors:typeof x.principalMinors,size:number)=>[one,...Array.from({length:size},(_,i)=>sum(minors.filter(v=>v.indices.length===i+1).map(v=>lo(v.determinant.real))))];
  const coefficients=elementary(x.principalMinors,d),gramCoefficients=elementary(x.gramPrincipalMinors,4);
  if(!cp.density.every((v,i)=>matches(v,i%2?neg(coefficients[i]):coefficients[i]))||!cp.gram.every((v,i)=>matches(v,i%2?neg(gramCoefficients[i]):gramCoefficients[i])))return false;
  return true;
 }
 const mat=(c:Check=q)=>matrix(2,c),initial=[p.blochX,p.blochY,p.blochZ].map(v=>f(BigInt(v),BigInt(p.blochDenominator))),initialR2=sum(initial.map(square)),transverse=sum(initial.slice(0,2).map(square)),etaEnd=f(BigInt(p.etaNumerator),BigInt(p.etaDenominator));
 const fractions=new Map<string,F>();for(let i=0;i<=p.sampleSteps;i++){const u=f(BigInt(i),BigInt(p.sampleSteps));fractions.set(u.n+'/'+u.d,u);}
 const crossing=etaEnd.n<BigInt(0)?div(one,sub(one,etaEnd)):null,zeroInserted=crossing!==null&&!fractions.has(crossing.n+'/'+crossing.d);if(crossing!==null)fractions.set(crossing.n+'/'+crossing.d,crossing);
 const ordered=[...fractions.values()].sort((a,b)=>eq(a,b)?0:le(a,b)?-1:1);
 const spectral:Check=x=>shape(x,{eigenvalue:prob,rank:v=>integer(v,1,2),projector:mat(),logEigenvalue:nullable(q),entropyTermNats:nonnegativeInterval,zeroEigenvalueExact:bool});
 const record:Check=x=>shape(x,{index:v=>integer(v,0,ordered.length-1),fraction:exactProb,eta:exact,bloch:array(3,exact),radiusSquared:exactProb,density:mat(exact),determinant:exactProb,purity:exactProb,linearEntropy:exactProb,spectralDirection:nullable(array(3,q)),spectrum:v=>Array.isArray(v)&&[1,2].includes(v.length)&&array(v.length,spectral)(v),entropyNats:nonnegativeInterval,entropyBits:prob,standardBasisDiagonalEntropyNats:nonnegativeInterval,standardBasisCoherenceNats:nonnegativeInterval,probeBasis:v=>shape(v,{density:mat(exact),purity:exactProb,radiusSquared:exactProb,diagonalEntropyNats:nonnegativeInterval,coherenceNats:nonnegativeInterval}),measurement:v=>shape(v,{matrix:mat(exact),projectors:array(2,mat(exact)),axis:array(3,exact),outcomes:array(2,x=>x===-1||x===1),probabilities:array(2,exactProb),mean:exact,variance:exactProb,units:literal('dimensionless Pauli observable,not spin')}),entropyGainFromInitialNats:nonnegativeInterval,coherenceLossFromInitialNats:nonnegativeInterval,traceDistanceToInitial:prob,squaredUhlmannFidelityToInitial:prob,squaredHilbertSchmidtDifferenceToInitial:nonnegative,channel:v=>shape(v,{identityWeight:exactProb,phaseFlipWeight:exactProb,krausOperators:array(2,mat()),choi:matrix(4,exact),choiIndexConvention:literal('output tensor input; index=2*output+input'),choiNormalization:literal('unnormalized; trace=2'),choiSpectrum:y=>Array.isArray(y)&&integer(y.length,2,4)&&array(y.length,z=>shape(z,{eigenvalue:nonnegative,rank:v=>integer(v,1,4)}))(y),choiRank:v=>integer(v,1,2),minimalKrausCount:v=>integer(v,1,2)}),exactResiduals:v=>shape(v,{trace:z,hermiticity:mat(z),purity:z,determinant:z,channelMixture:mat(z),choiAction:mat(z),choiTraceOutput:mat(z),choiTrace:z,probeBasisPurity:z,measurementMass:z,measurementMean:z,measurementSecondMoment:z,probeBasisPopulations:array(2,z)}),directedResiduals:v=>shape(v,{spectralReconstruction:mat(residual),projectorCompleteness:mat(residual),projectorIdempotence:y=>Array.isArray(y)&&[1,2].includes(y.length)&&array(y.length,mat(residual))(y),projectorHermiticity:y=>Array.isArray(y)&&[1,2].includes(y.length)&&array(y.length,mat(residual))(y),eigenEquation:y=>Array.isArray(y)&&[1,2].includes(y.length)&&array(y.length,mat(residual))(y),krausCompleteness:mat(residual),krausAction:mat(residual),coherenceEntropyAccount:residual}),classification:v=>shape(v,{...fields('pureState maximallyMixed spectralEigenspacesMerged selectedStateUnchanged spectrumUnchangedFromInitial unitaryChannel fullyDephasingChannel selectedStateCommutesWithZ',bool),channelCompletelyPositive:literal(true),channelTracePreserving:literal(true),channelUnital:literal(true)})});
 if(!shape(r,{...common,dimension:literal(2),logWorkBits:literal(800),logReportedBits:literal(768),initialDensity:mat(exact),initialBloch:array(3,exact),initialRadiusSquared:exactProb,endpointEta:exact,parameterUnits:literal('dimensionless interpolation u; not physical time'),entropyUnits:v=>shape(v,{nats:literal('natural logarithm'),bits:literal('base-two logarithm')}),lnTwo:nonnegativeInterval,probeBasisUnitary:mat(),records:array(ordered.length,record),classification:v=>shape(v,{physicalTimeSpecified:literal(false),markovianTimeLawSpecified:literal(false),tensorProductInputSupplied:literal(false),parameterPathIsTimeEvolution:literal(false),negativeEtaPermitted:literal(true),zeroCrossingInserted:literal(zeroInserted),continuousParameterPathEntropyNondecreasing:literal(etaEnd.n>=BigInt(0)||transverse.n===BigInt(0))})}))return false;
 const x=r as unknown as ReturnType<typeof densitySpectrumModel>;
 const densityMatches=(m:typeof x.initialDensity,v:F[])=>matches(m[0][0].real,div(add(one,v[2]),two))&&matches(m[1][1].real,div(sub(one,v[2]),two))&&m[0][0].imaginary.sign===0&&m[1][1].imaginary.sign===0&&matches(m[0][1].real,div(v[0],two))&&matches(m[1][0].real,div(v[0],two))&&matches(m[0][1].imaginary,neg(div(v[1],two)))&&matches(m[1][0].imaginary,div(v[1],two));
 if(!densityMatches(x.initialDensity,initial)||!x.initialBloch.every((v,i)=>matches(v,initial[i]))||!matches(x.initialRadiusSquared,initialR2)||!matches(x.endpointEta,etaEnd))return false;
 const a=int(p.probeReal0),b=int(p.probeImag0),c=int(p.probeReal1),d=int(p.probeImag1),N=sum([a,b,c,d].map(square)),axis=[div(mul(two,add(mul(a,c),mul(b,d))),N),div(mul(two,sub(mul(a,d),mul(b,c))),N),div(sub(add(square(a),square(b)),add(square(c),square(d))),N)];
 return x.records.every((row,i)=>{
  const u=ordered[i],eta=add(one,mul(sub(etaEnd,one),u)),v=[mul(eta,initial[0]),mul(eta,initial[1]),initial[2]],r2=sum(v.map(square)),purity=div(add(one,r2),two),pure=eq(r2,one),mixed=r2.n===BigInt(0),sameState=eq(eta,one)||transverse.n===BigInt(0),sameSpectrum=eq(r2,initialR2),unitary=eq(square(eta),one),det=div(sub(one,r2),int(4)),dot=sum(v.map((v,i)=>mul(v,axis[i]))),count=mixed?1:2;
  if(row.index!==i||!matches(row.fraction,u)||!matches(row.eta,eta)||!row.bloch.every((v,i)=>matches(v,[mul(eta,initial[0]),mul(eta,initial[1]),initial[2]][i]))||!densityMatches(row.density,v)||!matches(row.radiusSquared,r2)||!matches(row.purity,purity)||!matches(row.linearEntropy,sub(one,purity))||!matches(row.determinant,det)||!matches(row.probeBasis.purity,purity)||!matches(row.probeBasis.radiusSquared,r2))return false;
  const m=row.measurement;if(!m.axis.every((v,i)=>matches(v,axis[i]))||m.outcomes[0]!==-1||m.outcomes[1]!==1||!matches(m.mean,dot)||!matches(m.variance,sub(one,square(dot)))||!matches(m.probabilities[0],div(sub(one,dot),two))||!matches(m.probabilities[1],div(add(one,dot),two))||!matches(row.probeBasis.density[0][0].real,mul(f(BigInt(1),BigInt(2)),add(one,dot)))||!matches(row.probeBasis.density[1][1].real,div(sub(one,dot),two)))return false;
  for(let a=0;a<2;a++)for(let b=0;b<2;b++){
   const real=a===b?(a===0?axis[2]:neg(axis[2])):axis[0],imaginary=a===b?zero:a===0?neg(axis[1]):axis[1],matrix=m.matrix[a][b];
   if(!matches(matrix.real,real)||!matches(matrix.imaginary,imaginary))return false;
   for(let branch=0;branch<2;branch++){
    const sign=branch===0?neg(one):one,entry=m.projectors[branch][a][b];
    if(!matches(entry.real,div(add(a===b?one:zero,mul(sign,real)),two))||!matches(entry.imaginary,div(mul(sign,imaginary),two)))return false;
   }
  }
  const cl=row.classification;if(cl.pureState!==pure||cl.maximallyMixed!==mixed||cl.spectralEigenspacesMerged!==mixed||cl.selectedStateUnchanged!==sameState||cl.spectrumUnchangedFromInitial!==sameSpectrum||cl.unitaryChannel!==unitary||cl.fullyDephasingChannel!==(eta.n===BigInt(0))||cl.selectedStateCommutesWithZ!==(transverse.n===BigInt(0)))return false;
  if((row.spectralDirection===null)!==mixed||row.spectrum.length!==count||![row.directedResiduals.projectorIdempotence,row.directedResiduals.projectorHermiticity,row.directedResiduals.eigenEquation].every(v=>Array.isArray(v)&&v.length===count)||row.spectrum.some(s=>s.rank!==(mixed?2:1)))return false;
  for(const [j,s]of row.spectrum.entries()){
   const isZero=pure&&j===0;
   if(s.zeroEigenvalueExact!==isZero||(s.logEigenvalue===null)!==isZero)return false;
   if(isZero&&(s.eigenvalue.sign!==0||s.entropyTermNats.sign!==0))return false;
   if(mixed&&!matches(s.eigenvalue,div(one,two)))return false;
  }
  if(!mixed){const high=row.spectrum[1].eigenvalue,a=sub(mul(two,lo(high)),one),b=sub(mul(two,hi(high)),one);if(a.n<BigInt(0)||!le(square(a),r2)||!le(r2,square(b)))return false;
   if(!le(add(lo(row.spectrum[0].eigenvalue),lo(high)),one)||!le(one,add(hi(row.spectrum[0].eigenvalue),hi(high))))return false;}
  if(pure&&(row.entropyNats.sign!==0||row.entropyBits.sign!==0)||mixed&&!matches(row.entropyBits,one))return false;
  if(sameSpectrum&&(row.entropyGainFromInitialNats.sign!==0||row.coherenceLossFromInitialNats.sign!==0))return false;
  if(sameState&&(row.traceDistanceToInitial.sign!==0||!matches(row.squaredUhlmannFidelityToInitial,one)))return false;
  if(!matches(row.squaredHilbertSchmidtDifferenceToInitial,div(mul(square(sub(one,eta)),transverse),two)))return false;
  const ch=row.channel;if(ch.choiRank!==(unitary?1:2)||ch.minimalKrausCount!==ch.choiRank||!matches(ch.identityWeight,div(add(one,eta),two))||!matches(ch.phaseFlipWeight,div(sub(one,eta),two)))return false;
  for(let a=0;a<4;a++)for(let b=0;b<4;b++){const expected=a===b&&(a===0||a===3)?one:(a===0&&b===3||a===3&&b===0)?eta:zero;if(!matches(ch.choi[a][b].real,expected)||ch.choi[a][b].imaginary.sign!==0)return false;}
  const group=new Map<string,{e:F;rank:number}>();for(const e of [zero,zero,sub(one,eta),add(one,eta)]){const key=e.n+'/'+e.d,old=group.get(key);if(old)old.rank++;else group.set(key,{e,rank:1});}
  const groups=[...group.values()].sort((a,b)=>eq(a.e,b.e)?0:le(a.e,b.e)?-1:1);if(ch.choiSpectrum.length!==groups.length||!ch.choiSpectrum.every((s,i)=>s.rank===groups[i].rank&&matches(s.eigenvalue,groups[i].e)))return false;
  return true;
 });
}
