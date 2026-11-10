/** Product channel model; exact, finite and not yet connected to a reader. */
import { add, sub, mul, div } from './learning-quantum-hilbert-core';
import { f,zero,one,quadratic,complex,Z,O,plus,times,conjugate,real,qNeg,qSign,
  mAdd,mSub,mScale,identity,zeros,adjoint,product,trace,mEqual,mWire,tensor,requireMatrix,qDiv,
  principalMinorCertificate,partialTrace,requireDensity,circuitScalarWire,type M,type F,type Q } from './learning-qubit-channel-algebra';

export type WeightedOperator={weight:F;operator:M};
export type QubitMap={kind:'operator-sum';terms:WeightedOperator[]}|{kind:'transpose'}|{kind:'matrix-units';images:[M,M,M,M]};
export const pauliX:M=[[Z,O],[O,Z]],pauliZ:M=[[O,Z],[Z,real(-1)]];
export const pauliY:M=[[Z,complex(quadratic(zero),quadratic(f(BigInt(-1))))],[complex(quadratic(zero),quadratic(one)),Z]];
const unit=(i:number,j:number):M=>Array.from({length:2},(_,a)=>Array.from({length:2},(_,b)=>a===i&&b===j?O:Z));
export function requireMap(map:QubitMap){
  if(map.kind==='transpose')return;
  if(map.kind==='matrix-units'){
    if(map.images.length!==4)throw new RangeError('Four matrix-unit images required');map.images.forEach(a=>requireMatrix(a,2));return;
  }
  if(map.kind!=='operator-sum'||map.terms.length>8)throw new RangeError('At most eight weighted operators');
  map.terms.forEach(t=>{requireMatrix(t.operator,2);if(typeof t.weight.n!=='bigint'||typeof t.weight.d!=='bigint'||t.weight.d<=BigInt(0))throw new RangeError('Exact rational weight required');});
}
export function applyQubitMap(map:QubitMap,rho:M):M {
  requireMap(map);requireMatrix(rho,2);
  if(map.kind==='transpose')return rho[0].map((_,j)=>rho.map(row=>row[j]));
  if(map.kind==='matrix-units'){
    let result=zeros(2);
    map.images.forEach((image,k)=>{
      const coefficient=rho[Math.floor(k/2)][k%2];
      result=mAdd(result,image.map(row=>row.map(v=>times(coefficient,v))));
    });
    return result;
  }
  return map.terms.reduce((s,t)=>mAdd(s,mScale(product(product(t.operator,rho),adjoint(t.operator)),quadratic(t.weight))),zeros(2));
}
export function qubitChoi(map:QubitMap):M {
  let J=zeros(4);
  for(let i=0;i<2;i++)for(let j=0;j<2;j++)J=mAdd(J,tensor(unit(i,j),applyQubitMap(map,unit(i,j))));
  return J;
}
export function qubitMapFromChoi(J:M):QubitMap {
  requireMatrix(J,4);
  const images=Array.from({length:4},(_,k)=>Array.from({length:2},(_,a)=>Array.from({length:2},(_,b)=>J[2*Math.floor(k/2)+a][2*(k%2)+b])));
  return {kind:'matrix-units',images:images as [M,M,M,M]};
}
/** Compose chronologically: first then second. */
export const composeQubitMaps=(first:QubitMap,second:QubitMap):QubitMap=>({kind:'matrix-units',images:[applyQubitMap(second,applyQubitMap(first,unit(0,0))),applyQubitMap(second,applyQubitMap(first,unit(0,1))),applyQubitMap(second,applyQubitMap(first,unit(1,0))),applyQubitMap(second,applyQubitMap(first,unit(1,1)))]});
export function qubitDual(map:QubitMap,E:M):M {
  requireMatrix(E,2);
  const J=qubitChoi(map);
  // Hilbert-Schmidt adjoint. For physical HP maps and Hermitian observables,
  // this also gives Tr[E Phi(rho)] = Tr[Phi*(E) rho].
  return Array.from({length:2},(_,i)=>Array.from({length:2},(_,j)=>{
    let v=Z;for(let a=0;a<2;a++)for(let b=0;b<2;b++)v=plus(v,times(E[a][b],conjugate(J[2*i+a][2*j+b])));return v;
  }));
}
export function applyWithReference(map:QubitMap,rho:M):M {
  requireMatrix(rho,4);const out=zeros(4);
  for(let r=0;r<2;r++)for(let s=0;s<2;s++){
    const block=Array.from({length:2},(_,a)=>Array.from({length:2},(_,b)=>rho[2*r+a][2*s+b]));
    const image=applyQubitMap(map,block);for(let a=0;a<2;a++)for(let b=0;b<2;b++)out[2*r+a][2*s+b]=image[a][b];
  }return out;
}
export function channelReport(map:QubitMap,rho:M){
  requireDensity(rho);const J=qubitChoi(map),certificate=principalMinorCertificate(J);
  const marginal=partialTrace(J,2,2,'first'),unitalImage=applyQubitMap(map,identity(2));
  const dualIdentity=qubitDual(map,identity(2));
  const tracePreserving=mEqual(marginal,identity(2));
  const traceNonincreasing=principalMinorCertificate(mSub(identity(2),dualIdentity)).positiveSemidefinite;
  const output=applyQubitMap(map,rho),outputCertificate=principalMinorCertificate(output);
  const bell=mScale([[O,Z,Z,O],[Z,Z,Z,Z],[Z,Z,Z,Z],[O,Z,Z,O]],quadratic(f(BigInt(1),BigInt(2))));
  const bellOutput=applyWithReference(map,bell);
  return {schemaVersion:1,choiConvention:'reference-output-unnormalized',input:mWire(rho),choi:mWire(J),
    principalMinors:certificate.minors.map(v=>({mask:v.mask,value:circuitScalarWire(v.value)})),
    properties:{hermiticityPreserving:certificate.hermitian,completelyPositive:certificate.positiveSemidefinite,tracePreserving,traceNonincreasing,unital:mEqual(unitalImage,identity(2)),quantumChannel:certificate.positiveSemidefinite&&tracePreserving},
    choiOutputPartialTrace:mWire(marginal),dualIdentity:mWire(dualIdentity),unitalImage:mWire(unitalImage),
    output:mWire(output),outputTrace:circuitScalarWire(trace(output)),
    outputIsDensity:outputCertificate.positiveSemidefinite&&mEqual([[trace(output)]],[[O]]),
    bellOutput:mWire(bellOutput),bellOutputIsDensity:principalMinorCertificate(bellOutput).positiveSemidefinite&&mEqual([[trace(bellOutput)]],[[O]]),
    referenceMarginal:mWire(partialTrace(bellOutput,2,2,'first'))};
}

export type ChannelFamily='identity'|'damping'|'dephasing'|'depolarizing'|'postselect-zero'|'transpose';
function probability(p:F){if(p.d<=BigInt(0)||p.n<BigInt(0)||p.n>p.d)throw new RangeError('Parameter must lie in [0,1]');}
/** Rational stereographic t gives c=(1-t^2)/(1+t^2), s=2t/(1+t^2). */
export function dampingOperators(t:F){
  probability(t);const square=mul(t,t),den=add(one,square),c=div(sub(one,square),den),s=div(mul(f(BigInt(2)),t),den);
  const K0:M=[[O,Z],[Z,complex(quadratic(c))]],K1:M=[[Z,complex(quadratic(s))],[Z,Z]];
  return {c,s,K0,K1};
}
export function channelFamily(family:ChannelFamily,p:F=f(BigInt(1),BigInt(2))):QubitMap {
  probability(p);
  const term=(operator:M,weight=one)=>({weight,operator});
  if(family==='transpose')return {kind:'transpose'};
  if(family==='identity')return {kind:'operator-sum',terms:[term(identity(2))]};
  if(family==='postselect-zero')return {kind:'operator-sum',terms:[term(unit(0,0))]};
  if(family==='damping'){const {K0,K1}=dampingOperators(p);return {kind:'operator-sum',terms:[term(K0),term(K1)]};}
  if(family==='dephasing')return {kind:'operator-sum',terms:[term(identity(2),sub(one,p)),term(pauliZ,p)]};
  if(family==='depolarizing'){
    const quarter=div(p,f(BigInt(4)));return {kind:'operator-sum',terms:[term(identity(2),sub(one,mul(f(BigInt(3)),quarter))),term(pauliX,quarter),term(pauliY,quarter),term(pauliZ,quarter)]};
  }
  throw new RangeError('Unknown channel family');
}

export function dampingDilation(t:F){
  const {c,s,K0,K1}=dampingOperators(t),C=complex(quadratic(c)),S=complex(quadratic(s));
  const V:M=[[O,Z],[Z,S],[Z,C],[Z,Z]];
  const U:M=[[O,Z,Z,Z],[Z,C,S,Z],[Z,complex(qNeg(S.re)),C,Z],[Z,Z,Z,O]];
  return {V,U,K0,K1};
}
export function environmentReport(rho:M,t:F){
  requireDensity(rho);const {V,U,K0,K1}=dampingDilation(t);
  const joint=product(product(V,rho),adjoint(V)),system=partialTrace(joint,2,2,'first'),environment=partialTrace(joint,2,2,'second');
  const h=quadratic(zero,f(BigInt(1),BigInt(2)));
  const mixed=[mScale(mAdd(K0,K1),h),mScale(mSub(K0,K1),h)];
  const map:QubitMap={kind:'operator-sum',terms:[K0,K1].map(operator=>({weight:one,operator}))};
  const mixedMap:QubitMap={kind:'operator-sum',terms:mixed.map(operator=>({weight:one,operator}))};
  const referenceOutput=applyWithReference(map,mScale([[O,Z,Z,O],[Z,Z,Z,Z],[Z,Z,Z,Z],[O,Z,Z,O]],quadratic(f(BigInt(1),BigInt(2)))));
  return {schemaVersion:1,tensorOrder:'system-environment',isometry:mWire(V),unitaryCompletion:mWire(U),
    isometryResidual:mWire(mSub(product(adjoint(V),V),identity(2))),unitaryResidual:mWire(mSub(product(adjoint(U),U),identity(4))),
    joint:mWire(joint),system:mWire(system),environment:mWire(environment),
    reducedEqualsKraus:mEqual(system,applyQubitMap(map,rho)),mixedKraus:mixed.map(mWire),
    mixedChannelEqualsOriginal:mEqual(qubitChoi(map),qubitChoi(mixedMap)),
    referenceOutput:mWire(referenceOutput),referenceMarginal:mWire(partialTrace(referenceOutput,2,2,'first'))};
}

export type QubitInstrument={outcomes:[WeightedOperator[],WeightedOperator[]]};
export type InstrumentFamily='projective-z'|'reset-z'|'damping-record'|'damping-mixed-record';
export function instrumentFamily(family:InstrumentFamily,t:F=f(BigInt(1),BigInt(2))):QubitInstrument {
  const term=(operator:M)=>[{weight:one,operator}];
  if(family==='projective-z')return {outcomes:[term(unit(0,0)),term(unit(1,1))]};
  if(family==='reset-z')return {outcomes:[term(unit(0,0)),term([[Z,O],[Z,Z]])]};
  const {K0,K1}=dampingOperators(t);
  if(family==='damping-record')return {outcomes:[term(K0),term(K1)]};
  if(family==='damping-mixed-record'){
    const h=quadratic(zero,f(BigInt(1),BigInt(2)));return {outcomes:[term(mScale(mAdd(K0,K1),h)),term(mScale(mSub(K0,K1),h))]};
  }
  throw new RangeError('Unknown instrument family');
}
export function instrumentReport(instrument:QubitInstrument,rho:M,correctZ=false){
  requireDensity(rho);if(instrument.outcomes.length!==2)throw new RangeError('Two recorded outcomes required');
  const maps:QubitMap[]=instrument.outcomes.map(terms=>({kind:'operator-sum',terms}));
  const complete:QubitMap={kind:'operator-sum',terms:instrument.outcomes.flat()};
  const J=qubitChoi(complete);const valid=maps.every(map=>principalMinorCertificate(qubitChoi(map)).positiveSemidefinite)&&mEqual(partialTrace(J,2,2,'first'),identity(2));
  if(!valid)throw new RangeError('Instrument must have CP branches and TP sum');
  const outputs=maps.map(map=>applyQubitMap(map,rho)),effects=maps.map(map=>qubitDual(map,identity(2)));
  const forgotten=mAdd(outputs[0],outputs[1]),cq=mAdd(tensor(unit(0,0),outputs[0]),tensor(unit(1,1),outputs[1]));
  const corrected=correctZ?mAdd(outputs[0],product(product(pauliX,outputs[1]),pauliX)):forgotten;
  return {schemaVersion:1,recordOrder:'record-system',correctZ,branches:outputs.map((output,i)=>{
    const p=trace(output);if(qSign(p.re)<0)throw new RangeError('Negative outcome probability');
    return {label:i,probability:circuitScalarWire(p),effect:mWire(effects[i]),unnormalized:mWire(output),
      conditional:qSign(p.re)===0?null:mWire(mScale(output,importedReciprocal(p.re)))};
  }),classicalQuantum:mWire(cq),forgotten:mWire(forgotten),feedforward:mWire(corrected)};
}
// Separate name prevents treating normalized conditional updates as linear channels.
const importedReciprocal=(p:Q)=>qDiv(quadratic(one),p);

export function deferredMeasurementReport(rho:M){
  requireDensity(rho);
  const prepare=tensor(rho,unit(0,0));
  const cxBC:M=[[O,Z,Z,Z],[Z,O,Z,Z],[Z,Z,Z,O],[Z,Z,O,Z]];
  const cxCB:M=[[O,Z,Z,Z],[Z,Z,Z,O],[Z,Z,O,Z],[Z,O,Z,Z]];
  const bellLike=product(product(cxBC,prepare),adjoint(cxBC));
  const coherent=product(product(cxCB,bellLike),adjoint(cxCB));
  const projectors=[tensor(identity(2),unit(0,0)),tensor(identity(2),unit(1,1))];
  const dephased=projectors.reduce((s,p)=>mAdd(s,product(product(p,coherent),p)),zeros(4));
  const classical=instrumentReport(instrumentFamily('projective-z'),rho,true);
  const swap:M=[[O,Z,Z,Z],[Z,Z,O,Z],[Z,O,Z,Z],[Z,Z,Z,O]];
  const recordSystem=product(product(swap,dephased),swap);
  return {schemaVersion:1,tensorOrder:'system-record',coherent:mWire(coherent),dephased:mWire(dephased),
    recordSystem:mWire(recordSystem),classicalCorrected:classical.feedforward,
    system:mWire(partialTrace(coherent,2,2,'first')),record:mWire(partialTrace(coherent,2,2,'second'))};
}
