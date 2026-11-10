import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { f,zero,one,quadratic,complex,Z,O,real,qDiv,qSign,qMul,qEqual,minus,times,
  mWire,mEqual,product,trace,identity,partialTrace,principalMinorCertificate,densityFromVector,adjoint,
  type M,type F,type C } from '../src/lib/learning-qubit-channel-algebra';
import { channelReport,channelFamily,environmentReport,instrumentFamily,instrumentReport,deferredMeasurementReport,
  qubitChoi,qubitMapFromChoi,applyQubitMap,qubitDual,composeQubitMaps,applyWithReference,pauliX,
  type QubitMap,type WeightedOperator,type InstrumentFamily,type ChannelFamily } from '../src/lib/learning-qubit-channel';
import { verifyQubitChannelReport,verifyEnvironmentReport,verifyInstrumentReport,verifyDeferredMeasurementReport,
  channelValidatorPSD } from '../src/lib/learning-qubit-channel-validator';

type WireScalar={real:string[];imaginary:string[]};type WireMatrix=WireScalar[][];
type EncodedTerm={weight:string;operator:WireMatrix};
type EncodedMap={kind:'transpose'|'operator-sum'|'matrix-units';terms?:EncodedTerm[];images?:WireMatrix[]};
type Case={name:string;state:WireMatrix;expected:unknown};
type Reference={channelCases:(Case&{map:EncodedMap})[];environmentCases:(Case&{parameter:string})[];
  instrumentCases:(Case&{family:InstrumentFamily|'coarse-pauli-record';parameter:string;correctZ:boolean;outcomes:EncodedTerm[][]})[];deferredCases:Case[]};
const references=JSON.parse(readFileSync('scripts/fixtures/qubit-channel-independent.json','utf8')) as Reference;
const fraction=(s:string):F=>{const [n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const scalar=(v:WireScalar):C=>complex([fraction(v.real[0]),fraction(v.real[1])],[fraction(v.imaginary[0]),fraction(v.imaginary[1])]);
const matrix=(a:WireMatrix):M=>a.map(row=>row.map(scalar));
const terms=(encoded:EncodedTerm[]):WeightedOperator[]=>encoded.map(t=>({weight:fraction(t.weight),operator:matrix(t.operator)}));
const map=(description:EncodedMap):QubitMap=>description.kind==='transpose'?{kind:'transpose'}:description.kind==='operator-sum'?{kind:'operator-sum',terms:terms(description.terms!)}:{kind:'matrix-units',images:description.images!.map(matrix) as [M,M,M,M]};

for(const example of references.channelCases)test(`channel symbolic reference and independent validator: ${example.name}`,()=>{
  const model=map(example.map),rho=matrix(example.state),actual=channelReport(model,rho);
  assert.deepEqual(actual,example.expected);
  assert.deepEqual(verifyQubitChannelReport(model,rho,actual),{valid:true,errors:[]});
  // A well-shaped replacement from the exact same scalar type still must fail.
  const altered=structuredClone(actual);altered.output[0][0].real[0]=altered.output[0][0].real[0]==='0'?'1':'0';
  assert.equal(verifyQubitChannelReport(model,rho,altered).valid,false);
});
for(const example of references.environmentCases)test(`environment symbolic reference and independent contraction: ${example.name}`,()=>{
  const rho=matrix(example.state),t=fraction(example.parameter),actual=environmentReport(rho,t);
  assert.deepEqual(actual,example.expected);assert.equal(verifyEnvironmentReport(rho,t,actual).valid,true);
  const altered=structuredClone(actual);altered.system[0][0].real[0]='123';
  assert.equal(verifyEnvironmentReport(rho,t,altered).valid,false);
});
for(const example of references.instrumentCases)test(`instrument symbolic reference, zero branches and feedback: ${example.name}`,()=>{
  const rho=matrix(example.state),t=fraction(example.parameter),model={outcomes:example.outcomes.map(terms) as [WeightedOperator[],WeightedOperator[]]},actual=instrumentReport(model,rho,example.correctZ);
  if(example.family!=='coarse-pauli-record')assert.deepEqual(model,instrumentFamily(example.family,t));
  assert.deepEqual(actual,example.expected);assert.equal(verifyInstrumentReport(model,rho,example.correctZ,actual).valid,true);
  const altered=structuredClone(actual);altered.branches[0].probability.real[0]='99';
  assert.equal(verifyInstrumentReport(model,rho,example.correctZ,altered).valid,false);
});
for(const example of references.deferredCases)test(`deferred measurement and retained coherence: ${example.name}`,()=>{
  const rho=matrix(example.state),actual=deferredMeasurementReport(rho);
  assert.deepEqual(actual,example.expected);assert.equal(verifyDeferredMeasurementReport(rho,actual).valid,true);
  const altered=structuredClone(actual);altered.record[0][0].real[0]='777';
  assert.equal(verifyDeferredMeasurementReport(rho,altered).valid,false);
});

test('independent reference inventory is substantive',()=>{
  assert.equal(references.channelCases.length,216);assert.equal(references.environmentCases.length,45);
  assert.equal(references.instrumentCases.length,234);assert.equal(references.deferredCases.length,9);
});
test('public channel family constructors match the independent parametrizations',()=>{
  for(const example of references.channelCases.filter(v=>v.name.endsWith('/zero'))){
    const name=example.name.slice(0,-5),parameterized=/^(damping|dephasing|depolarizing)-(.+)$/.exec(name);
    if(parameterized)assert.deepEqual(channelFamily(parameterized[1] as ChannelFamily,fraction(parameterized[2])),map(example.map),name);
    else if(['identity','postselect-zero','transpose'].includes(name))assert.deepEqual(channelFamily(name as ChannelFamily),map(example.map),name);
  }
  assert.throws(()=>channelFamily('unknown' as ChannelFamily),RangeError);
});
test('quadratic division and exact signs survive nearly cancelling coefficients',()=>{
  const q=[one,one] as const;assert.ok(qEqual(qMul(qDiv([one,zero],q),q),[one,zero]));
  const scale=BigInt(10)**BigInt(200);
  assert.equal(qSign([f(-scale-BigInt(1),scale),zero]),-1);
  assert.equal(qSign([f(BigInt(-1)),one]),1);assert.equal(qSign([one,f(BigInt(-1))]),-1);
  let p=BigInt(1),r=BigInt(1);
  for(let k=0;k<200;k++)[p,r]=[p+BigInt(2)*r,p+r];
  assert.equal(p*p-BigInt(2)*r*r,BigInt(-1));
  assert.equal(qSign([f(p),f(-r)]),-1);assert.equal(qSign([f(-p),f(r)]),1);
  assert.throws(()=>qDiv([one,zero],[zero,zero]),RangeError);
});
test('normalization with an irrational squared norm matches the symbolic state',()=>{
  const vector=[complex([one,one]),complex(quadratic(zero),quadratic(one))];
  const example=references.channelCases.find(v=>v.name==='identity/irrational-norm')!;
  assert.deepEqual(mWire(densityFromVector(vector)),example.state);
});
test('calculation preserves model inputs and its exact response survives JSON transport',()=>{
  const rho=densityFromVector([O,complex(quadratic(zero),quadratic(one))]),model=channelFamily('damping');
  const encode=(x:unknown)=>JSON.stringify(x,(_key,v)=>typeof v==='bigint'?String(v):v);
  const before=encode({rho,model});const response=channelReport(model,rho);
  environmentReport(rho,f(BigInt(1),BigInt(2)));instrumentReport(instrumentFamily('damping-record'),rho);deferredMeasurementReport(rho);
  assert.equal(encode({rho,model}),before);
  assert.deepEqual(JSON.parse(JSON.stringify(response)),response);
});
test('PSD needs all principal minors, including a negative diagonal hidden by zero leading minors',()=>{
  const a:M=[[Z,Z],[Z,real(-1)]];
  assert.equal(principalMinorCertificate(a).positiveSemidefinite,false);assert.equal(channelValidatorPSD(a),false);
  const zeroPivotOffDiagonal:M=[[Z,O],[O,Z]];
  assert.equal(channelValidatorPSD(zeroPivotOffDiagonal),false);
  assert.equal(principalMinorCertificate([[O,complex(quadratic(zero),quadratic(one))],[Z,O]]).positiveSemidefinite,false);
});
test('transpose is positive on the chosen local state and not completely positive',()=>{
  const rho=densityFromVector([O,complex(quadratic(zero),quadratic(one))]);
  const result=channelReport({kind:'transpose'},rho);
  assert.equal(result.outputIsDensity,true);assert.equal(result.properties.tracePreserving,true);
  assert.equal(result.properties.completelyPositive,false);assert.equal(result.bellOutputIsDensity,false);
});
test('Schrodinger/Heisenberg pairing uses the transpose of the Choi partial trace',()=>{
  const rho=densityFromVector([complex(quadratic(one),quadratic(one)),real(2)]);
  const model:QubitMap={kind:'operator-sum',terms:[{weight:one,operator:[[O,complex(quadratic(zero),quadratic(one))],[real(1,2),real(1,3)]]}]};
  const E:M=[[real(2),complex(quadratic(zero),quadratic(one))],[complex(quadratic(zero),quadratic(f(BigInt(-1)))),real(3)]];
  assert.deepEqual(trace(product(E,applyQubitMap(model,rho))),trace(product(qubitDual(model,E),rho)));
  assert.ok(!mEqual(partialTrace(qubitChoi(model),2,2,'first'),qubitDual(model,identity(2))));
});
test('general non-HP maps require the Hilbert-Schmidt conjugation in the adjoint',()=>{
  const J:M=[[O,complex(quadratic(zero),quadratic(one)),complex(quadratic(zero),quadratic(one)),Z],[Z,O,Z,Z],[Z,Z,O,Z],[Z,Z,Z,O]];
  const model=qubitMapFromChoi(J),A:M=[[O,real(2)],[complex(quadratic(zero),quadratic(one)),Z]],B:M=[[real(2),real(3)],[Z,complex(quadratic(zero),quadratic(one))]];
  assert.deepEqual(trace(product(adjoint(A),applyQubitMap(model,B))),trace(product(adjoint(qubitDual(model,A)),B)));
});
test('composition order, Choi reconstruction and both tensor marginals',()=>{
  const damping=channelFamily('damping'),flip:QubitMap={kind:'operator-sum',terms:[{weight:one,operator:pauliX}]};
  const rho:M=[[O,Z],[Z,Z]];
  assert.ok(!mEqual(applyQubitMap(composeQubitMaps(damping,flip),rho),applyQubitMap(composeQubitMaps(flip,damping),rho)));
  assert.ok(mEqual(applyQubitMap(composeQubitMaps(damping,flip),rho),[[Z,Z],[Z,O]]));
  assert.ok(mEqual(applyQubitMap(composeQubitMaps(flip,damping),rho),[[real(16,25),Z],[Z,real(9,25)]]));
  const J=qubitChoi(damping);assert.ok(mEqual(J,qubitChoi(qubitMapFromChoi(J))));
  const r=densityFromVector([O,O]);assert.ok(mEqual(partialTrace(applyWithReference(damping,productTensor(r,r)),2,2,'first'),r));
});
function productTensor(a:M,b:M):M {return Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>times(a[Math.floor(i/2)][Math.floor(j/2)],b[i%2][j%2])));}
test('zero conditional states stay unavailable; a POVM does not determine disturbance',()=>{
  const rho:M=[[O,Z],[Z,Z]],p=instrumentReport(instrumentFamily('projective-z'),rho);
  assert.equal(p.branches[1].conditional,null);
  const plus=densityFromVector([O,O]);const a=instrumentReport(instrumentFamily('projective-z'),plus),b=instrumentReport(instrumentFamily('reset-z'),plus);
  assert.deepEqual(a.branches.map(x=>x.effect),b.branches.map(x=>x.effect));assert.notDeepEqual(a.forgotten,b.forgotten);
});
test('a complex entangled reference uses the reference-system order',()=>{
  const half=real(1,2),negativeI=complex(quadratic(zero),quadratic(f(BigInt(-1),BigInt(2))));
  const rho:M=[[Z,Z,Z,Z],[Z,half,negativeI,Z],[Z,minus(Z,negativeI),half,Z],[Z,Z,Z,Z]];
  const out=applyWithReference(channelFamily('damping'),rho);
  const coherence=complex(quadratic(zero),quadratic(f(BigInt(-3),BigInt(10))));
  assert.ok(mEqual(out,[[real(8,25),Z,Z,Z],[Z,real(9,50),coherence,Z],[Z,minus(Z,coherence),half,Z],[Z,Z,Z,Z]]));
  assert.ok(mEqual(partialTrace(out,2,2,'first'),[[half,Z],[Z,half]]));
});
test('coherent records are observable before final dephasing',()=>{
  const rho=densityFromVector([O,O]),r=deferredMeasurementReport(rho);
  assert.notDeepEqual(r.coherent,r.dephased);assert.deepEqual(r.record,mWire(rho));
});
test('strict validation rejects changed flags, missing/extra fields and another valid preparation',()=>{
  const rho:M=[[O,Z],[Z,Z]],model=channelFamily('identity'),report=channelReport(model,rho);
  for(const altered of [{...report,properties:{...report.properties,quantumChannel:false}},{...report,extra:true},{...report,choi:[]},null,channelReport(channelFamily('damping'),rho)])assert.equal(verifyQubitChannelReport(model,rho,altered).valid,false);
  const reordered=Object.fromEntries(Object.entries(report).reverse());assert.equal(verifyQubitChannelReport(model,rho,reordered).valid,true);
  const extraArrayProperty=structuredClone(report);Object.assign(extraArrayProperty.output,{extra:true});
  assert.equal(verifyQubitChannelReport(model,rho,extraArrayProperty).valid,false);
  for(const key of Object.keys(report)){
    const incomplete={...report} as Record<string,unknown>;delete incomplete[key];
    assert.equal(verifyQubitChannelReport(model,rho,incomplete).valid,false,key);
  }
});
test('reject invalid states, parameters, dimensions, channel families and instruments',()=>{
  const rho:M=[[O,Z],[Z,Z]];
  for(const p of [f(BigInt(-1)),f(BigInt(2))])assert.throws(()=>channelFamily('damping',p),RangeError);
  assert.throws(()=>channelReport(channelFamily('identity'),[[O,Z],[Z,O]]),RangeError);
  assert.throws(()=>channelReport(channelFamily('identity'),[[Z,O],[O,O]]),RangeError);
  assert.throws(()=>densityFromVector([Z,Z]),RangeError);
  assert.throws(()=>partialTrace(identity(4),1,2,'first'),RangeError);
  assert.throws(()=>partialTrace(identity(4),1.5,2,'first'),RangeError);
  assert.throws(()=>instrumentReport({outcomes:[[{weight:one,operator:identity(2)}],[{weight:one,operator:identity(2)}]]},rho),RangeError);
  assert.throws(()=>instrumentReport({outcomes:[[{weight:f(BigInt(-1)),operator:identity(2)}],[{weight:f(BigInt(2)),operator:identity(2)}]]},rho),RangeError);
  assert.equal(verifyInstrumentReport({outcomes:[[],[]]},rho,false,{}).valid,false);
});
