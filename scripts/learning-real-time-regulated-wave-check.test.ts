import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimeRegulatedWave} from '../src/lib/learning-real-time-regulated-wave-check';
import {buildRealTimeChainExact} from '../src/lib/learning-real-time-phase-exact';
import {realTimeChainCoreWire} from '../src/lib/learning-real-time-phase-core-wire';
import {observeRealTimeChainWave,realTimeRawValueRequiresScale} from '../src/lib/learning-real-time-wave-observations';
import {realTimePhaseExamples,realTimeChainDefault,realTimePacketDefault,type RealTimeChainInput} from '../src/lib/learning-real-time-phase-input';
import {CheckerRational as Q} from '../src/lib/learning-real-time-validation-rational';
const root='.runtime/learning/real-time-regulated-wave-check';mkdirSync(root,{recursive:true});
function candidate(input:RealTimeChainInput) {
  const r=buildRealTimeChainExact(input);return{exact:realTimeChainCoreWire(r),wave:observeRealTimeChainWave(r).regulated};
}
const cache=new Map<string,{input:RealTimeChainInput;data:ReturnType<typeof candidate>}>();
const measurements:{id:string;milliseconds:number;requiresScale:boolean}[]=[];
for(const row of realTimePhaseExamples)if(row.input.mode==='chain') {
  const input={...row.input,damping:row.input.damping==='0'?'1/3':row.input.damping};
  test('entire Abel wave: '+row.id,()=>{
    const data=candidate(input);cache.set(row.id,{input,data});assert(data.wave);
    const start=performance.now(),checked=verifyRealTimeRegulatedWave(input,data.exact,data.wave);assert(checked,row.id);
    assert.equal(checked.scope,'abel-chain-wave-only');assert.equal(checked.fullResponseValidated,false);assert(Object.isFrozen(checked.observation));
    assert.equal(data.wave.rawValueRequiresScaledRepresentation,realTimeRawValueRequiresScale(data.wave.value));
    measurements.push({id:row.id,milliseconds:performance.now()-start,requiresScale:data.wave.rawValueRequiresScaledRepresentation});
    writeFileSync(root+`/cases-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
  });
}
type Segment=string|number;
function leaves(x:unknown,base:Segment[]=[]):{path:Segment[];value:unknown}[] {
  if(x===null||typeof x!=='object')return[{path:base,value:x}];
  return Object.entries(x).flatMap(([key,value])=>leaves(value,[...base,Array.isArray(x)?Number(key):key]));
}
function mutate<T>(value:T,segments:Segment[],next:unknown):T {
  const copy=structuredClone(value);let at=copy as Record<Segment,unknown>;
  for(const key of segments.slice(0,-1))at=at[key]as Record<Segment,unknown>;at[segments.at(-1)!]=next;return copy;
}
test('every resolved Abel primitive field mutation is rejected',()=>{
  const {input,data}=cache.get('negative-mode-hand')!;assert(data.wave);let rejected=0;
  for(const leaf of leaves(data.wave)) {
    const next=typeof leaf.value==='string'&&/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(leaf.value)?Q.input(leaf.value).add(Q.of(BigInt(1))).text():
      typeof leaf.value==='number'?leaf.value+100:typeof leaf.value==='string'?'wrong-'+leaf.value:!leaf.value;
    assert.equal(verifyRealTimeRegulatedWave(input,data.exact,mutate(data.wave,leaf.path,next)),null,leaf.path.join('.'));rejected++;
  }
  writeFileSync(root+`/mutations-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({rejected},null,2)+'\n');
});

test('maximum model rejects last phase,log,branch andcoordinated sign changes',()=>{
  const {input,data}=cache.get('maximum-mixed-coprime')!;assert(data.wave);
  for(const target of [['unitPhase','imaginary','lower'],['logMagnitude','upper'],['negativeExponential','lower']]as Segment[][]) {
    let value:unknown=data.wave;for(const key of target)value=(value as Record<Segment,unknown>)[key];
    assert.equal(verifyRealTimeRegulatedWave(input,data.exact,mutate(data.wave,target,Q.input(value as string).add(Q.of(BigInt(1))).text())),null);
  }
  const wrong=structuredClone(data.wave);
  for(const c of [wrong.unitPhase,wrong.scaledValue,wrong.value])for(const name of ['real','imaginary']as const){const old=c[name];c[name]={lower:Q.input(old.upper).neg().text(),upper:Q.input(old.lower).neg().text()};}
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,wrong),null,'determinant square-root sign shortcut');
});

test('raw underflow retains nonzero scaled andlog evidence andbounded grid uncertainty',()=>{
  for(const id of ['minimum-positive-factors','maximum-mixed-coprime']) {
    const {input,data}=cache.get(id)!;assert(data.wave);
    const zero=structuredClone(data.wave);zero.value.real={lower:'0',upper:'0'};zero.value.imaginary={lower:'0',upper:'0'};
    assert.equal(verifyRealTimeRegulatedWave(input,data.exact,zero),null,'raw must not become anexact zero');
    const wide=structuredClone(data.wave);wide.value.scalePowerOfTwo+=100;
    assert.equal(verifyRealTimeRegulatedWave(input,data.exact,wide),null,'raw floor is bounded');
    const toggled={...data.wave,rawValueRequiresScaledRepresentation:!data.wave.rawValueRequiresScaledRepresentation};
    assert.equal(verifyRealTimeRegulatedWave(input,data.exact,toggled),null);
  }
});

test('small prefactor boundary remains valid after decay itself is resolved',()=>{
  const input={...realTimeChainDefault,intervals:32,time:'16',mass:'1/100000',hbar:'1/999999999999',omegaSquared:'0',start:'15',end:'15',damping:'16',probe:16};
  const data=candidate(input);assert(data.wave);assert(data.wave.rawValueRequiresScaledRepresentation);
  assert(verifyRealTimeRegulatedWave(input,data.exact,data.wave));
});

test('deep Abel decay never becomes afalse exact zero orarbitrary raw box',()=>{
  const input={...realTimeChainDefault,intervals:32,time:'16',hbar:'1/999999999999',omegaSquared:'0',start:'-16',end:'16',damping:'16',probe:16};
  const data=candidate(input);assert(data.wave);assert(data.wave.rawValueRequiresScaledRepresentation);
  const decay=Q.input(data.wave.negativeExponential.lower);assert(decay.n>BigInt(1000000)*decay.d);
  assert(verifyRealTimeRegulatedWave(input,data.exact,data.wave));
  const zero=structuredClone(data.wave);zero.value.real={lower:'0',upper:'0'};zero.value.imaginary={lower:'0',upper:'0'};
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,zero),null);
  const wide=structuredClone(data.wave);wide.value.scalePowerOfTwo+=100;
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,wide),null);
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,{...data.wave,rawValueRequiresScaledRepresentation:false}),null);
});

test('display annotation includes final observation rounding',()=>{
  const S=BigInt(1)<<BigInt(512);
  assert(realTimeRawValueRequiresScale({real:{lower:`-1/${S}`,upper:'1'},imaginary:{lower:'0',upper:'0'}}));
  assert(!realTimeRawValueRequiresScale({real:{lower:`1/${S}`,upper:'1'},imaginary:{lower:'0',upper:'0'}}));
});

test('absent regulators,ownership,attribution andpartial scope fail closed',()=>{
  const r=buildRealTimeChainExact(realTimeChainDefault),exact=realTimeChainCoreWire(r);
  assert(verifyRealTimeRegulatedWave(realTimeChainDefault,exact,null));
  assert.equal(verifyRealTimeRegulatedWave(realTimeChainDefault,exact,{}),null);
  const {input,data}=cache.get('negative-mode-hand')!;assert(data.wave);
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,null),null);
  assert.equal(verifyRealTimeRegulatedWave({...input,mass:'2'},data.exact,data.wave),null);
  assert.equal(verifyRealTimeRegulatedWave(realTimePacketDefault,data.exact,data.wave),null);
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,{regulated:data.wave,unregulated:null}),null);
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,mutate(data.wave,['negativeExponential','lower'],'2/2')),null);
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,mutate(data.wave,['value','real','upper'],'1/3')),null);
  let calls=0;const getter={...data.wave};Object.defineProperty(getter,'unitPhase',{get(){calls++;throw Error('getter');}});
  assert.equal(verifyRealTimeRegulatedWave(input,data.exact,getter),null);assert.equal(calls,0);
  const revoked=Proxy.revocable({},{});revoked.revoke();assert.equal(verifyRealTimeRegulatedWave(input,data.exact,revoked.proxy),null);
});

test('Abel checker runtime graph excludes producer arithmetic andfixtures',()=>{
  const queue=['src/lib/learning-real-time-regulated-wave-check.ts'],visited=new Set<string>();
  const allowed=new Set(['learning-real-time-regulated-wave-check','learning-real-time-observation-check','learning-real-time-checker-interval',
    'learning-real-time-exact-core-check','learning-real-time-validation-rational','learning-real-time-phase-protocol','learning-real-time-phase-input','learning-qubit-wire'].map(x=>'src/lib/'+x+'.ts'));
  while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);assert(allowed.has(file),file);
    const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)) {
      const spec=statement.moduleSpecifier.text;if(spec.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),spec+'.ts')));
    }
  }
  assert.equal(visited.size,allowed.size);
});
