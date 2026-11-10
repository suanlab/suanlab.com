import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimeUnregulatedWave} from '../src/lib/learning-real-time-unregulated-wave-check';
import {buildRealTimeChainExact} from '../src/lib/learning-real-time-phase-exact';
import {realTimeChainCoreWire} from '../src/lib/learning-real-time-phase-core-wire';
import {observeRealTimeChainWave} from '../src/lib/learning-real-time-wave-observations';
import {realTimePhaseExamples,realTimeChainDefault,type RealTimeChainInput} from '../src/lib/learning-real-time-phase-input';
import {CheckerRational as Q} from '../src/lib/learning-real-time-validation-rational';
const root='.runtime/learning/real-time-unregulated-wave-check';mkdirSync(root,{recursive:true});
function candidate(input:RealTimeChainInput) {
  const r=buildRealTimeChainExact(input);
  return{exact:realTimeChainCoreWire(r),wave:observeRealTimeChainWave({...r,regulator:null}).unregulated};
}
const cache=new Map<string,ReturnType<typeof candidate>>(),measurements:{id:string;milliseconds:number}[]=[];
for(const row of realTimePhaseExamples)if(row.input.mode==='chain') {
  const input=row.input;
  test('entire unregulated wave: '+row.id,()=>{
    const data=candidate(input);cache.set(row.id,data);const start=performance.now();
    const checked=verifyRealTimeUnregulatedWave(input,data.exact,data.wave);assert(checked,row.id);
    assert.equal(checked.scope,'unregulated-chain-wave-only');assert.equal(checked.fullResponseValidated,false);
    assert(Object.isFrozen(checked.observation));
    measurements.push({id:row.id,milliseconds:performance.now()-start});
    writeFileSync(root+`/cases-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
  });
}
type Segment=string|number;
function leaves(x:unknown,base:Segment[]=[]):{path:Segment[];value:unknown}[] {
  if(x===null||typeof x!=='object')return[{path:base,value:x}];
  return Object.entries(x).flatMap(([key,value])=>leaves(value,[...base,Array.isArray(x)?Number(key):key]));
}
function mutated<T>(value:T,segments:Segment[],next:unknown):T {
  const copy=structuredClone(value);let at=copy as Record<Segment,unknown>;
  for(const key of segments.slice(0,-1))at=at[key]as Record<Segment,unknown>;at[segments.at(-1)!]=next;return copy;
}
for(const id of ['harmonic-four','negative-mode-hand','caustic-compatible','zero-leading-pivot-invertible'])test('every primitive wave field mutation: '+id,()=>{
  const input=realTimePhaseExamples.find(x=>x.id===id)!.input,data=cache.get(id)!;let rejected=0;
  for(const leaf of leaves(data.wave)) {
    const next=typeof leaf.value==='string'&&/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(leaf.value)?Q.input(leaf.value).add(Q.of(BigInt(1))).text():
      typeof leaf.value==='number'?leaf.value+100:typeof leaf.value==='string'?'wrong-'+leaf.value:'not-null';
    assert.equal(verifyRealTimeUnregulatedWave(input,data.exact,mutated(data.wave,leaf.path,next)),null,leaf.path.join('.'));rejected++;
  }
  writeFileSync(root+`/mutations-${id}-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({id,rejected},null,2)+'\n');
});

test('wrong attribution,accessors,canonical endpoints andwhole-wave scope fail closed',()=>{
  const data=cache.get('harmonic-four')!;
  assert.equal(verifyRealTimeUnregulatedWave({...realTimeChainDefault,hbar:'1'},data.exact,data.wave),null);
  assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,{unregulated:data.wave,regulated:null,workingBits:8192,observationBits:512}),null);
  assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,mutated(data.wave,['amplitude','lower'],'2/2')),null);
  assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,mutated(data.wave,['unitPhase','real','upper'],'1/3')),null);
  assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,{...data.wave,extra:true}),null);
  let calls=0;const getter={...data.wave};Object.defineProperty(getter,'unitPhase',{get(){calls++;throw Error('getter');}});
  assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,getter),null);assert.equal(calls,0);
  assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,Object.create(data.wave)),null);
  const revoked=Proxy.revocable({},{});revoked.revoke();assert.equal(verifyRealTimeUnregulatedWave(realTimeChainDefault,data.exact,revoked.proxy),null);
});

test('Fresnel octants preserve allfour signature quadrants atzero action',()=>{
  for(const input of [
    {...realTimeChainDefault,intervals:2,time:'2',omegaSquared:'0',start:'0',end:'0',probe:1},
    {...realTimeChainDefault,intervals:2,time:'2',omegaSquared:'3',start:'0',end:'0',probe:1},
    {...realTimeChainDefault,intervals:3,time:'3',omegaSquared:'4',start:'0',end:'0',probe:1},
    {...realTimeChainDefault,intervals:4,time:'4',omegaSquared:'4',start:'0',end:'0',probe:1},
  ]){const data=candidate(input);assert(verifyRealTimeUnregulatedWave(input,data.exact,data.wave));}
});

test('unregulated checker runtime graph excludes producer math andfixtures',()=>{
  const queue=['src/lib/learning-real-time-unregulated-wave-check.ts'],visited=new Set<string>();
  const allowed=new Set(['learning-real-time-unregulated-wave-check','learning-real-time-observation-check','learning-real-time-checker-interval',
    'learning-real-time-exact-core-check','learning-real-time-validation-rational','learning-real-time-phase-protocol','learning-real-time-phase-input','learning-qubit-wire'].map(x=>'src/lib/'+x+'.ts'));
  while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);assert(allowed.has(file),file);
    const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)) {
      const spec=statement.moduleSpecifier.text;if(spec.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),spec+'.ts')));
    }
  }
  assert.equal(visited.size,allowed.size);
});
