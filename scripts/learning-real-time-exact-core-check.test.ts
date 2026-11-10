import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimeExactCore} from '../src/lib/learning-real-time-exact-core-check';
import {buildRealTimeChainExact,buildRealTimePacketExact} from '../src/lib/learning-real-time-phase-exact';
import {realTimeChainCoreWire,realTimePacketCoreWire} from '../src/lib/learning-real-time-phase-core-wire';
import {realTimePhaseExamples,realTimeChainDefault,realTimePacketDefault,type RealTimePhaseInput} from '../src/lib/learning-real-time-phase-input';
import {Rational as Q} from '../src/lib/learning-graph-gauge-validation-math';
const root='.runtime/learning/real-time-exact-check';mkdirSync(root,{recursive:true});
const fixture=JSON.parse(readFileSync('scripts/fixtures/real-time-phase-exact-references.json','utf8')) as {cases:{id:string;input:RealTimePhaseInput;expected:{map:string[][];inertia:number[]}}[]};
const values=new Map<string,unknown>(),measurements:{id:string;milliseconds:number}[]=[];
function candidate(input:RealTimePhaseInput):unknown {
  if(input.mode==='chain')return realTimeChainCoreWire(buildRealTimeChainExact(input));
  if(input.mode==='packet')return realTimePacketCoreWire(buildRealTimePacketExact(input));
  throw Error('Fold has no finite exact core');
}
for(const item of fixture.cases)test('independent exact-core certificate: '+item.id,()=>{
  const wire=candidate(item.input);values.set(item.id,wire);const start=performance.now(),verified=verifyRealTimeExactCore(item.input,wire);
  assert(verified,item.id);assert.equal(verified.scope,'finite-exact-core-only');assert.equal(verified.fullResponseValidated,false);assert(Object.isFrozen(verified.wire));
  assert.deepEqual(verified.map.map(row=>row.map(x=>x.text())),item.expected.map);
  assert.deepEqual([verified.inertia.positive,verified.inertia.negative,verified.inertia.zero],item.expected.inertia);
  if(verified.inverseEntry)for(let i=0;i<verified.input.intervals-1;i++)assert(verified.inverseEntry(i,i).d>BigInt(0));
  measurements.push({id:item.id,milliseconds:performance.now()-start});
  writeFileSync(root+`/verification-times-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});
type Segment=string|number;
function leaves(value:unknown,base:Segment[]=[]):{path:Segment[];value:unknown}[] {
  if(value===null||typeof value!=='object')return[{path:base,value}];
  return Object.entries(value).flatMap(([key,x])=>leaves(x,[...base,Array.isArray(value)?Number(key):key]));
}
function change(value:unknown){
  if(typeof value==='string')return /^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(value)?Q.read(value).add(new Q(BigInt(1))).text():'wrong-'+value;
  if(typeof value==='number')return value+1;if(typeof value==='boolean')return!value;return'not-null';
}
function mutate(value:unknown,segments:Segment[],next:unknown):unknown {
  const out=structuredClone(value) as Record<string,unknown>;let at=out as Record<Segment,unknown>;
  for(const key of segments.slice(0,-1))at=at[key] as Record<Segment,unknown>;at[segments.at(-1)!]=next;return out;
}
for(const id of ['negative-mode-hand','zero-leading-pivot-invertible','caustic-compatible','caustic-incompatible','regulated-free-branch','fixed-width-free','scaled-width-free','harmonic-parity'])test('reject every primitive exact-core mutation: '+id,()=>{
  const input=realTimePhaseExamples.find(row=>row.id===id)!.input,wire=values.get(id)??candidate(input);let rejected=0;
  for(const leaf of leaves(wire)){assert.equal(verifyRealTimeExactCore(input,mutate(wire,leaf.path,change(leaf.value))),null,id+' '+leaf.path.join('.'));rejected++;}
  assert(rejected>20);
  writeFileSync(root+`/mutations-${id}-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({id,rejected},null,2)+'\n');
});
test('maximum32interval mixed fractions reject late-field and coordinated certificate changes',()=>{
  const input=realTimePhaseExamples.find(row=>row.id==='maximum-mixed-coprime')!.input,wire=values.get('maximum-mixed-coprime')??candidate(input);
  const targets=[['refinements',3,'map',1,0],['stationary',30],['determinant'],['regulator','mean',30,'imaginary'],['regulator','exponent','real'],['regulator','continuants',31,'real'],['regulator','distancePowers',30,'real']] as Segment[][];
  for(const target of targets){let at:unknown=wire;for(const key of target)at=(at as Record<Segment,unknown>)[key];assert.equal(verifyRealTimeExactCore(input,mutate(wire,target,change(at))),null,target.join('.'));}
  const wrongInput={...realTimeChainDefault,mass:'2'},other=candidate(wrongInput) as {input:unknown};const forged=structuredClone(other);forged.input={...realTimeChainDefault};assert.equal(verifyRealTimeExactCore(realTimeChainDefault,forged),null);
});
test('canonical fields,ownership andscope remain distinct from whole response validation',()=>{
  const input=realTimeChainDefault,wire=candidate(input);
  assert.equal(verifyRealTimeExactCore(input,{kind:'chain',input,exact:wire,wave:{forged:true}}),null);
  assert.equal(verifyRealTimeExactCore({mode:'fold',control:'0',hbar:'1'},wire),null);
  for(const s of ['2/2','-0','+1','01','1/0','1e0','1/01'])assert.equal(verifyRealTimeExactCore(input,mutate(wire,['inverse','scale'],s)),null);
  assert.equal(verifyRealTimeExactCore(input,{...(wire as object),extra:true}),null);
  assert.equal(verifyRealTimeExactCore(input,{...(wire as object),[Symbol('hidden')]:true}),null);
  let calls=0;const getter={...(wire as object)};Object.defineProperty(getter,'determinant',{get(){calls++;throw Error('getter');},enumerable:true});assert.equal(verifyRealTimeExactCore(input,getter),null);assert.equal(calls,0);
  const cycle={...(wire as object),extra:null as unknown};cycle.extra=cycle;assert.equal(verifyRealTimeExactCore(input,cycle),null);
  assert.equal(verifyRealTimeExactCore(input,Object.create(wire as object)),null);
  assert.equal(verifyRealTimeExactCore(input,mutate(wire,['dimension'],NaN)),null);
  const zero=realTimePhaseExamples.find(x=>x.id==='negative-mode-hand')!;
  assert.equal(verifyRealTimeExactCore(zero.input,mutate(values.get(zero.id)??candidate(zero.input),['inertia','positive'],-0)),null);
});
test('exact checker runtime dependency graph excludes producer engines andfixtures',()=>{
  const visited=new Set<string>(),queue=['src/lib/learning-real-time-exact-core-check.ts'];
  const allowed=new Set(['src/lib/learning-real-time-exact-core-check.ts','src/lib/learning-real-time-validation-rational.ts','src/lib/learning-real-time-phase-input.ts','src/lib/learning-real-time-phase-protocol.ts','src/lib/learning-qubit-wire.ts']);
  while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);
    assert(allowed.has(file),'unexpected runtime dependency: '+file);
    const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)){
      const specifier=statement.moduleSpecifier.text;if(specifier.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),specifier+'.ts')));
    }
  }
  assert(visited.has('src/lib/learning-real-time-validation-rational.ts'));
});

test('every supported interval count checks signed curvature,caustics andboth packet width families',()=>{
  let chains=0,packets=0;
  for(let n=2;n<=32;n++) {
    const delta=n<=16?1:0.5,time=n<=16?String(n):`${n}/2`;
    for(const v of [0,1,2,3,4,-1]) {
      const input={...realTimeChainDefault,intervals:n,time,omegaSquared:String(v/(delta*delta)),start:'1/3',end:n%2?'1/3':'-1/3',probe:n};
      assert(verifyRealTimeExactCore(input,candidate(input)),`n${n},v${v}`);chains++;
    }
    for(const widthMode of ['fixed-variance','hbar-scaled-variance']as const) {
      const input={...realTimePacketDefault,intervals:n,time,mass:'2/3',hbar:'1/3',omegaSquared:'-3/2',position:'1/3',momentum:'-2/5',chirp:'-7/5',width:'5/7',widthMode};
      assert(verifyRealTimeExactCore(input,candidate(input)),`packet n${n},${widthMode}`);packets++;
    }
  }
  assert.equal(chains,186);assert.equal(packets,62);
  writeFileSync(root+`/interval-grid-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({minimumIntervals:2,maximumIntervals:32,chainCases:chains,packetCases:packets},null,2)+'\n');
});
