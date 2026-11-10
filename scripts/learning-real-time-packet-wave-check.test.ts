import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimePacketWave} from '../src/lib/learning-real-time-packet-wave-check';
import {buildRealTimePacketExact} from '../src/lib/learning-real-time-phase-exact';
import {realTimePacketCoreWire} from '../src/lib/learning-real-time-phase-core-wire';
import {observeRealTimePacketWave} from '../src/lib/learning-real-time-wave-observations';
import {realTimePhaseExamples,realTimePacketDefault,realTimeChainDefault,type RealTimePacketInput} from '../src/lib/learning-real-time-phase-input';
import {CheckerRational as Q} from '../src/lib/learning-real-time-validation-rational';
const root='.runtime/learning/real-time-packet-wave-check';mkdirSync(root,{recursive:true});
const examples:{id:string;input:RealTimePacketInput}[]=realTimePhaseExamples.filter(x=>x.input.mode==='packet')as{id:string;input:RealTimePacketInput}[];
examples.push(
  {id:'negative-B-negative-alpha-real',input:{...realTimePacketDefault,intervals:2,time:'2',omegaSquared:'3'}},
  {id:'negative-B-positive-alpha-real',input:{...realTimePacketDefault,intervals:2,time:'2',omegaSquared:'3',chirp:'-1'}},
  {id:'positive-B-negative-alpha-real',input:{...realTimePacketDefault,chirp:'-2'}},
  {id:'near-parity-left',input:{...realTimePacketDefault,intervals:32,time:'16',mass:'1/999999999989',hbar:'1/999999999983',omegaSquared:'799999999908/99999999989'}},
  {id:'near-parity-right',input:{...realTimePacketDefault,intervals:32,time:'16',mass:'1/999999999989',hbar:'1/999999999983',omegaSquared:'799999999916/99999999989'}},
  {id:'mixed-coprime-packet',input:{...realTimePacketDefault,intervals:32,time:'999999999983/999999999989',mass:'999999999979/999999999961',hbar:'999999999953/999999999929',omegaSquared:'-999999999893/999999999883',position:'-999999999757/999999999749',momentum:'999999999727/999999999719',width:'999999999797/999999999761',chirp:'-3/7'}},
);
function candidate(input:RealTimePacketInput) {
  const r=buildRealTimePacketExact(input);return{exact:realTimePacketCoreWire(r),wave:observeRealTimePacketWave(r)};
}
const cache=new Map<string,ReturnType<typeof candidate>>(),measurements:{id:string;input:RealTimePacketInput;milliseconds:number;samples:number;BSign:number;alphaRealSign:number}[]=[];
for(const row of examples)test('full packet wave andall65samples: '+row.id,()=>{
  const data=candidate(row.input);cache.set(row.id,data);const start=performance.now(),checked=verifyRealTimePacketWave(row.input,data.exact,data.wave);assert(checked,row.id);
  assert.equal(checked.scope,'packet-wave-only');assert.equal(checked.fullResponseValidated,false);assert.equal(checked.samplesVerified,65);assert(Object.isFrozen(checked.observation));
  const B=Q.input(data.exact.map[0][1]),alphaReal=Q.input(data.exact.map[0][0]).add(B.mul(Q.input(row.input.chirp)));
  measurements.push({id:row.id,input:row.input,milliseconds:performance.now()-start,samples:65,BSign:B.n<BigInt(0)?-1:B.n>BigInt(0)?1:0,alphaRealSign:alphaReal.n<BigInt(0)?-1:alphaReal.n>BigInt(0)?1:0});
  writeFileSync(root+`/cases-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});
type Segment=string|number;
function leaves(x:unknown,base:Segment[]=[]):{path:Segment[];value:unknown}[] {
  if(x===null||typeof x!=='object')return[{path:base,value:x}];
  return Object.entries(x).flatMap(([key,value])=>leaves(value,[...base,Array.isArray(x)?Number(key):key]));
}
function mutate<T>(value:T,segments:Segment[],next:unknown):T {
  const copy=structuredClone(value);let at=copy as Record<Segment,unknown>;
  for(const key of segments.slice(0,-1))at=at[key]as Record<Segment,unknown>;at[segments.at(-1)!]=next;return copy;
}
test('alltop-level primitive fields andeveryfield atfirst,center,last samples reject mutation',()=>{
  const data=cache.get('fixed-width-free')!;let rejected=0;
  const targets=leaves(data.wave).filter(x=>x.path[0]!=='samples'||[0,32,64].includes(x.path[1]as number));
  for(const leaf of targets) {
    const next=typeof leaf.value==='string'&&/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(leaf.value)?Q.input(leaf.value).add(Q.of(BigInt(1))).text():
      typeof leaf.value==='number'?leaf.value+100:typeof leaf.value==='string'?'wrong-'+leaf.value:!leaf.value;
    assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,mutate(data.wave,leaf.path,next)),null,leaf.path.join('.'));rejected++;
  }
  writeFileSync(root+`/primitive-mutations-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({rejected},null,2)+'\n');
});
test('every sample position is actually checked,including middle entries',()=>{
  const data=cache.get('fixed-width-free')!;
  for(let i=0;i<65;i++) {
    const lower=Q.input(data.wave.samples[i].position.lower).add(Q.of(BigInt(1))).text();
    assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,mutate(data.wave,['samples',i,'position','lower'],lower)),null,String(i));
  }
});
test('density uses the normalized whole-line Gaussian,notfinite-window renormalization',()=>{
  const data=cache.get('fixed-width-free')!,wrong=structuredClone(data.wave),S=BigInt(1)<<BigInt(512),factor=Q.of(BigInt(1000000),BigInt(999937));
  const floor=(n:bigint,d:bigint)=>n>=BigInt(0)?n/d:-((-n+d-BigInt(1))/d);
  for(const sample of wrong.samples)for(const side of ['lower','upper']as const) {
    const x=Q.input(sample.density[side]).mul(factor),n=side==='lower'?floor(x.n*S,x.d):-floor(-x.n*S,x.d);
    sample.density[side]=Q.of(n,S).text();
  }
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,wrong),null);
});
test('sample reordering,metadata,attribution,ownership andpartial scope fail closed',()=>{
  const data=cache.get('fixed-width-free')!;
  const swapped=structuredClone(data.wave);[swapped.samples[3],swapped.samples[61]]=[swapped.samples[61],swapped.samples[3]];
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,swapped),null);
  const short=structuredClone(data.wave);short.samples.pop();assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,short),null);
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,mutate(data.wave,['samples',32,'standardDeviations'],-0)),null);
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,mutate(data.wave,['samples',64,'density','upper'],'1/3')),null);
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,{kind:'packet',input:realTimePacketDefault,exact:data.exact,wave:data.wave}),null);
  assert.equal(verifyRealTimePacketWave(realTimeChainDefault,data.exact,data.wave),null);
  assert.equal(verifyRealTimePacketWave({...realTimePacketDefault,hbar:'1'},data.exact,data.wave),null);
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,mutate(data.wave,['normalization','lower'],'2/2')),null);
  let calls=0;const getter={...data.wave};Object.defineProperty(getter,'samples',{get(){calls++;throw Error('getter');}});
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,getter),null);assert.equal(calls,0);
  const revoked=Proxy.revocable({},{});revoked.revoke();assert.equal(verifyRealTimePacketWave(realTimePacketDefault,data.exact,revoked.proxy),null);
});
test('near-parity andmixed models reject branch signs andlast-sample corruption',()=>{
  for(const id of ['near-parity-left','near-parity-right','mixed-coprime-packet']) {
    const row=examples.find(x=>x.id===id)!,data=cache.get(id)!;
    const wrong=structuredClone(data.wave);
    for(const phase of [wrong.metaplecticPhase,wrong.centerPhase,wrong.centerValue,...wrong.samples.flatMap(x=>[x.unitPhase,x.value])])
      for(const key of ['real','imaginary']as const){const x=phase[key];phase[key]={lower:Q.input(x.upper).neg().text(),upper:Q.input(x.lower).neg().text()};}
    assert.equal(verifyRealTimePacketWave(row.input,data.exact,wrong),null,id+' sign');
    const late=mutate(data.wave,['samples',64,'value','imaginary','upper'],Q.input(data.wave.samples[64].value.imaginary.upper).add(Q.of(BigInt(1))).text());
    assert.equal(verifyRealTimePacketWave(row.input,data.exact,late),null,id+' late');
  }
});
test('fixed standardized-factor cache cannot certify other physical inputs orbad exact witnesses',()=>{
  const a=cache.get('fixed-width-free')!,b=cache.get('scaled-width-free')!;
  assert(verifyRealTimePacketWave(realTimePacketDefault,a.exact,a.wave));
  const input=examples.find(x=>x.id==='scaled-width-free')!.input;
  assert.equal(verifyRealTimePacketWave(input,b.exact,a.wave),null);
  assert(verifyRealTimePacketWave(input,b.exact,b.wave));
  const forged=mutate(a.exact,['finalCovariance',0,0],'1');
  assert.equal(verifyRealTimePacketWave(realTimePacketDefault,forged,a.wave),null);
});
test('negative B with exactly zero real alpha uses the correct focus branch',()=>{
  const input={...realTimePacketDefault,intervals:2,time:'2',omegaSquared:'3',chirp:'-1/2'},data=candidate(input);
  const B=Q.input(data.exact.map[0][1]),alpha=Q.input(data.exact.map[0][0]).add(B.mul(Q.input(input.chirp)));
  assert(B.n<BigInt(0));assert.equal(alpha.n,BigInt(0));
  const start=performance.now(),checked=verifyRealTimePacketWave(input,data.exact,data.wave);assert(checked);assert.equal(checked.samplesVerified,65);
  writeFileSync(root+`/negative-B-focus-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({input,samplesVerified:65,milliseconds:performance.now()-start},null,2)+'\n');
});
test('packet checker runtime graph excludes producer arithmetic andfixtures',()=>{
  const queue=['src/lib/learning-real-time-packet-wave-check.ts'],visited=new Set<string>();
  const allowed=new Set(['learning-real-time-packet-wave-check','learning-real-time-observation-check','learning-real-time-checker-interval',
    'learning-real-time-exact-core-check','learning-real-time-validation-rational','learning-real-time-phase-protocol','learning-real-time-phase-input','learning-qubit-wire'].map(x=>'src/lib/'+x+'.ts'));
  while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);assert(allowed.has(file),file);
    const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)) {
      const spec=statement.moduleSpecifier.text;if(spec.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),spec+'.ts')));
    }
  }
  assert.equal(visited.size,allowed.size);
});
