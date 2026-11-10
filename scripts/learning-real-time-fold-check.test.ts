import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimeFold} from '../src/lib/learning-real-time-fold-check';
import {checkerRealTimeFoldReference,checkerRealTimeAiryTerms} from '../src/lib/learning-real-time-checker-airy';
import {checkerRealTimeAiryOrigin} from '../src/lib/learning-real-time-checker-airy-origin';
import {createRealTimeCheckerInterval,type CheckerInterval as I} from '../src/lib/learning-real-time-checker-interval';
import {calculateRealTimeFold,type RealTimeAiryStrategy} from '../src/lib/learning-real-time-airy';
import {realTimeChainDefault,type RealTimeFoldInput} from '../src/lib/learning-real-time-phase-input';
import {CheckerRational as Q} from '../src/lib/learning-real-time-validation-rational';
const root='.runtime/learning/real-time-fold-check';mkdirSync(root,{recursive:true});
type Cell={lower:string;upper:string};
const fixture=JSON.parse(readFileSync('scripts/fixtures/real-time-airy-references.json','utf8'))as{
 referenceBits:number;cases:{id:string;input:RealTimeFoldInput;strategy:RealTimeAiryStrategy;expected:Record<string,unknown>}[]};
function overlap(actual:I,bits:number,cell:Cell,refBits:number) {
 const a=BigInt(1)<<BigInt(bits),b=BigInt(1)<<BigInt(refBits);
 assert(actual.lower*b<=BigInt(cell.upper)*a);assert(actual.upper*b>=BigInt(cell.lower)*a);
}
test('independent positive Gamma origin certificates agree withcertified integral references',()=>{
 const origin=checkerRealTimeAiryOrigin(),reference=JSON.parse(readFileSync('scripts/fixtures/real-time-airy-origin-references.json','utf8'));
 for(const [key,name]of [['value','airyAtZero'],['derivative','airyDerivativeAtZero'],['gammaOneThird','gammaOneThird'],['gammaTwoThirds','gammaTwoThirds']]as const) {
  overlap(origin[key],origin.bits,reference[name],reference.referenceBits);
  assert(origin[key].upper-origin[key].lower<(BigInt(1)<<BigInt(origin.bits-2400)));
 }
 assert(Object.isFrozen(origin)&&Object.isFrozen(origin.value)&&Object.isFrozen(origin.terms));assert.equal(checkerRealTimeAiryOrigin(),origin);
 const ctx=createRealTimeCheckerInterval(origin.bits),product=ctx.multiply(origin.gammaOneThird,origin.gammaTwoThirds),reflection=ctx.divide(ctx.scale(ctx.pi(),2),ctx.root(ctx.integer(3),2));
 assert(product.lower<=reflection.upper&&product.upper>=reflection.lower);
 writeFileSync(root+`/origin-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({workBits:origin.bits,certifiedBits:origin.certifiedBits,cutoff:origin.cutoff,terms:origin.terms},null,2)+'\n');
});
const cache=new Map<string,ReturnType<typeof calculateRealTimeFold>>(),measurements:{id:string;milliseconds:number;method:string;terms:number;rawBelowQuantum:boolean}[]=[];
for(const row of fixture.cases)test('whole fold response andindependent reference: '+row.id,()=>{
 const result=calculateRealTimeFold(row.input,row.strategy);cache.set(row.id,result);const start=performance.now(),checked=verifyRealTimeFold(row.input,result);assert(checked,row.id);
 assert.equal(checked.scope,'canonical-fold-response-only');assert.equal(checked.fullResponseValidated,false);assert(Object.isFrozen(checked.response));
 const ref=checkerRealTimeFoldReference(row.input,result.method);
 for(const key of ['argument','phaseScale','airy','scaledAiry','integral','scaledIntegral','negativeExponential','logMagnitude','stationaryApproximation']as const) {
  const actual=ref[key],cell=row.expected[key];if(actual&&cell)overlap(actual,ref.ctx.bits,cell as Cell,fixture.referenceBits);
 }
 measurements.push({id:row.id,milliseconds:performance.now()-start,method:result.method,terms:result.terms,rawBelowQuantum:result.rawValueBelowObservationQuantum});
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
for(const id of ['u0-h1','u1-h1','u-1-h1','u-16-h1/999999999989'])test('reject every primitive fold mutation: '+id,()=>{
 const row=fixture.cases.find(x=>x.id===id)!,result=cache.get(id)!;let rejected=0;
 for(const leaf of leaves(result)) {
  const next=leaf.path.length===1&&leaf.path[0]==='terms'?0:
   typeof leaf.value==='string'&&/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(leaf.value)?Q.input(leaf.value).add(Q.of(BigInt(1))).text():
   typeof leaf.value==='number'?leaf.value+1:typeof leaf.value==='string'?'wrong-'+leaf.value:typeof leaf.value==='boolean'?!leaf.value:'not-null';
  assert.equal(verifyRealTimeFold(row.input,mutate(result,leaf.path,next)),null,leaf.path.join('.'));rejected++;
 }
 writeFileSync(root+`/mutations-${encodeURIComponent(id)}-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({id,rejected},null,2)+'\n');
});
test('terms are valid remainder witnesses rather than producer execution transcripts',()=>{
 const input={mode:'fold',control:'1',hbar:'1'}as const,r=cache.get('u1-h1')!;
 assert(checkerRealTimeAiryTerms(input,r.method,r.terms));assert(verifyRealTimeFold(input,{...r,terms:r.terms+1}));
 for(const n of [0,-1,1e9,NaN])assert(!checkerRealTimeAiryTerms(input,r.method,n));
 assert(!checkerRealTimeAiryTerms(input,'oscillatory-asymptotic',2));
 assert(!checkerRealTimeAiryTerms({mode:'fold',control:'16',hbar:'1/999999999989'},'central-ode-series',2048));
});
test('raw below-quantum classification crosses the physical decay boundary',()=>{
 const flags:boolean[]=[];
 for(const denominator of [520,530]) {
  const input={mode:'fold',control:'-1',hbar:`1/${denominator}`}as const,r=calculateRealTimeFold(input);
  assert(verifyRealTimeFold(input,r));flags.push(r.rawValueBelowObservationQuantum);
  assert.equal(verifyRealTimeFold(input,{...r,rawValueBelowObservationQuantum:!r.rawValueBelowObservationQuantum}),null);
 }
 assert.deepEqual(flags,[false,true]);
});
test('canonical result bounds,raw input spelling,ownership andattribution remain distinct',()=>{
 const input={mode:'fold',control:'2/2',hbar:'3/3'}as const,result=calculateRealTimeFold(input);assert(verifyRealTimeFold(input,result));
 assert.equal(verifyRealTimeFold({...input,control:'1'},result),null);
 assert.equal(verifyRealTimeFold(realTimeChainDefault,result),null);
 assert.equal(verifyRealTimeFold(input,mutate(result,['airy','lower'],'2/2')),null);
 assert.equal(verifyRealTimeFold(input,mutate(result,['airy','upper'],'1/3')),null);
 assert.equal(verifyRealTimeFold(input,{...result,extra:true}),null);
 assert.equal(verifyRealTimeFold(input,{...result,[Symbol('hidden')]:true}),null);
 let calls=0;const getter={...result};Object.defineProperty(getter,'scaledAiry',{get(){calls++;throw Error('getter');}});
 assert.equal(verifyRealTimeFold(input,getter),null);assert.equal(calls,0);
 const revoked=Proxy.revocable({},{});revoked.revoke();assert.equal(verifyRealTimeFold(input,revoked.proxy),null);
});
test('deep decay preserves scaled/log evidence andcannot claim exactzero orwrong contours',()=>{
 const input={mode:'fold',control:'-16',hbar:'1/999999999989'}as const,r=cache.get('u-16-h1/999999999989')!;
 assert(r.rawValueBelowObservationQuantum);assert(r.logMagnitude);
 for(const key of ['airy','integral','scaledAiry','scaledIntegral']as const)assert.equal(verifyRealTimeFold(input,mutate(r,[key],{lower:'0',upper:'0'})),null,key);
 assert.equal(verifyRealTimeFold(input,{...r,exactZeroClaimed:true}),null);
 assert.equal(verifyRealTimeFold(input,mutate(r,['complexStationaryPoints',1,'contributesToRealContour'],true)),null);
 assert.equal(verifyRealTimeFold(input,{...r,stationaryPoints:[{position:{lower:'0',upper:'0'},action:{lower:'0',upper:'0'},curvature:{lower:'0',upper:'0'}}]}),null);
});
test('fold checker runtime graph excludes all producer engines andfixtures',()=>{
 const queue=['src/lib/learning-real-time-fold-check.ts'],visited=new Set<string>();
 const allowed=new Set(['learning-real-time-fold-check','learning-real-time-checker-airy','learning-real-time-checker-airy-origin','learning-real-time-checker-interval',
  'learning-real-time-observation-check','learning-real-time-validation-rational','learning-real-time-phase-protocol','learning-real-time-phase-input','learning-qubit-wire'].map(x=>'src/lib/'+x+'.ts'));
 while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);assert(allowed.has(file),file);
  const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
  for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)) {
   const spec=statement.moduleSpecifier.text;if(spec.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),spec+'.ts')));
  }
 }
 assert.equal(visited.size,allowed.size);
});
