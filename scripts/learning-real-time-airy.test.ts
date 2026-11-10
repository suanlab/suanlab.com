import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {calculateRealTimeFold,observeRealTimeAiryOrigin,type RealTimeAiryStrategy} from '../src/lib/learning-real-time-airy';
import {realTimeFraction as read} from '../src/lib/learning-real-time-phase-exact';
import {certifiedRealTimeAiryOrigin} from '../src/lib/learning-real-time-airy-origin';
import {createRealTimeIntervalContext,type RealTimeRange} from '../src/lib/learning-real-time-interval';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
interface ReferenceCell {lower:string;upper:string;}
interface ComplexReferencePoint {position:{real:ReferenceCell;imaginary:ReferenceCell};action:{real:ReferenceCell;imaginary:ReferenceCell};curvature:{real:ReferenceCell;imaginary:ReferenceCell};contributesToRealContour:boolean;}
interface Row {id:string;input:{mode:'fold';control:string;hbar:string};strategy:RealTimeAiryStrategy;expected:Record<string,ReferenceCell|null|{position:ReferenceCell;action:ReferenceCell;curvature:ReferenceCell}[]|ComplexReferencePoint[]>;}
const fixture=JSON.parse(readFileSync('scripts/fixtures/real-time-airy-references.json','utf8')) as {referenceBits:number;cases:Row[]};
const root='.runtime/learning/real-time-airy';mkdirSync(root,{recursive:true});
const measurements:{id:string;milliseconds:number;bytes:number;method:string;terms:number;scaledSign:string;maxWidthQuanta:string}[]=[];
function compare(actual:RealTimeRange,expected:ReferenceCell,label:string) {
  const lo=read(actual.lower),hi=read(actual.upper),S=BigInt(1)<<BigInt(fixture.referenceBits);
  assert(lo.n*hi.d<=hi.n*lo.d,'ordered '+label);
  assert(lo.n*S<=BigInt(expected.lower)*lo.d,'lower '+label);
  assert(hi.n*S>=BigInt(expected.upper)*hi.d,'upper '+label);
}
for(const item of fixture.cases)test('certified physical fold versus independent diagnostic: '+item.id,()=>{
  const started=performance.now(),r=calculateRealTimeFold(item.input,item.strategy);
  assert.deepEqual(r.input,item.input);assert(Object.isFrozen(r.input));assert.deepEqual(snapshotQubitWire(r),r);
  let maximum=BigInt(0);
  for(const field of ['argument','phaseScale','airy','scaledAiry','integral','scaledIntegral','negativeExponential'] as const) {
    compare(r[field],item.expected[field] as ReferenceCell,item.id+' '+field);
    const a=read(r[field].lower),b=read(r[field].upper),n=(b.n*a.d-a.n*b.d)*(BigInt(1)<<BigInt(512)),d=a.d*b.d;
    const width=(n+d-BigInt(1))/d;if(width>maximum)maximum=width;
  }
  if(r.logMagnitude)compare(r.logMagnitude,item.expected.logMagnitude as ReferenceCell,'log magnitude');
  else assert.equal(r.scaledSign,'unresolved');
  if(r.stationaryApproximation)compare(r.stationaryApproximation,item.expected.stationaryApproximation as ReferenceCell,'stationary approximation');
  if(item.expected.stationaryPoints) {
    const expected=item.expected.stationaryPoints as {position:ReferenceCell;action:ReferenceCell;curvature:ReferenceCell}[];
    assert.equal(r.stationaryPoints.length,expected.length);
    r.stationaryPoints.forEach((point,i)=>{for(const key of ['position','action','curvature'] as const)compare(point[key],expected[i][key],'stationary point '+i+' '+key);});
  }
  const complexExpected=item.expected.complexStationaryPoints as ComplexReferencePoint[];
  assert.equal(r.complexStationaryPoints.length,complexExpected.length);
  r.complexStationaryPoints.forEach((point,i)=>{
    assert.equal(point.contributesToRealContour,complexExpected[i].contributesToRealContour);
    for(const key of ['position','action','curvature'] as const)for(const part of ['real','imaginary'] as const)compare(point[key][part],complexExpected[i][key][part],'complex stationary '+i+' '+key+' '+part);
  });
  assert.equal(r.exactZeroClaimed,false);
  if(item.input.control.startsWith('-')) {
    assert.equal(r.scaledSign,'positive');assert(read(r.scaledIntegral.lower).n>BigInt(0));assert(read(r.integral.lower).n>=BigInt(0));assert(r.logMagnitude);
    if(item.input.hbar==='1/999999999989'&&item.input.control==='-16') {
      assert(r.rawValueBelowObservationQuantum);assert.equal(r.integral.lower,'0');assert(read(r.integral.upper).n>BigInt(0));
      assert(read(r.logMagnitude.upper).n<BigInt(0));
    }
  }
  assert(maximum<=BigInt(16),'unexpectedly wide512bit observations');
  measurements.push({id:item.id,milliseconds:performance.now()-started,bytes:Buffer.byteLength(JSON.stringify(r)),method:r.method,terms:r.terms,scaledSign:r.scaledSign,maxWidthQuanta:String(maximum)});
  writeFileSync(`${root}/measurements-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});
test('computed origin constants agree with independently certified positive-integral bounds',()=>{
  const reference=JSON.parse(readFileSync('scripts/fixtures/real-time-airy-origin-references.json','utf8'));
  const result=observeRealTimeAiryOrigin(),S=BigInt(1)<<BigInt(reference.referenceBits);
  for(const [name,key] of [['value','airyAtZero'],['derivative','airyDerivativeAtZero']] as const) {
    const lo=read(result[name].lower),hi=read(result[name].upper);
    assert(lo.n*S<=BigInt(reference[key].lower)*lo.d);assert(hi.n*S>=BigInt(reference[key].upper)*hi.d);
  }
  const a=certifiedRealTimeAiryOrigin();assert(Object.isFrozen(a)&&Object.isFrozen(a.value)&&Object.isFrozen(a.derivative));assert.equal(certifiedRealTimeAiryOrigin(),a);
});
test('central and asymptotic paths overlap at both signs without replacing physical controls',()=>{
  for(const control of ['4','17/4','9/2','-4','-17/4','-9/2']) {
    const input={mode:'fold',control,hbar:'1/64'};
    const a=calculateRealTimeFold(input,'central'),b=calculateRealTimeFold(input,'asymptotic');
    for(const field of ['scaledAiry','scaledIntegral'] as const) {
      const al=read(a[field].lower),ah=read(a[field].upper),bl=read(b[field].lower),bh=read(b[field].upper);
      assert(al.n*bh.d<=bh.n*al.d&&bl.n*ah.d<=ah.n*bl.d,'disjoint overlap '+control+' '+field);
    }
  }
});
test('boundaries and invalid strategies do not silently switch to a reduced input problem',()=>{
  assert.throws(()=>calculateRealTimeFold({mode:'fold',control:'17',hbar:'1'}),RangeError);
  assert.throws(()=>calculateRealTimeFold({mode:'fold',control:'0',hbar:'0'}),RangeError);
  assert.throws(()=>calculateRealTimeFold({mode:'fold',control:'16',hbar:'1/999999999989'},'central'),RangeError);
  assert.throws(()=>calculateRealTimeFold({mode:'fold',control:'1',hbar:'1'},'asymptotic'),RangeError);
  assert.throws(()=>calculateRealTimeFold({mode:'fold',control:'0',hbar:'1'},'unknown' as RealTimeAiryStrategy),RangeError);
});
test('directed real-time roots, logarithm and decay retain known exact cases and nonzero underflow bounds',()=>{
  const ctx=createRealTimeIntervalContext(1024);
  for(const [n,k,r] of [[4,2,2],[8,3,2],[81,4,3]] as const)assert.deepEqual(ctx.root(ctx.integer(n),k),ctx.integer(r));
  const log=ctx.log(ctx.one);assert.equal(log.lower,BigInt(0));assert.equal(log.upper,BigInt(0));
  assert.deepEqual(ctx.expNegative(ctx.zero),ctx.one);
  const tiny=ctx.expNegative(ctx.integer(100000));assert.equal(tiny.lower,BigInt(0));assert(tiny.upper>BigInt(0));
  assert.throws(()=>ctx.divide(ctx.one,{lower:-BigInt(1),upper:BigInt(1)}),RangeError);
  assert.throws(()=>ctx.root(ctx.integer(-1),2),RangeError);assert.throws(()=>ctx.log(ctx.zero),RangeError);
});
