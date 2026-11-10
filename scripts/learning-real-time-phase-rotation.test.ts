import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {certifiedRealTimeRotation as rotate,realTimePiInterval,type PhaseInterval} from '../src/lib/learning-real-time-phase-rotation';
const fixtures=JSON.parse(readFileSync('scripts/fixtures/real-time-phase-rotation-references.json','utf8')) as {
  certifiedReference:boolean;pi:{referenceBits:number;lower:string;upper:string};
  cases:{id:string;numerator:string;denominator:string;precisionBits:number;referenceBits:number;sin:{lower:string;upper:string};cos:{lower:string;upper:string}}[]};
const root='.runtime/learning/real-time-phase-rotation';mkdirSync(root,{recursive:true});
const measurements:{id:string;milliseconds:number;reductionBits:number;widthSin:string;widthCos:string}[]=[];
for(const item of fixtures.cases)test('directed large-phase rotation versus independent diagnostic: '+item.id,()=>{
  const started=performance.now();
  const result=rotate({n:BigInt(item.numerator),d:BigInt(item.denominator)},item.precisionBits);
  const factor=BigInt(1)<<BigInt(item.referenceBits-item.precisionBits);
  for(const name of ['sin','cos'] as const) {
    const value=result[name],reference=item[name];
    assert(value.lower<=value.upper);assert(value.lower*factor<=BigInt(reference.lower),item.id+' '+name+' lower');
    assert(value.upper*factor>=BigInt(reference.upper),item.id+' '+name+' upper');
    assert(value.upper-value.lower<=BigInt(2),'observation wider than two quanta');
    const scale=BigInt(1)<<BigInt(item.precisionBits);assert(value.lower>=-scale&&value.upper<=scale);
  }
  measurements.push({id:item.id,milliseconds:performance.now()-started,reductionBits:result.reductionBits,widthSin:String(result.sin.upper-result.sin.lower),widthCos:String(result.cos.upper-result.cos.lower)});
  writeFileSync(root+'/measurements-node'+process.versions.node.split('.')[0]+'.json',JSON.stringify(measurements,null,2)+'\n');
});
test('Machin pi contains the independent high precision diagnostic and has a controlled width',()=>{
  const pi=realTimePiInterval(1024),factor=BigInt(1)<<BigInt(fixtures.pi.referenceBits-1024);
  assert(pi.lower*factor<=BigInt(fixtures.pi.lower));assert(pi.upper*factor>=BigInt(fixtures.pi.upper));
  assert(pi.upper-pi.lower<BigInt(8192));assert(Object.isFrozen(pi));
  assert.equal(realTimePiInterval(1024),pi);
});
test('phase odd/even parity, exact zero and norm enclosure survive large signed input',()=>{
  const n=(BigInt(1)<<BigInt(4096))+BigInt(123),d=BigInt(999999999989);
  const a=rotate({n,d}),b=rotate({n:-n,d});
  assert.equal(a.sin.lower,-b.sin.upper);assert.equal(a.sin.upper,-b.sin.lower);assert.deepEqual(a.cos,b.cos);
  const square=(x:PhaseInterval)=>({lower:x.lower<=BigInt(0)&&x.upper>=BigInt(0)?BigInt(0):[x.lower*x.lower,x.upper*x.upper].reduce((a,b)=>a<b?a:b),upper:[x.lower*x.lower,x.upper*x.upper].reduce((a,b)=>a>b?a:b)});
  const s=square(a.sin),c=square(a.cos),norm=BigInt(1)<<BigInt(1024);
  assert(s.lower+c.lower<=norm&&s.upper+c.upper>=norm);
  const zero=rotate({n:BigInt(0),d:BigInt(3)});assert.deepEqual(zero.sin,{lower:BigInt(0),upper:BigInt(0)});assert.equal(zero.cos.lower,zero.cos.upper);
});
test('invalid rational and precision inputs are rejected before expensive work',()=>{
  for(const phase of [{n:BigInt(1),d:BigInt(0)},{n:BigInt(1),d:BigInt(-1)},{n:BigInt(1)<<BigInt(65536),d:BigInt(1)}])assert.throws(()=>rotate(phase),RangeError);
  for(const bits of [0,63,2049,Infinity,NaN,512.5])assert.throws(()=>rotate({n:BigInt(1),d:BigInt(1)},bits),RangeError);
  for(const bits of [63,68097,Infinity,NaN,512.5])assert.throws(()=>realTimePiInterval(bits),RangeError);
});
