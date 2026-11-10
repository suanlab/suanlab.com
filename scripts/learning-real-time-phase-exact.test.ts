import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {buildRealTimeChainExact,buildRealTimePacketExact,realTimeFractionText as text} from '../src/lib/learning-real-time-phase-exact';
import {sub,mul,sum,type F,type C} from '../src/lib/learning-quantum-hilbert-core';
import {Rational as Q} from '../src/lib/learning-graph-gauge-validation-math';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
import {realTimeChainCoreWire,realTimePacketCoreWire} from '../src/lib/learning-real-time-phase-core-wire';
import {realTimePhaseExamples,type RealTimePhaseInput} from '../src/lib/learning-real-time-phase-input';
interface RegulatorReference {inverseHash:string;mean:{real:string;imaginary:string}[];determinant:{real:string;imaginary:string};exponent:{real:string;imaginary:string};}
interface Reference {
  map:string[][];inertia:number[];determinant:string;hessianHash:string;refinements:string[][][];
  forcing:string[];constant:string;compatible:boolean;stationary:string[]|null;stationaryAction:string|null;inverseHash:string|null;regulator:RegulatorReference|null;
  variance:string;initialCovariance:string[][];finalCovariance:string[][];center:string[];centralAction:string;finalWidth:{real:string;imaginary:string};
}
const fixture=JSON.parse(readFileSync('scripts/fixtures/real-time-phase-exact-references.json','utf8')) as {cases:{id:string;input:RealTimePhaseInput;expected:Reference}[]};
const matrix=(a:F[][])=>a.map(row=>row.map(text));
const complex=(x:C)=>({real:text(x.re),imaginary:text(x.im)});
const qtext=(x:Q)=>x.d===BigInt(1)?String(x.n):`${x.n}/${x.d}`;
type GI={real:bigint;imaginary:bigint;denominator:bigint};
const gcd=(a:bigint,b:bigint):bigint=>{a=a<BigInt(0)?-a:a;b=b<BigInt(0)?-b:b;while(b)[a,b]=[b,a%b];return a;};
const gaussian=(x:{real:string;imaginary:string}):GI=>{
  const re=Q.read(x.real),im=Q.read(x.imaginary),denominator=re.d/gcd(re.d,im.d)*im.d;
  return {real:re.n*(denominator/re.d),imaginary:im.n*(denominator/im.d),denominator};
};
const gm=(a:GI,b:GI):GI=>({real:a.real*b.real-a.imaginary*b.imaginary,imaginary:a.real*b.imaginary+a.imaginary*b.real,denominator:a.denominator*b.denominator});
const quotient=(a:GI,b:GI)=>{
  const denominator=a.denominator*(b.real*b.real+b.imaginary*b.imaginary);
  return {real:qtext(new Q((a.real*b.real+a.imaginary*b.imaginary)*b.denominator,denominator)),
    imaginary:qtext(new Q((a.imaginary*b.real-a.real*b.imaginary)*b.denominator,denominator))};
};
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const root='.runtime/learning/real-time-phase-exact';mkdirSync(root,{recursive:true});
const measurements:{id:string;milliseconds:number;dimension:number;wireBytes:number;wireCharacters:number;maxString:number}[]=[];
for(const item of fixture.cases)test('finite exact core versus independent full dense reference: '+item.id,()=>{
  const started=performance.now(),e=item.expected;let wire:unknown;
  if(item.input.mode==='chain') {
    const r=buildRealTimeChainExact(item.input),encoded=realTimeChainCoreWire(r);wire=encoded;
    if(encoded.inverse) {
      const factors=encoded.inverse.continuants.map(Q.read),scale=Q.read(encoded.inverse.scale);
      const recovered=Array.from({length:r.d},(_,i)=>Array.from({length:r.d},(_,j)=>qtext(new Q(factors[Math.min(i,j)].n*factors[r.d-1-Math.max(i,j)].n*scale.d*factors[r.d].d,factors[Math.min(i,j)].d*factors[r.d-1-Math.max(i,j)].d*scale.n*factors[r.d].n))));
      assert.equal(hash(recovered),e.inverseHash,'all exact inverse cells reconstructed from transport factors');
    }
    if(encoded.regulator) {
      const factors=encoded.regulator.continuants.map(gaussian),powers=encoded.regulator.distancePowers.map(gaussian);
      const upper:({real:string;imaginary:string}|undefined)[][]=Array.from({length:r.d},()=>[]);
      const recovered=Array.from({length:r.d},(_,i)=>Array.from({length:r.d},(_,j)=>{
        if(i>j)return upper[j][i]!;
        const z=quotient(gm(gm(factors[i],factors[r.d-1-j]),powers[j-i]),factors[r.d]);upper[i][j]=z;return z;
      }));
      assert.equal(hash(recovered),e.regulator?.inverseHash,'all complex inverse cells reconstructed from transport factors');
    }
    assert.deepEqual(r.input,item.input);assert(Object.isFrozen(r.input));
    assert.deepEqual(matrix(r.map.total),e.map);assert.deepEqual([r.inertia.positive,r.inertia.negative,r.inertia.zero],e.inertia);
    assert.equal(text(r.determinant),e.determinant);
    assert.equal(hash(Array.from({length:r.d},(_,i)=>Array.from({length:r.d},(_,j)=>text(r.hessianEntry(i,j))))),e.hessianHash);
    assert.deepEqual(r.forcing.map(text),e.forcing);assert.equal(text(r.constant),e.constant);assert.equal(r.compatible,e.compatible);
    assert.equal(r.stationaryAction?text(r.stationaryAction):null,e.stationaryAction);
    if(e.inverseHash) {
      assert.equal(hash(Array.from({length:r.d},(_,i)=>Array.from({length:r.d},(_,j)=>text(r.inverseEntry(i,j))))),e.inverseHash);
      assert.deepEqual(r.stationary?.map(text),e.stationary);
    } else {
      assert.throws(()=>r.inverseEntry(0,0),RangeError);assert.equal(r.nullVector?.length,r.d);
      assert.equal(r.causticIndex,r.inertia.negative+1);
      assert.equal(text(r.map.total[0][0]),r.causticIndex!%2?'-1':'1');
      if(r.stationary)for(let i=0;i<r.d;i++) {
        const residual=sub(sum(r.stationary.map((x,j)=>mul(r.hessianEntry(i,j),x))),r.forcing[i]);
        assert.equal(text(residual),'0','affine representative residual');
        assert.equal(text(sum(r.nullVector!.map((x,j)=>mul(r.hessianEntry(i,j),x)))),'0','full null vector');
      }
    }
    assert.deepEqual(r.refinements.map(v=>matrix(v.map)),e.refinements);
    if(e.regulator) {
      assert(r.regulator);const q=r.regulator;
      assert.equal(hash(Array.from({length:r.d},(_,i)=>Array.from({length:r.d},(_,j)=>complex(q.inverseEntry(i,j))))),e.regulator.inverseHash);
      assert.deepEqual(q.mean.map(complex),e.regulator.mean);assert.deepEqual(complex(q.determinant),e.regulator.determinant);
      assert.deepEqual(complex(q.exponent),e.regulator.exponent);assert(q.pivots.every(v=>v.re.n>BigInt(0)));
    } else assert.equal(r.regulator,null);
  } else if(item.input.mode==='packet') {
    const r=buildRealTimePacketExact(item.input);wire=realTimePacketCoreWire(r);
    assert.deepEqual(matrix(r.map.total),e.map);assert.equal(text(r.variance),e.variance);
    assert.deepEqual(matrix(r.initialCovariance),e.initialCovariance);assert.deepEqual(matrix(r.finalCovariance),e.finalCovariance);
    assert.deepEqual(r.center.map(text),e.center);assert.equal(text(r.centralAction),e.centralAction);assert.deepEqual(complex(r.finalWidth),e.finalWidth);
    assert.equal(r.negativeModes,e.inertia[1]);assert(r.finalWidth.im.n>BigInt(0));
    assert.deepEqual(r.refinements.map(v=>matrix(v.map)),e.refinements);
    const determinant=sub(mul(r.finalCovariance[0][0],r.finalCovariance[1][1]),mul(r.finalCovariance[0][1],r.finalCovariance[1][0]));
    assert.equal(text(mul(determinant,{n:BigInt(4),d:BigInt(1)})),text(mul(r.hbar,r.hbar)));
  } else assert.fail('fold does not belong to the finite quadratic core');
  assert.deepEqual(snapshotQubitWire(wire),wire);
  const lengths:number[]=[];const visit=(value:unknown):void=>{if(typeof value==='string')lengths.push(value.length);else if(Array.isArray(value))value.forEach(visit);else if(value&&typeof value==='object')Object.values(value).forEach(visit);};visit(wire);
  measurements.push({id:item.id,milliseconds:performance.now()-started,dimension:item.input.intervals-1,wireBytes:Buffer.byteLength(JSON.stringify(wire)),wireCharacters:lengths.reduce((a,b)=>a+b,0),maxString:Math.max(...lengths)});
  writeFileSync(root+'/measurements-node'+process.versions.node.split('.')[0]+'.json',JSON.stringify(measurements,null,2)+'\n');
});
test('finite-core mode boundaries do not reinterpret fold or packet data as chain inputs',()=>{
  assert.deepEqual(fixture.cases.map(item=>item.id),realTimePhaseExamples.filter(item=>item.input.mode!=='fold').map(item=>item.id));
  assert.throws(()=>buildRealTimeChainExact({mode:'fold',control:'0',hbar:'1'}),RangeError);
  assert.throws(()=>buildRealTimePacketExact({mode:'fold',control:'0',hbar:'1'}),RangeError);
});
