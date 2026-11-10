import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {realTimePhaseExamples,type RealTimePhaseInput} from '../src/lib/learning-real-time-phase-input';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
import {packRealTimeSymmetricReal,packRealTimeSymmetricComplex,type BinaryCoefficientRange} from '../src/lib/learning-real-time-matrix-observations';
const Z=BigInt(0),O=BigInt(1),REF_BITS=704;
type Json=string|number|boolean|null|Json[]|{[key:string]:Json};
const fixture=JSON.parse(readFileSync('scripts/fixtures/real-time-wave-references.json','utf8')) as {referenceBits:number;cases:{id:string;input:RealTimePhaseInput;expected:Json}[]};
const root='.runtime/learning/real-time-phase-numerics';mkdirSync(root,{recursive:true});
const measurements:{id:string;milliseconds:number;bytes:number;characters:number;matrixCells:number;samples:number}[]=[];
const read=(s:string)=>{const[n,d='1']=s.split('/');return{n:BigInt(n),d:BigInt(d)};};
function match(actual:unknown,expected:Json,path:string,scale=0):void {
  if(expected===null||typeof expected!=='object'){assert.equal(actual,expected,path);return;}
  assert(actual&&typeof actual==='object',path);
  if(Array.isArray(expected)) {assert(Array.isArray(actual));assert.equal(actual.length,expected.length);expected.forEach((x,i)=>match(actual[i],x,path+'/'+i));return;}
  const a=actual as Record<string,unknown>,e=typeof a.scalePowerOfTwo==='number'?a.scalePowerOfTwo:scale;
  if(typeof expected.lower==='string'&&typeof expected.upper==='string') {
    assert.equal(typeof a.lower,'string');assert.equal(typeof a.upper,'string');
    const lo=read(a.lower as string),hi=read(a.upper as string),bits=typeof expected.referenceBits==='number'?expected.referenceBits:REF_BITS,S=O<<BigInt(bits);
    const nlo=e>=0?lo.n<<BigInt(e):lo.n,nhi=e>=0?hi.n<<BigInt(e):hi.n;
    const dlo=e>=0?lo.d:lo.d<<BigInt(-e),dhi=e>=0?hi.d:hi.d<<BigInt(-e);
    assert(nlo*S<=BigInt(expected.lower)*dlo,'lower '+path);assert(nhi*S>=BigInt(expected.upper)*dhi,'upper '+path);return;
  }
  for(const [key,value]of Object.entries(expected))match(a[key],value,path+'/'+key,e);
}
function coefficientContains(range:BinaryCoefficientRange,exponent:number,n:bigint,d:bigint,label:string) {
  if(d<Z){n=-n;d=-d;}assert(d>Z);
  const shift=512-exponent;
  if(shift>=0)n<<=BigInt(shift);else d<<=BigInt(-shift);
  assert(BigInt(range.lower)*d<=n&&BigInt(range.upper)*d>=n,'matrix enclosure '+label);
  assert(BigInt(range.upper)-BigInt(range.lower)<=BigInt(3),'matrix width '+label);
}
type GI={r:bigint;i:bigint;d:bigint};
const gaussian=(z:{real:string;imaginary:string}):GI=>{const r=read(z.real),i=read(z.imaginary);return{r:r.n*i.d,i:i.n*r.d,d:r.d*i.d};};
const gm=(a:GI,b:GI):GI=>({r:a.r*b.r-a.i*b.i,i:a.r*b.i+a.i*b.r,d:a.d*b.d});
for(const item of fixture.cases)test('full finite real-time producer andall wave samples: '+item.id,()=>{
  const started=performance.now(),r=calculateRealTimePhase(item.input);assert(r.kind!=='canonical-fold');
  assert.deepEqual(r.input,item.input);assert.deepEqual(snapshotQubitWire(r),r);match(r.wave,item.expected,item.id);
  let cells=0;
  if(r.kind==='chain') {
    const exact=r.exact,dim=exact.dimension;
    if(r.inverse) {
      assert(exact.inverse);const factors=exact.inverse.continuants.map(read),s=read(exact.inverse.scale),det=factors[dim];
      assert.equal(r.inverse.upperRows.length,dim);
      for(let i=0;i<dim;i++) {
        assert.equal(r.inverse.upperRows[i].length,dim-i);
        for(let j=i;j<dim;j++) {
          const a=factors[i],b=factors[dim-1-j];
          coefficientContains(r.inverse.upperRows[i][j-i],r.inverse.scalePowerOfTwo,a.n*b.n*s.d*det.d,a.d*b.d*s.n*det.n,item.id+' inverse '+i+','+j);cells++;
        }
      }
    } else {assert.equal(exact.inertia.zero,1);assert.equal(r.wave.unregulated.kind,'parity-delta');assert.equal(r.wave.unregulated.pointwiseValue,null);}
    if(r.wave.regulated?.rawValueRequiresScaledRepresentation) {
      for(const value of [r.wave.regulated.value.real,r.wave.regulated.value.imaginary])assert(read(value.lower).n<=Z&&read(value.upper).n>=Z,'raw enclosure cannot establish nonzero magnitude');
      assert(r.wave.regulated.logMagnitude);
    }
    if(r.wave.regulated&&r.exact.regulator?.exponent.real==='0')assert.deepEqual(r.wave.regulated.value,r.wave.regulated.scaledValue,'no damping must preserve the scaled value');
    if(r.regulatedConnectedResponse) {
      assert(exact.regulator);assert.equal(exact.regulator.pivotLayout,'successive-continuant-ratios');assert.equal(exact.regulator.pivotRootBranch,'positive-real-part-per-ratio');const theta=exact.regulator.continuants.map(gaussian),powers=exact.regulator.distancePowers.map(gaussian),det=theta[dim],h=read(r.input.hbar);
      for(let i=0;i<dim;i++) {
        assert.equal(r.regulatedConnectedResponse.upperRows[i].length,dim-i);
        for(let j=i;j<dim;j++) {
          const a=gm(gm(theta[i],theta[dim-1-j]),powers[j-i]),d=a.d*(det.r*det.r+det.i*det.i)*h.d;
          const nre=(a.r*det.r+a.i*det.i)*det.d*h.n,nim=(a.i*det.r-a.r*det.i)*det.d*h.n;
          const observed=r.regulatedConnectedResponse.upperRows[i][j-i],e=r.regulatedConnectedResponse.scalePowerOfTwo;
          coefficientContains(observed.real,e,nre,d,item.id+' covariance real '+i+','+j);coefficientContains(observed.imaginary,e,nim,d,item.id+' covariance imaginary '+i+','+j);cells++;
        }
      }
    }
  } else assert.equal(r.wave.samples.length,65);
  let characters=0;const visit=(x:unknown):void=>{if(typeof x==='string')characters+=x.length;else if(Array.isArray(x))x.forEach(visit);else if(x&&typeof x==='object')Object.values(x).forEach(visit);};visit(r);
  assert(characters<=2000000);
  measurements.push({id:item.id,milliseconds:performance.now()-started,bytes:Buffer.byteLength(JSON.stringify(r)),characters,matrixCells:cells,samples:r.kind==='packet'?65:0});
  writeFileSync(`${root}/wave-tests-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});
test('all finite teaching examples have independent wave references',()=>{
  assert.deepEqual(fixture.cases.map(x=>x.id),realTimePhaseExamples.filter(x=>x.input.mode!=='fold').map(x=>x.id));assert.equal(fixture.referenceBits,REF_BITS);
});
test('shared binary matrix scale handles extreme magnitudes,tiny signed entries andexact zeros',()=>{
  for(const e of [-10000,-512,0,512,10000]) {
    const shift=O<<BigInt(Math.abs(e));const factor=e>=0?{n:shift,d:O}:{n:O,d:shift};
    const values=[[{n:BigInt(-11)*factor.n,d:BigInt(7)*factor.d},{n:Z,d:O}],[{n:Z,d:O},{n:factor.n,d:BigInt(3)*factor.d}]];
    const p=packRealTimeSymmetricReal(2,(i,j)=>values[i][j]);assert.deepEqual(p.upperRows[0][1],{lower:'0',upper:'0'});
    for(let i=0;i<2;i++)for(let j=i;j<2;j++)coefficientContains(p.upperRows[i][j-i],p.scalePowerOfTwo,values[i][j].n,values[i][j].d,'toy');
    const c=packRealTimeSymmetricComplex(2,(i,j)=>({re:values[i][j],im:{n:-values[i][j].n,d:values[i][j].d}}));
    for(let i=0;i<2;i++)for(let j=i;j<2;j++){
      coefficientContains(c.upperRows[i][j-i].real,c.scalePowerOfTwo,values[i][j].n,values[i][j].d,'complex real');
      coefficientContains(c.upperRows[i][j-i].imaginary,c.scalePowerOfTwo,-values[i][j].n,values[i][j].d,'complex imaginary');
    }
    assert.deepEqual(snapshotQubitWire(c),c);
  }
  assert.throws(()=>packRealTimeSymmetricReal(32,()=>({n:O,d:O})),RangeError);
  assert.throws(()=>packRealTimeSymmetricReal(1,()=>({n:O,d:Z})),RangeError);
});
test('all four Fresnel quadrants retain negative-mode phase at zero stationary action',()=>{
  for(const [n,omega,nu,realSign,imagSign]of [[2,'0',0,1,-1],[2,'4',1,-1,-1],[3,'4',2,-1,1],[4,'4',3,1,1]] as const) {
    const r=calculateRealTimePhase({mode:'chain',intervals:n,time:String(n),mass:'1',hbar:'1/2',omegaSquared:omega,start:'0',end:'0',damping:'0',probe:1});
    assert(r.kind==='chain');assert.equal(r.exact.inertia.negative,nu);assert.equal(r.exact.stationaryAction,'0');assert(r.wave.unregulated.kind==='finite-quadratic-kernel');
    for(const [range,sign]of [[r.wave.unregulated.unitPhase.real,realSign],[r.wave.unregulated.unitPhase.imaginary,imagSign]] as const) {
      const a=read(sign>0?range.lower:range.upper),b=read(sign>0?range.upper:range.lower),lo=sign>0?a.n:-a.n,hi=sign>0?b.n:-b.n;
      assert(lo>Z&&hi>Z);assert(BigInt(2)*lo*lo<=a.d*a.d&&BigInt(2)*hi*hi>=b.d*b.d,'exact square root of one half');
    }
  }
});
