import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import ts from 'typescript';
import {createRealTimeCheckerInterval as context,checkerRealTimeRotation,checkerFloor,type CheckerInterval as I} from '../src/lib/learning-real-time-checker-interval';
import {certifiedRealTimeRotation} from '../src/lib/learning-real-time-phase-rotation';
const Z=BigInt(0),O=BigInt(1);
const root='.runtime/learning/real-time-checker-interval';mkdirSync(root,{recursive:true});
type Cell={lower:string;upper:string};
const fixture=JSON.parse(readFileSync('scripts/fixtures/real-time-checker-transcendental-references.json','utf8'))as{
  referenceBits:number;cases:{id:string;numerator:string;denominator:string;log:Cell;expNegative:Cell;highPrecisionExpNegative?:Cell&{referenceBits:number}}[]};
const rotations=JSON.parse(readFileSync('scripts/fixtures/real-time-phase-rotation-references.json','utf8'))as{
  pi:Cell&{referenceBits:number};cases:{id:string;numerator:string;denominator:string;precisionBits:number;referenceBits:number;sin:Cell;cos:Cell}[]};
const abs=(n:bigint)=>n<Z?-n:n;
function overlap(x:I,bits:number,cell:Cell,refBits:number) {
  const a=O<<BigInt(bits),b=O<<BigInt(refBits);
  assert(x.lower*b<=BigInt(cell.upper)*a,'lower diagnostic overlap');
  assert(x.upper*b>=BigInt(cell.lower)*a,'upper diagnostic overlap');
}
const c=context(1536);
for(const row of fixture.cases)test('independent log/exp: '+row.id,()=>{
  const x=c.from({n:BigInt(row.numerator),d:BigInt(row.denominator)}),log=c.log(x),exp=c.expNegative(x);
  overlap(log,c.bits,row.log,fixture.referenceBits);overlap(exp,c.bits,row.expNegative,fixture.referenceBits);
  assert(log.upper-log.lower<(O<<BigInt(c.bits-1088)),'log resolution');
  assert(exp.upper-exp.lower<(O<<BigInt(c.bits-1088)),'exp resolution');
  if(row.id==='one')assert.deepEqual(log,{lower:Z,upper:Z});
  if(row.id==='large')assert.deepEqual(row.expNegative,{lower:'0',upper:'1'},'Decimal underflow must not claim anexact zero');
});

test('independent pi identity and logarithm interval monotonicity',()=>{
  const pi=c.pi();overlap(pi,c.bits,rotations.pi,rotations.pi.referenceBits);
  assert(pi.upper-pi.lower<(O<<BigInt(32)));
  const a=c.from({n:BigInt(2),d:BigInt(3)}),b=c.from({n:BigInt(7),d:BigInt(3)});
  const interval=c.log({lower:a.lower,upper:b.upper});
  assert.equal(interval.lower,c.log(a).lower);assert.equal(interval.upper,c.log(b).upper);
});

test('directed integer arithmetic and integer-root inequalities on signed rational grid',()=>{
  for(let n=-17;n<=17;n++)for(const d of [1,3,19]) {
    const q={n:BigInt(n),d:BigInt(d)},x=c.from(q);
    assert(x.lower*q.d<=q.n*c.S&&x.upper*q.d>=q.n*c.S);
    for(const m of [-13,0,11]) {
      const y=c.from({n:BigInt(m),d:BigInt(7)}),product=c.multiply(x,y),num=BigInt(n*m),den=BigInt(d*7);
      assert(product.lower*den<=num*c.S&&product.upper*den>=num*c.S);
      if(m) {
        const quotient=c.divide(x,y),sign=m<0?-O:O,qn=BigInt(n*7)*sign,qd=BigInt(d*m)*sign;
        assert(quotient.lower*qd<=qn*c.S&&quotient.upper*qd>=qn*c.S);
      }
    }
    const square=c.square(x);assert(square.lower>=Z);
    const positive=c.from({n:BigInt(n*n),d:BigInt(d)});
    for(const degree of [2,3,4]as const){const r=c.root(positive,degree),k=BigInt(degree);
      assert(r.lower**k<=positive.lower*c.S**(k-O));assert(r.upper**k>=positive.upper*c.S**(k-O));
    }
  }
  assert.equal(checkerFloor(-BigInt(7),BigInt(3)),-BigInt(3));
  assert.equal(checkerFloor(BigInt(7),-BigInt(3)),-BigInt(3));
  assert.throws(()=>c.divide(c.one,{lower:-O,upper:O}));
  assert.throws(()=>c.log(c.zero));assert.throws(()=>c.expNegative(c.neg(c.one)));
  assert.throws(()=>c.root(c.neg(c.one),2));assert.throws(()=>c.from({n:O,d:Z}));
  assert.throws(()=>context(255));assert.throws(()=>context(32769));
});

const measurements:{id:string;milliseconds:number;workBits:number;halvings:number}[]=[];
const supported=rotations.cases.filter(row=>Math.max(0,abs(BigInt(row.numerator)).toString(2).length-BigInt(row.denominator).toString(2).length+1)<=8192);
for(const row of supported)test('halving/doubling phase: '+row.id,()=>{
  const start=performance.now(),actual=checkerRealTimeRotation({n:BigInt(row.numerator),d:BigInt(row.denominator)},row.precisionBits);
  for(const name of ['sin','cos']as const){
    overlap(actual[name],actual.bits,row[name],row.referenceBits);
    assert(actual[name].upper-actual[name].lower<=BigInt(2),'phase resolution');
    const limit=O<<BigInt(actual.bits);assert(actual[name].lower>=-limit&&actual[name].upper<=limit);
  }
  measurements.push({id:row.id,milliseconds:performance.now()-start,workBits:actual.workBits,halvings:actual.halvings});
  writeFileSync(root+`/rotations-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});

test('entire physical phase-height boundary and rational numerator budget',()=>{
  const q={n:(O<<BigInt(4568))+BigInt(17),d:BigInt(3)},a=checkerRealTimeRotation(q,1024),b=certifiedRealTimeRotation(q,1024);
  for(const key of ['sin','cos']as const)assert(a[key].lower<=b[key].upper&&a[key].upper>=b[key].lower);
  assert.throws(()=>checkerRealTimeRotation({n:O<<BigInt(8193),d:O}));
  assert.throws(()=>checkerRealTimeRotation({n:O,d:Z}));assert.throws(()=>checkerRealTimeRotation(q,2049));
  assert.throws(()=>checkerRealTimeRotation({n:O<<BigInt(65536),d:(O<<BigInt(65536))-O}));
});

test('raw underflow requires sufficient absolute work precision',()=>{
  const fine=context(9216);
  for(const row of fixture.cases.filter(x=>x.highPrecisionExpNegative)) {
    const actual=fine.expNegative(fine.from({n:BigInt(row.numerator),d:BigInt(row.denominator)})),ref=row.highPrecisionExpNegative!;
    overlap(actual,fine.bits,ref,ref.referenceBits);
    assert(actual.lower>Z,'fine precision resolves the tiny positive value');
    assert((actual.upper-actual.lower)*(O<<BigInt(512))<actual.lower,'more than512relative bits');
    assert.equal(c.expNegative(c.from({n:BigInt(row.numerator),d:BigInt(row.denominator)})).lower,Z);
  }
});

test('checker transcendental engine has no runtime imports',()=>{
  const source=ts.createSourceFile('checker.ts',readFileSync('src/lib/learning-real-time-checker-interval.ts','utf8'),ts.ScriptTarget.Latest,true);
  assert.equal(source.statements.filter(x=>ts.isImportDeclaration(x)).length,0);
  assert(supported.length>=15);assert(supported.length<rotations.cases.length,'arbitrary65536bit phases are outside the physical contract');
});
