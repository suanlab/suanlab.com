import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { schmidtSpectrumCertificate as spectrum, schmidtLogCertificate as logarithm } from '../src/lib/learning-many-body-schmidt-spectral-wire';
import { f, zero, one, int, add, sub, mul, div, neg, eq, le, point, radd, rmul, type F, type Range } from '../src/lib/learning-tensor-wire-math';
ensureTestArtifactDirectory('.runtime/learning/many-body-schmidt-spectral', { recursive: true });

const runtime = '.runtime/learning/many-body-schmidt-spectral';
const refs = JSON.parse(readFileSync('scripts/fixtures/schmidt-reference.json','utf8')) as {cases:{label:string;expected:{cuts:{characteristicPolynomial:string[];schmidtRank:number;spectrum:{weight:string;multiplicity:number;zeroWeight:boolean}[];entropyNats:string;entropyBits:string}[]}}[]};
const logs = JSON.parse(readFileSync('scripts/fixtures/schmidt-wire-log-reference.json','utf8')) as {cases:{numerator:string;denominator:string;logarithm:string}[]};
function parse(s:string):F {
  if(s.includes('/')){const[n,d]=s.split('/');return f(BigInt(n),BigInt(d));}
  const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s.replace('decimal:',''));assert(m,s);
  const n=BigInt(m[1]+m[2]+(m[3]??'')),e=Number(m[4]??0)-(m[3]??'').length;return e>=0?f(n*BigInt(10)**BigInt(e)):f(n,BigInt(10)**BigInt(-e));
}
const abs=(a:F)=>a.n<BigInt(0)?neg(a):a;
const width=(r:Range)=>sub(r.upper,r.lower);
const includes=(r:Range,v:F)=>le(r.lower,v)&&le(v,r.upper);
const overlaps=(a:Range,b:Range)=>le(a.lower,b.upper)&&le(b.lower,a.upper);
const scale=(a:Range,x:F):Range=>x.n>=BigInt(0)?{lower:mul(a.lower,x),upper:mul(a.upper,x)}:{lower:mul(a.upper,x),upper:mul(a.lower,x)};
function referenceOverlap(r:Range,s:string,digits=240){const v=parse(s),a=abs(v),unit=le(a,one)?one:a,eps=div(unit,f(BigInt(10)**BigInt(digits)));assert(overlaps(r,{lower:sub(v,eps),upper:add(v,eps)}),'reference overlap '+s.slice(0,70));}
let cutComparisons=0,rootComparisons=0,entropyComparisons=0,logarithmComparisons=0;
for(const ref of refs.cases)test(`independent Descartes spectrum and series entropy: ${ref.label}`,()=>{
  for(const cut of ref.expected.cuts){
    const cert=spectrum(cut.characteristicPolynomial.map(parse));assert.equal(cert.isolationBits,4096);assert.equal(cert.roots.length,cut.spectrum.length);
    assert(Object.isFrozen(cert)&&Object.isFrozen(cert.roots));let entropy=point(zero),rank=0;
    cert.roots.forEach((r,i)=>{
      const target=cut.spectrum[i];assert(Object.isFrozen(r)&&Object.isFrozen(r.lower));assert.equal(r.multiplicity,target.multiplicity);
      assert(le(zero,r.lower)&&le(r.upper,one)&&le(width(r),f(BigInt(1),BigInt(1)<<BigInt(4096))));
      if(target.weight.startsWith('decimal:'))referenceOverlap(r,target.weight);else assert(includes(r,parse(target.weight)));
      assert.equal(eq(r.lower,zero)&&eq(r.upper,zero),target.zeroWeight);
      if(!target.zeroWeight){rank+=r.multiplicity;assert(r.lower.n>BigInt(0));const lo=logarithm(r.lower),hi=logarithm(r.upper),log={lower:lo.lower,upper:hi.upper};entropy=radd(entropy,scale(rmul(r,log),int(-r.multiplicity)));}
      if(i>0)assert(le(r.upper,cert.roots[i-1].lower)&&!eq(r.upper,cert.roots[i-1].lower));rootComparisons++;
    });
    assert.equal(rank,cut.schmidtRank);referenceOverlap(entropy,cut.entropyNats);assert(le(width(entropy),f(BigInt(1),BigInt(10)**BigInt(440))));
    const ln2=logarithm(int(2)),bits=rmul(entropy,{lower:div(one,ln2.upper),upper:div(one,ln2.lower)});referenceOverlap(bits,cut.entropyBits);entropyComparisons+=2;cutComparisons++;
  }
});

test('individual natural logarithms match independently generated 520-digit values',()=>{
  for(const row of logs.cases){const x=f(BigInt(row.numerator),BigInt(row.denominator)),r=logarithm(x);assert(Object.isFrozen(r)&&Object.isFrozen(r.lower));referenceOverlap(r,row.logarithm,450);assert(le(width(r),f(BigInt(1),BigInt(10)**BigInt(450))));logarithmComparisons++;}
  assert(eq(logarithm(one).lower,zero)&&eq(logarithm(one).upper,zero));
});

function polynomial(roots:{value:F;multiplicity:number}[]):F[]{let a=[one];for(const r of roots)for(let j=0;j<r.multiplicity;j++){const next=Array.from({length:a.length+1},()=>zero);a.forEach((v,i)=>{next[i]=add(next[i],v);next[i+1]=sub(next[i+1],mul(v,r.value));});a=next;}return a;}
test('exact multiplicities survive repeated non-dyadic rational roots and endpoints',()=>{
  for(const expected of [[{value:f(BigInt(1),BigInt(3)),multiplicity:3},{value:zero,multiplicity:1}],[{value:f(BigInt(2),BigInt(7)),multiplicity:2},{value:f(BigInt(1),BigInt(7)),multiplicity:2}],[{value:one,multiplicity:1},{value:zero,multiplicity:3}]]){
    const cert=spectrum(polynomial(expected));assert.equal(cert.roots.length,expected.length);cert.roots.forEach((r,i)=>{assert.equal(r.multiplicity,expected[i].multiplicity);assert(includes(r,expected[i].value));});
  }
});

test('nearby rational roots and tiny positive roots remain distinct from degeneracy and zero',()=>{
  const d=BigInt(10)**BigInt(180),expected=[{value:f(d+BigInt(1),BigInt(2)*d),multiplicity:1},{value:f(d-BigInt(1),BigInt(2)*d),multiplicity:1},{value:f(BigInt(1),BigInt(10)**BigInt(220)),multiplicity:1},{value:zero,multiplicity:1}];
  const cert=spectrum(polynomial(expected));assert.equal(cert.roots.length,4);cert.roots.forEach((r,i)=>{assert.equal(r.multiplicity,1);assert(includes(r,expected[i].value));});assert(cert.roots[2].lower.n>BigInt(0));
});

test('quartic irrational spectra have independently checkable exact quadratic enclosures',()=>{
  // Product of x²-x+1/5 and x²-x+1/6 has four distinct roots in (0,1).
  const cert=spectrum([one,int(-2),f(BigInt(41),BigInt(30)),f(BigInt(-11),BigInt(30)),f(BigInt(1),BigInt(30))]);assert.equal(cert.roots.length,4);
  const discriminants=[f(BigInt(1),BigInt(3)),f(BigInt(1),BigInt(5)),f(BigInt(1),BigInt(5)),f(BigInt(1),BigInt(3))];
  cert.roots.forEach((r,i)=>{const a=sub(mul(int(2),r.lower),one),b=sub(mul(int(2),r.upper),one),lower=i<2?mul(a,a):mul(b,b),upper=i<2?mul(b,b):mul(a,a);assert(le(lower,discriminants[i])&&le(discriminants[i],upper));});
});

test('invalid spectra and logarithm arguments are rejected within explicit budgets',()=>{
  for(const p of [[],[one],[zero,one],[one,int(-2)],[one,one],[one,zero,one],[one,int(-1),one],[one,{n:BigInt(1),d:BigInt(0)}],[one,f(BigInt(1)<<BigInt(16385))]])assert.throws(()=>spectrum(p));
  for(const x of [zero,int(-1),{n:BigInt(1),d:BigInt(0)},{n:BigInt(1),d:BigInt(-1)},f(BigInt(1)<<BigInt(16385))])assert.throws(()=>logarithm(x));
});

test('logarithm bounds respect inversion, dyadic scaling and exact monotonicity',()=>{
  for(const x of [f(BigInt(1),BigInt(1000000)),f(BigInt(3),BigInt(7)),f(BigInt(11),BigInt(3)),int(1000000)]){
    const a=logarithm(x),b=logarithm(div(one,x));assert(includes(radd(a,b),zero));
    const twice=logarithm(mul(int(2),x));assert(overlaps(twice,radd(a,logarithm(int(2)))));
    if(le(x,one))assert(le(a.upper,zero));else assert(le(zero,a.lower));
  }
  const d=BigInt(10)**BigInt(400),x=f(d+BigInt(1),d),r=logarithm(x),epsilon=sub(x,one);
  // Integral of 1/t bounds log(1+epsilon) between epsilon/(1+epsilon) and epsilon.
  assert(overlaps(r,{lower:div(epsilon,x),upper:epsilon}));assert(r.lower.n>BigInt(0));
});

test('root cache snapshots are immutable, canonical and bounded without numerical engines',async()=>{
  const p=[one,int(-1),f(BigInt(1),BigInt(5))],a=spectrum(p);assert.equal(spectrum(p.map(v=>mul(v,int(7)))),a);
  assert.throws(()=>{(a.roots as any).push({});});assert.equal(spectrum(p).roots.length,2);
  for(let i=2;i<72;i++)spectrum([one,f(BigInt(-1),BigInt(i))]);assert.deepEqual(spectrum(p),a);
  for(let i=1;i<270;i++)logarithm(f(BigInt(i+1),BigInt(270)));assert(includes(logarithm(one),zero));
  const code=await build({entryPoints:['src/lib/learning-many-body-schmidt-spectral-wire.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p=>/learning-(schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|ritz-interval|interval-transcendentals|many-body-evaluator)\.ts$/.test(p)),inputs.join('\n'));
  writeFileSync(`${runtime}/spectral-dependency-graph.json`,JSON.stringify({inputs,bytes:code.outputFiles[0].contents.length,productionNumericalEnginesExcluded:true},null,2)+'\n');
});
after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({referencePreparations:refs.cases.length,cutComparisons,rootComparisons,entropyComparisons,logarithmComparisons},null,2)+'\n'));
