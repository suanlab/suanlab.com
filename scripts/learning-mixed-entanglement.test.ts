import test from 'node:test';import assert from 'node:assert/strict';import{readFileSync,writeFileSync}from'node:fs';
import{mixedEntanglementModel as model,isMixedEntanglementParameters as valid,type MixedEntanglementParameters as P}from'../src/lib/learning-mixed-entanglement';
import{mixedEntanglementKeys as keys}from'../src/lib/learning-mixed-entanglement-input';
import{tensorPureModel}from'../src/lib/learning-tensor-pure';
import{isMasterObservation}from'../src/lib/learning-master-equation-protocol';
import type{PartitionObservation as Q}from'../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/mixed-entanglement-reference.json','utf8'))as{cases:{label:string;input:P;expected:any}[]};
type F={n:bigint;d:bigint};const fraction=(n:bigint,d=BigInt(1)):F=>({n,d}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,eq=(a:F,b:F)=>a.n*b.d===b.n*a.d;
function parse(s:string):F{if(s.includes('/')){const[n,d]=s.split('/');return fraction(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),k=Number(m[4]||0)-(m[3]||'').length;return k>=0?fraction(n*BigInt(10)**BigInt(k)):fraction(n,BigInt(10)**BigInt(-k));}
const endpoint=(q:NonNullable<Q['bounds']>['lower'])=>fraction(BigInt(q.numerator.coefficient),BigInt(q.denominator.coefficient));let directedChecks=0,exactChecks=0,maxCoefficientDigits=0;
function check(actual:unknown,expected:string,path:string,exact=false){assert(isMasterObservation(actual),path+' shape');const q=actual as Q;assert(q.bounds,path);const lo=endpoint(q.bounds.lower),hi=endpoint(q.bounds.upper);assert(le(lo,hi),path+' order');maxCoefficientDigits=Math.max(maxCoefficientDigits,...[q.bounds.lower,q.bounds.upper].flatMap(b=>[b.numerator.coefficient.length,b.denominator.coefficient.length]));
 const decimal=expected.startsWith('decimal:'),x=parse(decimal?expected.slice(8):expected),absolute=x.n<0?-x.n:x.n,scale=absolute>x.d?absolute:x.d;
 if(decimal){const eps=BigInt(10)**BigInt(200),lower=fraction(x.n*eps-scale,x.d*eps),upper=fraction(x.n*eps+scale,x.d*eps);assert(le(lo,upper)&&le(lower,hi),path+' reference '+expected);}
 else{assert(le(lo,x)&&le(x,hi),path+' contains '+expected);if(exact){assert(eq(lo,x)&&eq(hi,x),path+' exact');exactChecks++;}}
 assert(le(fraction(hi.n*lo.d-lo.n*hi.d,hi.d*lo.d),fraction(scale,x.d*BigInt(10)**BigInt(125))),path+' width');if(!exact)directedChecks++;
}
function compare(actual:unknown,expected:unknown,path='result'){
 if(typeof expected==='string'&&(/^-?\d/.test(expected)||expected.startsWith('decimal:'))){const exact=!expected.startsWith('decimal:');check(actual,expected,path,exact);return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,path+'.'+i));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,path+'.'+k);return;}assert.equal(actual,expected,path);
}


function zeros(value:unknown,exact:boolean,path='residual'){if(value===null)return;if(isMasterObservation(value)){check(value,'0',path,exact);if(exact){assert(isMasterObservation(value,true));assert.equal((value as Q).sign,0,path);}return;}if(Array.isArray(value)){value.forEach((v,i)=>zeros(v,exact,path+'.'+i));return;}assert(value&&typeof value==='object',path);for(const[k,v]of Object.entries(value))zeros(v,exact,path+'.'+k);}

for(const c of refs.cases)test('independent general mixed two-qubit reference: '+c.label,()=>{
 const x=model(c.input);compare(x,c.expected,c.label);zeros(x.exactResiduals,true);
 for(const row of x.localTests.pairs)for(const pair of row)zeros(pair.residuals,true);
 for(const row of x.localTests.conditionals)zeros(row.noSignallingResidual,true);
 for(const rows of [x.joint.spectrum,x.partialTranspose.spectrum,x.chsh.gramSpectrum])assert.equal(rows.reduce((s,r)=>s+r.multiplicity,0),rows===x.chsh.gramSpectrum?3:4);
 assert.equal(x.reducedA.spectrum.reduce((s,r)=>s+r.multiplicity,0),2);assert.equal(x.reducedB.spectrum.reduce((s,r)=>s+r.multiplicity,0),2);
 assert(!x.chsh.selectedViolates||x.chsh.optimizedViolates);assert(!x.chsh.optimizedViolates||x.classification.entangledTwoQubit);
 assert.equal(x.partialTranspose.negativeMultiplicity,x.classification.entangledTwoQubit?1:0);
 assert.doesNotThrow(()=>JSON.stringify(x));
});
const input=(label:string)=>refs.cases.find(c=>c.label===label)!.input,M=(label:string)=>model(input(label));
function near(actual:number|null,expected:number){assert.notEqual(actual,null);assert(Math.abs(actual!-expected)<1e-12,`${actual} != ${expected}`);}

test('same maximally mixed marginals distinguish Bell,classical correlation and independent states',()=>{
 const bell=M('Bell plus poor settings'),classical=M('classical correlated'),independent=M('independent maximally mixed');
 for(const x of [bell,classical,independent]){near(x.reducedA.purity.value,.5);near(x.reducedB.purity.value,.5);assert.deepEqual(x.reducedA.density,bell.reducedA.density);assert.deepEqual(x.reducedB.density,bell.reducedB.density);}
 assert(bell.classification.entangledTwoQubit);assert(classical.classification.separableTwoQubit);assert(independent.classification.separableTwoQubit);
 near(bell.mutualInformationBits.value,2);near(classical.mutualInformationBits.value,1);check(independent.mutualInformationBits,'0','independent mutual information');
 near(bell.conditionalEntropyAGivenBNats.value,-Math.log(2));check(classical.conditionalEntropyAGivenBNats,'0','classical conditional entropy');near(independent.conditionalEntropyAGivenBNats.value,Math.log(2));
 near(bell.partialTranspose.negativity.value,.5);near(classical.partialTranspose.negativity.value,0);near(independent.partialTranspose.negativity.value,0);
});
test('selected settings can miss violation even when optimized Bell value is Tsirelson',()=>{
 const poor=M('Bell plus poor settings'),good=M('Bell plus violating settings'),negative=M('Bell plus negative violation');
 assert(!poor.chsh.selectedViolates);assert(good.chsh.selectedViolates);assert(negative.chsh.selectedViolates);assert(negative.chsh.selectedSigned.value!< -2);
 for(const x of [poor,good,negative]){assert(x.chsh.optimizedViolates);near(x.chsh.optimizedMaximum.value,2*Math.sqrt(2));assert(!x.chsh.optimalAxesComputed);}
 assert.deepEqual(poor.joint.density,good.joint.density);assert.deepEqual(good.joint.density,negative.joint.density);
});
test('Werner PPT and optimized CHSH thresholds differ without claiming all nonviolating states local',()=>{
 const below=M('Werner 99999/300000'),edge=M('Werner 1/3'),above=M('Werner 100001/300000');
 assert(below.classification.separableTwoQubit);assert(edge.classification.separableTwoQubit);assert(above.classification.entangledTwoQubit);
 assert(edge.partialTranspose.spectrum.some(r=>r.sign===0));assert(!above.chsh.optimizedViolates);
 const half=M('Werner 1/2');assert(half.classification.entangledTwoQubit);assert(!half.chsh.optimizedViolates);near(half.partialTranspose.negativity.value,1/8);assert(!half.classification.noChshViolationProvesLocalHiddenVariableModel);
 assert(!M('Werner 353553/500000').chsh.optimizedViolates);assert(M('Werner 353554/500000').chsh.optimizedViolates);
});
test('exact shifted-Gram certificates resolve entangled boundary,perturbations and repeated minima',()=>{
 for(const label of ['entangled exact CHSH boundary','rotated entangled CHSH boundary']){
  const x=M(label);assert(x.classification.entangledTwoQubit);assert.equal(x.chsh.optimizedComparisonToLocalBound,0);assert(x.chsh.exactBoundaryCertificate.positiveSemidefinite);check(x.chsh.exactBoundaryCertificate.determinant,'0',label,true);near(x.chsh.optimizedMaximum.value,2);
 }
 assert.equal(M('CHSH boundary perturbation -1').chsh.optimizedComparisonToLocalBound,-1);assert.equal(M('CHSH boundary perturbation 1').chsh.optimizedComparisonToLocalBound,1);
 assert.equal(M('pure product zero outcomes').chsh.optimizedComparisonToLocalBound,0);assert.equal(M('classical correlated').chsh.optimizedComparisonToLocalBound,0);assert.equal(M('independent maximally mixed').chsh.optimizedComparisonToLocalBound,-1);
});
test('complex Pauli Y sign and both partial traces reconstruct a general state exactly',()=>{
 const x=M('complex product');near(x.reducedA.blochVector[1].value,1);near(x.reducedA.blochVector[0].value,0);near(x.reducedA.density[0][1].imaginary.value,-.5);zeros(x.exactResiduals.pauliReconstruction,true);
 for(const c of refs.cases.filter(c=>c.label.startsWith('general complex'))){const y=model(c.input);assert(y.rawHermitianNumerator.flat().some(z=>z.imaginary.sign!==0));assert.equal(y.joint.rank,Number(/rank(\d)/.exec(c.label)![1]));}
});
test('nonselective remote marginals do not depend on local setting and zero events have null states',()=>{
 const x=M('pure product zero outcomes'),a=x.localTests.conditionals[0];assert.equal(a.rows[1].probability.sign,0);assert.equal(a.rows[1].conditionalDensity,null);assert.equal(a.rows[1].conditionalPurity,null);
 for(const y of [x,M('Bell plus violating settings'),M('general complex rank4 case0')]){
  assert.deepEqual(y.localTests.conditionals[0].nonselectiveDensity,y.reducedB.density);assert.deepEqual(y.localTests.conditionals[1].nonselectiveDensity,y.reducedB.density);
  assert.deepEqual(y.localTests.conditionals[2].nonselectiveDensity,y.reducedA.density);assert.deepEqual(y.localTests.conditionals[3].nonselectiveDensity,y.reducedA.density);
  for(let i=0;i<2;i++)assert.deepEqual(y.localTests.pairs[i][0].marginalA,y.localTests.pairs[i][1].marginalA);
  for(let j=0;j<2;j++)assert.deepEqual(y.localTests.pairs[0][j].marginalB,y.localTests.pairs[1][j].marginalB);
 }
});
test('normalization and local representative rescaling preserve physical statistics',()=>{
 const a=M('Bell plus violating settings');
 for(const label of ['scaled Bell numerator','scaled Bell local spinors']){const b=M(label);assert.deepEqual(a.joint,b.joint);assert.deepEqual(a.partialTranspose,b.partialTranspose);assert.deepEqual(a.localTests,b.localTests);assert.deepEqual(a.chsh,b.chsh);}
});
test('all15 principal minors are required;leading zeros cannot hide a negative direction',()=>{
 const p=input('pure product zero outcomes'),bad={...p,hReal00:1,hReal11:0,hReal22:-1,hReal33:1};assert(!valid(bad));assert.throws(()=>model(bad),RangeError);
 const negativeOffDiagonal={...p,hReal00:1,hReal11:1,hReal22:1,hReal33:1,hReal03:2};assert(!valid(negativeOffDiagonal));assert.throws(()=>model(negativeOffDiagonal),RangeError);
 for(const label of ['pure product zero outcomes','classical correlated','Werner 1/3'])assert(valid(input(label)));
});
test('32 owned bounded integer inputs reject invalid states,zero probes and foreign fields',()=>{
 const p=input('Bell plus poor settings');assert.equal(keys.length,32);assert.equal(new Set(keys).size,32);assert(valid(p));
 for(const key of keys){for(const bad of [NaN,Infinity,1.5,1000001,-1000001]){assert(!valid({...p,[key]:bad}));assert.throws(()=>model({...p,[key]:bad}),RangeError);}const missing={...p};delete missing[key];assert(!valid(missing));}
 const zero=Object.fromEntries(keys.map(k=>[k,0])),zeroProbe={...p,a0Real0:0,a0Imag0:0,a0Real1:0,a0Imag1:0};
 for(const bad of [null,[],Object.create(p),zero,zeroProbe,{...p,extra:1},{...p,[Symbol('foreign')]:1}])assert(!valid(bad));
 const before=structuredClone(p);const x=model(p);assert(Object.isFrozen(x.parameters));assert.deepEqual(p,before);assert.deepEqual(Object.keys(x.parameters),keys);
 const negativeZero={...p,hImag01:-0};assert(!Object.is(model(negativeZero).parameters.hImag01,-0));
 assert(valid({...input('pure product zero outcomes'),hReal00:1000000,a0Real0:1000000}));
});
test('pure two-qubit density,partial traces and negativity agree across both calculation families',()=>{
 const pureRefs=JSON.parse(readFileSync('scripts/fixtures/tensor-pure-reference.json','utf8'))as{cases:{label:string;input:Record<string,number>}[]};
 for(const label of ['Bell plus','Bell imaginary','unequal Schmidt','complex B sign']){
  const p=pureRefs.cases.find(c=>c.label===label)!.input,pure=tensorPureModel(p),mixed={...input('Bell plus poor settings')};
  for(let i=0;i<4;i++)for(let j=i;j<4;j++){
   const z=pure.jointDensity[i][j],N=pure.rawNormSquared.value!;mixed[`hReal${i}${j}`]=Math.round(z.real.value!*N);if(i!==j)mixed[`hImag${i}${j}`]=Math.round(z.imaginary.value!*N);
  }
  const x=model(mixed);assert.deepEqual(x.joint.density,pure.jointDensity);assert.deepEqual(x.reducedA.density,pure.reducedA.density);assert.deepEqual(x.reducedB.density,pure.reducedB.density);
  for(const[a,b]of [[x.partialTranspose.negativity,pure.partialTranspose.negativity],[x.partialTranspose.logNegativityBits,pure.partialTranspose.logNegativityBits],[x.reducedA.entropyBits,pure.entropyBits]])assert(le(endpoint(a.bounds!.lower),endpoint(b.bounds!.upper))&&le(endpoint(b.bounds!.lower),endpoint(a.bounds!.upper)));
 }
});
test('a bounded ill-conditioned numerator retains its tiny fourth positive eigenvalue',()=>{
 const x=M('ill-conditioned full rank bounded numerator');assert.equal(x.joint.rank,4);assert(x.joint.spectrum.every(r=>r.sign===1));assert(x.joint.spectrum[0].eigenvalue.value!>0);assert(x.joint.spectrum[0].eigenvalue.value!<1e-23);
});
test('independent inventory covers ranks1..4,complex mixtures,PPT/CHSH boundaries and directed precision',()=>{
 assert.equal(refs.cases.length,53);assert(exactChecks>20000&&directedChecks>1000);assert.equal(new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,53);
 writeFileSync('.runtime/learning/quantum-mixed-entanglement/reference-checks.json',JSON.stringify({preparations:refs.cases.length,uniquePreparations:new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,exactRationalCertificates:exactChecks,directedObservationChecks:directedChecks,maxCoefficientDigits,maximumRelativeWidthAgainstMaxOneReference:1e-125},null,2)+'\n');
});


