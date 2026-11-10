import test from 'node:test';import assert from 'node:assert/strict';import{readFileSync,mkdirSync,writeFileSync}from'node:fs';
import{tensorPureModel as model,isTensorPureParameters as valid,type TensorPureParameters as P}from'../src/lib/learning-tensor-pure';
import{tensorPureKeys as keys}from'../src/lib/learning-tensor-pure-input';
import{isMasterObservation}from'../src/lib/learning-master-equation-protocol';
import type{PartitionObservation as Q}from'../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/tensor-pure-reference.json','utf8'))as{cases:{label:string;input:P;expected:any}[]};
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


for(const c of refs.cases)test('independent tensor/partial-trace reference: '+c.label,()=>{
 const x=model(c.input);compare(x,c.expected,c.label);zeros(x.exactResiduals,true);zeros(x.directedResiduals,false);zeros(x.projectorTests.conditionAOnB.noSignallingResidual,true);zeros(x.projectorTests.conditionBOnA.noSignallingResidual,true);assert.equal(x.schmidtSpectrum.reduce((s,v)=>s+v.multiplicity,0),x.schmidtRank);assert.equal(x.partialTranspose.zeroMultiplicity+x.schmidtRank*x.schmidtRank,x.jointDimension);assert.doesNotThrow(()=>JSON.stringify(x));
});
const input=(label:string)=>refs.cases.find(c=>c.label===label)!.input,M=(label:string)=>model(input(label));
function near(actual:number|null,expected:number){assert.notEqual(actual,null);assert(Math.abs(actual!-expected)<1e-12,`${actual} != ${expected}`);}
test('Bell phase choices retain local I/2 while the joint off-diagonal distinguishes them from classical mixtures',()=>{
 for(const label of ['Bell plus','Bell minus','Bell imaginary']){const x=M(label);assert.equal(x.schmidtRank,2);near(x.reducedA.density[0][0].real.value,.5);near(x.reducedB.density[1][1].real.value,.5);near(x.entropyBits.value,1);near(x.partialTranspose.negativity.value,.5);near(x.partialTranspose.logNegativityBits.value,1);near(x.basisDephasedReference.purity.value,.5);assert(x.basisDephasedReference.sameReducedStates);assert(x.basisDephasedReference.separableByProductBasisMixture);near(x.basisDephasedReference.connectedProbability[0][0].value,.25);assert(x.entanglementWitnessMinor);}
 near(M('Bell imaginary').jointDensity[0][3].imaginary.value,-.5);near(M('Bell plus').jointDensity[0][3].real.value,.5);near(M('Bell minus').jointDensity[0][3].real.value,-.5);
});
test('complex B marginal uses the transpose of C dagger C,not its untransposed imaginary sign',()=>{
 const x=M('complex B sign');assert.equal(x.schmidtRank,1);near(x.reducedB.density[0][1].imaginary.value,-.5);near(x.coefficientGramB[0][1].imaginary.value,.5);assert(!x.classification.entangledPureState);near(x.reducedA.purity.value,1);assert(x.productFactorization);assert.equal(x.entanglementWitnessMinor,null);near(x.entropyBits.value,0);
});
test('rank-three and rank-four states are calculated rather than restricted to binary Schmidt weights',()=>{
 for(const d of [3,4]){const x=M(`${d}x${d} maximal Schmidt`);assert.equal(x.schmidtRank,d);assert.equal(x.schmidtSpectrum.length,1);assert.equal(x.schmidtSpectrum[0].multiplicity,d);near(x.schmidtSpectrum[0].weight.value,1/d);near(x.reducedA.purity.value,1/d);near(x.entropyBits.value,Math.log2(d));near(x.partialTranspose.negativity.value,(d-1)/2);near(x.partialTranspose.logNegativityBits.value,Math.log2(d));assert.equal(x.partialTranspose.pairs.length,d*(d-1)/2);assert.equal(x.partialTranspose.zeroMultiplicity,0);}
 const rectangular=M('3x4 maximal Schmidt');assert.equal(rectangular.zeroMultiplicityA,0);assert.equal(rectangular.zeroMultiplicityB,1);assert.equal(rectangular.partialTranspose.zeroMultiplicity,3);
});
test('all16 local dimension pairs retain product factorization and pure local states',()=>{
 for(let a=1;a<=4;a++)for(let b=1;b<=4;b++){const x=M(`${a}x${b} complex product`);assert.equal(x.jointDimension,a*b);assert.equal(x.schmidtRank,1);assert(x.productFactorization);near(x.productFactorization.normProduct.value,x.rawNormSquared.value!);assert.equal(x.entanglementWitnessMinor,null);near(x.reducedA.purity.value,1);near(x.reducedB.purity.value,1);near(x.partialTranspose.negativity.value,0);assert.equal(x.partialTranspose.pairs.length,0);}
});
test('a tiny nonzero minor and a small fourth Schmidt weight retain rank and positive entropy',()=>{
 const x=M('tiny entanglement');assert.equal(x.schmidtRank,2);assert(x.entanglementWitnessMinor);assert.equal(x.localLinearEntropy.sign,1);assert.equal(x.entropyBits.sign,1);assert.equal(x.partialTranspose.negativity.sign,1);assert(x.schmidtSpectrum[0].weight.value!>0&&x.schmidtSpectrum[0].weight.value!<1e-20);
 const q=M('tiny fourth Schmidt');assert.equal(q.schmidtRank,4);assert.equal(q.schmidtSpectrum.reduce((s,v)=>s+v.multiplicity,0),4);assert(q.schmidtSpectrum[0].weight.value!>0);assert(q.schmidtSpectrum[0].weight.value!<1e-10);
});
test('local conditioning preserves no-signalling and distinguishes zero probabilities from mixed complements',()=>{
 const zero=M('zero outcome').projectorTests.conditionAOnB.rows[0];assert.equal(zero.probability.sign,0);assert.equal(zero.conditionalDensity,null);assert.equal(zero.conditionalPurity,null);
 const bell=M('Bell plus');for(const r of bell.projectorTests.conditionAOnB.rows){near(r.probability.value,.5);near(r.conditionalPurity!.value,1);}assert.deepEqual(bell.projectorTests.conditionAOnB.nonselectiveDensity,bell.reducedA.density);
 const x=M('rank two complement conditional').projectorTests.conditionAOnB;near(x.rows[0].conditionalPurity!.value,1);assert(x.rows[1].conditionalPurity!.value!<1);assert(x.rows[1].probability.value!>0);
});
test('global representative phase changes coefficients but preserves reduced states and Schmidt invariants',()=>{
 const x=M('unequal Schmidt');for(const label of ['global raw sign','global raw phase']){const y=M(label);assert.deepEqual(x.jointDensity,y.jointDensity);assert.deepEqual(x.reducedA,y.reducedA);assert.deepEqual(x.reducedB,y.reducedB);assert.deepEqual(x.schmidtSpectrum,y.schmidtSpectrum);assert.deepEqual(x.partialTranspose,y.partialTranspose);assert.notDeepEqual(x.rawCoefficients,y.rawCoefficients);}
});
test('local Householder actions satisfy tensor indexing and preserve local spectra without changing Schmidt rank',()=>{
 const x=M('complex rank four');assert.notDeepEqual(x.reducedA.density,x.localReflection.transformedReducedA);assert.deepEqual(x.reducedA.purity,x.localReflection.transformedPurityA);assert.deepEqual(x.reducedB.purity,x.localReflection.transformedPurityB);zeros(x.exactResiduals.reflectionTensorAction,true);zeros(x.exactResiduals.reflectionCharacteristicA,true);zeros(x.exactResiduals.reflectionCharacteristicB,true);
});
test('50 owned bounded integers reject inactive state/probe entries instead of truncating the tensor',()=>{
 const p=input('Bell plus');assert.equal(keys.length,50);assert(valid(p));for(const key of keys){for(const bad of [NaN,Infinity,1.5,1000001,-1000001]){assert(!valid({...p,[key]:bad}));assert.throws(()=>model({...p,[key]:bad}),RangeError);}const q={...p};delete q[key];assert(!valid(q));}
 const zeroState={...p};for(const k of keys)if(k.startsWith('state'))zeroState[k]=0;const zeroProbe={...p};for(const k of keys)if(k.startsWith('aProbe'))zeroProbe[k]=0;
 for(const bad of [null,[],Object.create(p),{...p,dimensionA:0},{...p,dimensionB:5},{...p,stateReal33:1},{...p,bProbeImag3:1},zeroState,zeroProbe,{...p,extra:1},{...p,[Symbol('foreign')]:1}])assert(!valid(bad));const before=structuredClone(p);assert(Object.isFrozen(model(p).parameters));assert.deepEqual(p,before);
});
test('explicit product factors reconstruct every coefficient in A-major order',()=>{
 const read=(q:Q)=>endpoint(q.bounds!.lower),times=(a:F,b:F)=>fraction(a.n*b.n,a.d*b.d),plus=(a:F,b:F)=>fraction(a.n*b.d+b.n*a.d,a.d*b.d),minus=(a:F,b:F)=>fraction(a.n*b.d-b.n*a.d,a.d*b.d);
 for(let a=1;a<=4;a++)for(let b=1;b<=4;b++){
  const x=M(`${a}x${b} complex product`),factors=x.productFactorization!;for(let i=0;i<a;i++)for(let j=0;j<b;j++){
   const u=factors.rawA[i],v=factors.rawB[j],re=minus(times(read(u.real),read(v.real)),times(read(u.imaginary),read(v.imaginary))),im=plus(times(read(u.real),read(v.imaginary)),times(read(u.imaginary),read(v.real)));
   assert(eq(re,read(x.rawCoefficients[i][j].real)));assert(eq(im,read(x.rawCoefficients[i][j].imaginary)));
   const na=factors.normalizedA[i],nb=factors.normalizedB[j],z=x.normalizedAmplitudes[i*b+j];near(na.real.value!*nb.real.value!-na.imaginary.value!*nb.imaginary.value!,z.real.value!);near(na.real.value!*nb.imaginary.value!+na.imaginary.value!*nb.real.value!,z.imaginary.value!);
  }
 }
});
test('reference inventory covers all dimension pairs,nonbinary rank and directed spectral precision',()=>{
 assert(exactChecks>20000&&directedChecks>500);const dimensions=new Set(refs.cases.map(c=>`${c.input.dimensionA}x${c.input.dimensionB}`));assert.equal(dimensions.size,16);mkdirSync('.runtime/learning/quantum-tensor-pure',{recursive:true});writeFileSync('.runtime/learning/quantum-tensor-pure/reference-checks.json',JSON.stringify({preparations:refs.cases.length,uniquePreparations:new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,dimensionPairs:16,exactRationalCertificates:exactChecks,directedObservationChecks:directedChecks,maxCoefficientDigits,maximumRelativeWidthAgainstMaxOneReference:1e-125},null,2)+'\n');
});
