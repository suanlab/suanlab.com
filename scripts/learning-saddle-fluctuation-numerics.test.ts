import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { Rational as Q, Z, O, T, sum } from '../src/lib/learning-graph-gauge-validation-math';
import { readSaddleFluctuationInput, saddleFluctuationDefault, saddleFluctuationExamples, type SaddleFluctuationInput } from '../src/lib/learning-saddle-fluctuation-input';
import { calculateSaddleFluctuation, type SaddleRange } from '../src/lib/learning-saddle-fluctuation';
import { snapshotQubitWire } from '../src/lib/learning-qubit-wire';

interface ChainReference {
  kind: 'chain';
  stationary: { point: string[]; errorUpper: string; exact: boolean; action: SaddleRange; actionExact: string | null };
  gaussian: { normalizedDiagonal: string[]; normalizedDiagonalErrorUpper: string; normalizedPivots: string[]; normalizedPivotErrorUpper: string; inverseNormalized: string[][]; inverseNormalizedErrorUpper: string; determinantNormalized: SaddleRange; prefactorSquaredTimesPi: SaddleRange };
  correction: Record<'cubic'|'quartic'|'first'|'gaussianFourthSum'|'gaussianFourthSumSquared', SaddleRange>;
  ratio: SaddleRange | null;
}
interface WellReference {
  kind: 'well'; regime: string; positiveRoot: SaddleRange | null; minimumAction: string;
  firstCoefficient: string | null; sumPrefactorSquaredDividedByPi: string | null; criticalQuarticWidthScale: SaddleRange;
}
const fixture = JSON.parse(readFileSync('scripts/fixtures/saddle-fluctuation-references.json','utf8')) as { referenceBits: number; cases: { id: string; input: SaddleFluctuationInput; expected: ChainReference | WellReference }[] };
const integer = (n: number) => new Q(BigInt(n)), square = (x: Q) => x.mul(x), abs = (x: Q) => x.n < BigInt(0) ? x.neg() : x;
const compare = (a: Q, b: Q) => a.n*b.d < b.n*a.d ? -1 : a.n*b.d > b.n*a.d ? 1 : 0;
const max = (a: Q, b: Q) => compare(a,b)>0?a:b, min = (a: Q, b: Q) => compare(a,b)<0?a:b;
const point = (s: string) => new Q(BigInt(s), BigInt(1)<<BigInt(fixture.referenceBits));
function encloses(actual: SaddleRange, lower: Q, upper = lower, label = '') {
  const lo=Q.read(actual.lower),hi=Q.read(actual.upper);
  assert(compare(lo,hi)<=0,'unordered '+label);
  assert(compare(lo,lower)<=0 && compare(hi,upper)>=0,'independent enclosure not contained: '+label);
}
const reference = (actual: SaddleRange, expected: SaddleRange, label = '') => encloses(actual,Q.read(expected.lower),Q.read(expected.upper),label);
const root = '.runtime/learning/saddle-fluctuation-numerics';mkdirSync(root,{recursive:true});
const measurements: {id:string;milliseconds:number;bytes:number;strings:number}[]=[];
function stringCount(value: unknown): number {
  if(typeof value==='string') return value.length;
  if(Array.isArray(value)) return value.reduce((n,x)=>n+stringCount(x),0);
  return value&&typeof value==='object'?Object.values(value).reduce<number>((n,x)=>n+stringCount(x),0):0;
}

for(const item of fixture.cases) test('independent certified full reference: '+item.id,()=>{
  const started=performance.now(),r=calculateSaddleFluctuation(item.input),expected=item.expected;
  assert.deepEqual(r.input,item.input);assert(Object.isFrozen(r.input));
  const snapshot=snapshotQubitWire(r);assert.deepEqual(snapshot,r);
  const bytes=Buffer.byteLength(JSON.stringify(r)),strings=stringCount(r);
  assert(strings<=2000000);measurements.push({id:item.id,milliseconds:performance.now()-started,bytes,strings});
  writeFileSync(root+'/measurements-node'+process.versions.node.split('.')[0]+'.json',JSON.stringify(measurements,null,2)+'\n');
  if(r.kind==='well' && expected.kind==='well') {
    assert.equal(r.regime,expected.regime);assert.equal(r.minimumAction,expected.minimumAction);assert.equal(r.firstCoefficient,expected.firstCoefficient);assert.equal(r.sumPrefactorSquaredDividedByPi,expected.sumPrefactorSquaredDividedByPi);
    reference(r.criticalQuarticWidthScale,expected.criticalQuarticWidthScale,'critical quarter scale');
    if(expected.positiveRoot){reference(r.stationaryPoints[2].position,expected.positiveRoot,'positive minimum');encloses(r.stationaryPoints[0].position,Q.read(expected.positiveRoot.upper).neg(),Q.read(expected.positiveRoot.lower).neg(),'negative minimum');assert.equal(r.minimaIncluded,2);assert.equal(r.stationaryPoints[1].classification,'maximum');assert.equal(r.stationaryPoints[1].gaussianVariance,null);}
    else{assert.equal(r.minimaIncluded,1);assert.equal(r.gaussianAvailable,r.regime!=='degenerate-quartic');}
    const hbar=Q.read(r.input.hbar),kappa=Q.read(r.input.kappa);assert.equal(r.hbarFirst,r.firstCoefficient?Q.read(r.firstCoefficient).mul(hbar).text():null);assert.equal(r.smallParameter,kappa.n===BigInt(0)?null:hbar.div(square(kappa)).text());for(const p of r.stationaryPoints){if(p.classification==='maximum'){assert.equal(p.action,'0');assert.equal(p.curvature,kappa.neg().text());assert.equal(p.gaussianVariance,null);}else{assert.equal(p.action,expected.minimumAction);assert.equal(p.curvature,kappa.n>BigInt(0)?kappa.mul(T).text():kappa.neg().text());assert.equal(p.gaussianVariance,kappa.n===BigInt(0)?null:hbar.div(Q.read(p.curvature)).text());}}assert.equal(r.scope.integralEvaluated,false);assert.equal(r.scope.fullQuantumDoubleWellComputed,false);return;
  }
  assert(r.kind==='chain'&&expected.kind==='chain');
  const input=r.input,n=input.intervals,d=n-1,delta=Q.read(input.time).div(integer(n)),mass=Q.read(input.mass),hbar=Q.read(input.hbar),omega=Q.read(input.omegaSquared),lam=Q.read(input.quartic),start=Q.read(input.start),end=Q.read(input.end),s=mass.div(delta);
  const radius=Q.read(expected.stationary.errorUpper),pointPath=expected.stationary.point.map(Q.read);
  assert.equal(r.stationary.position.length,n+1);assert.equal(r.gaussian.hessian.length,d);assert.equal(r.gaussian.inverseCurvature.length,n+1);assert.equal(r.gaussian.covariance.length,n+1);
  for(let i=0;i<d;i++)encloses(r.stationary.position[i+1],pointPath[i].sub(radius),pointPath[i].add(radius),'root '+i);
  encloses(r.stationary.position[0],start);encloses(r.stationary.position[n],end);
  if(expected.stationary.actionExact)encloses(r.stationary.minimumAction,Q.read(expected.stationary.actionExact),Q.read(expected.stationary.actionExact),'exact action');
  else reference(r.stationary.minimumAction,expected.stationary.action,'minimum action');
  const hdError=Q.read(expected.gaussian.normalizedDiagonalErrorUpper),pivotError=Q.read(expected.gaussian.normalizedPivotErrorUpper),inverseError=Q.read(expected.gaussian.inverseNormalizedErrorUpper);
  for(let i=0;i<d;i++){
    const v=point(expected.gaussian.normalizedPivots[i]);encloses(r.gaussian.normalizedPivots[i],v.sub(pivotError),v.add(pivotError),'normalized pivot '+i);
    for(let j=0;j<d;j++){
      if(i===j){const v=point(expected.gaussian.normalizedDiagonal[i]);encloses(r.gaussian.hessian[i][j],v.sub(hdError).mul(s),v.add(hdError).mul(s),'H diagonal');}
      else encloses(r.gaussian.hessian[i][j],Math.abs(i-j)===1?s.neg():Z);
    }
  }
  for(let i=0;i<=n;i++)for(let j=0;j<=n;j++){
    if(i===0||j===0||i===n||j===n){assert.deepEqual(r.gaussian.inverseCurvature[i][j],{lower:'0',upper:'0'});assert.deepEqual(r.gaussian.covariance[i][j],{lower:'0',upper:'0'});}
    else{const v=point(expected.gaussian.inverseNormalized[i-1][j-1]),lower=max(Z,v.sub(inverseError)).div(s),upper=v.add(inverseError).div(s);encloses(r.gaussian.inverseCurvature[i][j],lower,upper,'R');encloses(r.gaussian.covariance[i][j],lower.mul(hbar),upper.mul(hbar),'hbar R');assert.deepEqual(r.gaussian.inverseCurvature[i][j],r.gaussian.inverseCurvature[j][i]);assert.deepEqual(r.gaussian.covariance[i][j],r.gaussian.covariance[j][i]);}
  }
  assert.equal(r.gaussian.determinant.scale,s.text());assert.equal(r.gaussian.determinant.power,d);
  reference(r.gaussian.determinant.normalized,expected.gaussian.determinantNormalized,'factored determinant');reference(r.gaussian.prefactorSquaredTimesPi,expected.gaussian.prefactorSquaredTimesPi,'Gaussian prefactor');
  for(const key of ['cubic','quartic','first','gaussianFourthSum','gaussianFourthSumSquared'] as const)reference(r.correction[key],expected.correction[key],key);
  encloses(r.correction.hbarFirst,Q.read(expected.correction.first.lower).mul(hbar),Q.read(expected.correction.first.upper).mul(hbar),'hbar c1');
  if(expected.ratio){assert(r.ratio.lower!==null&&r.ratio.upper!==null);assert(compare(Q.read(r.ratio.lower),Q.read(expected.ratio.lower))<=0);assert(compare(Q.read(r.ratio.upper),Q.read(expected.ratio.upper))>=0);}
  else{assert.equal(r.ratio.status,'not-applicable-shifted-cubic');assert.equal(r.ratio.lower,null);assert.equal(r.ratio.upper,null);}
  // Independent arithmetic reconstructs the exact claimed candidate certificate.
  const q=r.stationary.candidate.slice(1,-1).map(Q.read),path=[start,...q,end];
  const g=q.map((v,i)=>s.mul(v.mul(T).sub(path[i]).sub(path[i+2])).add(delta.mul(mass.mul(omega).mul(v).add(lam.mul(v).mul(v).mul(v)))));
  assert.deepEqual(r.stationary.gradient,g.map(x=>x.text()));const rho=g.map(abs).reduce(max,Z),mu=mass.div(delta.mul(integer(n*n))).add(delta.mul(mass).mul(omega));
  assert.equal(r.stationary.residualMaximum,rho.text());assert.equal(r.stationary.curvatureLowerBound,mu.text());assert.equal(r.stationary.positionErrorUpper,integer(d).mul(rho).div(mu).text());assert.equal(r.stationary.actionErrorUpper,integer(d).mul(square(rho)).div(T.mul(mu)).text());
  const potential=(x:Q)=>mass.mul(omega).mul(square(x)).div(T).add(lam.mul(square(square(x))).div(integer(4)));
  const action=s.mul(sum(path.slice(1).map((x,i)=>square(x.sub(path[i]))))).div(T).add(delta.mul(sum(q.map(potential)).add(potential(start).add(potential(end)).div(T))));
  assert.equal(r.stationary.candidateAction,action.text());assert.equal(r.stationary.minimumAction.upper,action.text());
  const e=Q.read(r.stationary.positionErrorUpper),eta=integer(3).mul(delta).mul(lam).mul(e).mul(q.map(abs).reduce(max,Z).mul(T).add(e));
  assert.equal(r.stationary.hessianErrorUpper,eta.text());assert.equal(r.stationary.inverseErrorUpper,eta.div(square(mu)).text());
  assert.equal(r.scope.generalQuarticIntegralEvaluated,false);assert.equal(Object.hasOwn(r.scope,'higherOrderErrorBoundComputed'),false);assert.equal(r.scope.generalHigherOrderErrorBoundComputed,false);
});

test('all32 named teaching inputs retain their full legal spellings and fixed-dimensional models',()=>{
  assert.equal(saddleFluctuationExamples.length,32);assert.equal(new Set(saddleFluctuationExamples.map(x=>x.id)).size,32);
  for(const example of saddleFluctuationExamples){assert.deepEqual(readSaddleFluctuationInput(example.input),example.input);assert(Object.isFrozen(example.input));}
  const exact=readSaddleFluctuationInput({...saddleFluctuationDefault,start:'-999999999983/999999999989',hbar:'2/2'});assert(exact&&exact.mode==='chain');assert.equal(exact.start.length,26);assert.equal(exact.hbar,'2/2');
});
for(const value of ['0','-0','-0/7','1/0','01','1/01','1.0','1e0',' 1','1 ','+1','NaN','Infinity','1/2/3','9999999999999/1000000000000','17'])test('invalid positive scalar is rejected: '+JSON.stringify(value),()=>{
  assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,hbar:value}),null);assert.throws(()=>calculateSaddleFluctuation({...saddleFluctuationDefault,hbar:value}),RangeError);
});
for(const value of [1,33,2.1,NaN,Infinity,-0,'2','2.0000000000000001'])test('interval grammar rejects '+String(value),()=>assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,intervals:value}),null));
for(const value of [-1,3,0.2,NaN,Infinity,-0,'1'])test('probe grammar rejects '+String(value),()=>assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,probe:value}),null));
test('input ownership rejects mixed modes,getters,symbols and inherited fields without evaluating them',()=>{
  let calls=0;const source={...saddleFluctuationDefault},owned=readSaddleFluctuationInput(source);assert(owned);source.end='7';assert.equal(owned.mode==='chain'&&owned.end,'37/32');assert(Object.isFrozen(owned));
  assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,extra:1}),null);assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,[Symbol('extra')]:1}),null);
  assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,get quartic(){calls++;return '1';}}),null);
  assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,get mode(){calls++;return 'chain';}}),null);
  assert.equal(readSaddleFluctuationInput(Object.create(saddleFluctuationDefault)),null);
  assert.equal(readSaddleFluctuationInput({mode:'well',hbar:'1',kappa:'0',intervals:2}),null);assert.equal(readSaddleFluctuationInput({mode:'well',hbar:'1',kappa:'-0'}),null);assert.equal(readSaddleFluctuationInput({...saddleFluctuationDefault,quartic:'-1'}),null);assert.equal(calls,0);
});
test('quadratic tiny-mass determinant stays factored and the symmetric nonzero-endpoint zero saddle receives a valid ratio bound',()=>{
  const c=fixture.cases.find(c=>c.id==='tiny-mass-free-determinant')!;const r=calculateSaddleFluctuation(c.input);assert(r.kind==='chain');encloses(r.gaussian.determinant.normalized,integer(32));assert.equal(r.ratio.status,'exact-quadratic');assert.equal(r.scope.stationaryMeanInterpretation,'exact-Gaussian-mean');
  const opposite=calculateSaddleFluctuation(saddleFluctuationExamples.find(x=>x.id==='opposite-endpoints')!.input);assert(opposite.kind==='chain');assert.equal(opposite.stationary.residualMaximum,'0');assert.equal(opposite.ratio.status,'certified-zero-saddle');assert.equal(opposite.scope.stationaryMeanInterpretation,'zero-mean-by-inversion-symmetry');
  const broad=calculateSaddleFluctuation(fixture.cases.find(c=>c.id==='zero-saddle-strong-quartic')!.input);assert(broad.kind==='chain');assert.deepEqual([broad.ratio.lower,broad.ratio.upper],['0','1']);
});

test('hbar changes reference covariance without moving the stationary path or its Hessian',()=>{
  const a=calculateSaddleFluctuation(saddleFluctuationDefault),b=calculateSaddleFluctuation({...saddleFluctuationDefault,hbar:'2'});assert(a.kind==='chain'&&b.kind==='chain');
  assert.deepEqual(a.stationary,b.stationary);assert.deepEqual(a.gaussian.hessian,b.gaussian.hessian);assert.deepEqual(a.gaussian.inverseCurvature,b.gaussian.inverseCurvature);assert.deepEqual(a.correction.first,b.correction.first);assert.notDeepEqual(a.gaussian.covariance,b.gaussian.covariance);
});
test('positive quartic changes the zero-saddle ratio while retaining the Gaussian Hessian',()=>{
  const a=calculateSaddleFluctuation({...saddleFluctuationDefault,end:'0'}),b=calculateSaddleFluctuation({...saddleFluctuationDefault,end:'0',quartic:'4'});assert(a.kind==='chain'&&b.kind==='chain');
  assert.deepEqual(a.gaussian,b.gaussian);assert.equal(a.stationary.residualMaximum,'0');assert.equal(b.stationary.residualMaximum,'0');assert(a.ratio.lower&&b.ratio.upper);
  assert(compare(Q.read(b.ratio.upper),Q.read(a.ratio.lower))<0);
  const free=calculateSaddleFluctuation({...saddleFluctuationDefault,intervals:32,time:'16',mass:'1/999999999989',omegaSquared:'0',start:'0',end:'0',probe:16});assert(free.kind==='chain');
  const reduction=Q.read(free.input.quartic).mul(Q.read(free.input.time)).mul(Q.read(free.input.time)).mul(Q.read(free.input.time)).div(integer(40).mul(square(Q.read(free.input.mass)))).mul(O.sub(O.div(integer(32**4))));
  encloses(free.correction.first,reduction.neg());
});
test('reader-safe owned input module bundles without a numerical engine or browser globals',async()=>{
  const {build}=await import('esbuild');const result=await build({entryPoints:['src/lib/learning-saddle-fluctuation-input.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',logLevel:'silent'});
  assert.deepEqual(Object.keys(result.metafile!.inputs),['src/lib/learning-saddle-fluctuation-input.ts']);
  assert(!/\b(?:Worker|window|document|BigInt)\b/.test(result.outputFiles[0].text));
});
