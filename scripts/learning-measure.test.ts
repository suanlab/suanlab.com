import test from 'node:test';
import assert from 'node:assert/strict';
import { measureKinds, evaluateMeasure as evaluate, finiteAtoms, partitionExpectation } from '../src/lib/learning-measure';
import { measureLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('finite sigma-algebras are closed under complement and unions and reject split atoms',()=>{
 for(let mode=0;mode<3;mode++){
  const atoms=finiteAtoms(mode),events=evaluate('sigma-algebras-measures',[mode,0]).stats['measurable event masks'] as number[];assert.equal(events.length,2**atoms.length);
  for(const A of events){assert.ok(events.includes(15^A));for(const B of events)assert.ok(events.includes(A|B));}
  for(let A=0;A<16;A++){const r=evaluate('sigma-algebras-measures',[mode,A]);assert.equal(r.stats['1{event measurable}'],Number(events.includes(A)));if(!events.includes(A))assert.equal(r.stats['P(event)'],null);}
 }
});
test('Lebesgue simple integrals agree with independent midpoint integration and monotone error bounds',()=>{
 let previous=-1;for(let m=0;m<=4;m++){const r=evaluate('lebesgue-simple-functions',[m,.6]),integral=Number(r.stats['simple integral']);assert.ok(integral>=previous);assert.ok(1/3-integral<=2**(-m)+1e-12);previous=integral;let numeric=0;const N=40000,M=2**m;for(let i=0;i<N;i++)numeric+=Math.floor(M*((i+.5)/N)**2)/M/N;close(integral,numeric,2e-5);}
 const endpoint=evaluate('lebesgue-simple-functions',[0,1]);assert.equal(endpoint.stats['simple value at query'],1);assert.equal(endpoint.stats['simple integral'],0);
});
test('null-set changes alter sup differences while leaving integrals and L1 equivalence unchanged',()=>{
 for(let a=-3;a<=3;a++)for(let b=-3;b<=3;b++){const r=evaluate('almost-sure-null-sets',[a,b]);assert.equal(r.stats['P(f=g)'],1);assert.equal(r.stats['L1 difference'],0);assert.equal(r.stats['sup difference'],Math.max(Math.abs(a),Math.abs(b)));assert.equal(r.stats['1{equal everywhere}'],Number(a===0&&b===0));}
});
test('convergence examples distinguish domination, concentration and monotone limits at endpoints',()=>{
 for(const n of [1,2,4,8,16,32]){
  const d=evaluate('convergence-dominated-integration',[0,n,1]);assert.equal(d.stats['pointwise limit at query'],1);close(Number(d.stats['integral f_n']),1/(n+1));
  const spike=evaluate('convergence-dominated-integration',[1,n,1/n]);assert.equal(spike.stats['f_n(query)'],0);assert.equal(spike.stats['integral f_n'],1);assert.equal(spike.stats['integral of limit'],0);assert.equal(spike.stats['1{integrable domination}'],0);
  const increasing=evaluate('convergence-dominated-integration',[2,n,.25]);close(Number(increasing.stats['integral f_n']),2-1/n);assert.equal(increasing.stats['1{monotone increasing}'],1);assert.equal(increasing.stats['1{integrable domination}'],1);
 }
});
test('monotone truncated singular integrals agree with numerical integration and dominate every plotted point',()=>{
 for(const n of [1,2,4,8]){let numeric=0;const N=50000;for(let i=0;i<N;i++)numeric+=Math.min(n,1/Math.sqrt((i+.5)/N))/N;close(numeric,2-1/n,1e-4);for(const x of [.001,.1,.5,1]){const r=evaluate('convergence-dominated-integration',[2,n,x]);assert.ok(Number(r.stats['f_n(query)'])<=Number(r.stats['pointwise limit at query']));}}
});
test('finite Fubini rectangles commute while different infinite iterated sums and nonnegative infinity remain explicit',()=>{
 for(let N=1;N<=8;N++)for(let M=1;M<=8;M++)for(const mode of [0,1]){
  const r=evaluate('fubini-tonelli-series',[N,M,mode]);assert.equal(r.stats['finite sum by rows'],r.stats['finite sum by columns']);assert.equal(r.stats['finite sum by rows'],Math.min(N,M)+(mode?1:-1)*Math.min(N,M-1));assert.equal(r.stats['infinite absolute sum'],Infinity);
  assert.equal(r.stats['sum of infinite row sums'],mode?Infinity:0);assert.equal(r.stats['sum of infinite column sums'],mode?Infinity:1);
 }
});
test('pushforward laws aggregate outcomes, discard zero-mass support points and preserve centered moments',()=>{
 for(let weights=0;weights<2;weights++)for(let map=0;map<2;map++)for(let scale=-2;scale<=2;scale++){
  const r=evaluate('probability-measures-expectations',[weights,map,scale]),masses=r.stats['pushforward masses'] as number[];close(masses.reduce((s,v)=>s+v,0),1);assert.ok(masses.every(v=>v>0));close(Number(r.stats['E_P[X]']),Number(r.stats['integral via pushforward']));const mean=Number(r.stats['E_P[X]']),variance=r.table.rows.reduce((s,row)=>s+Number(row[1])*(Number(row[2])-mean)**2,0);close(variance,Number(r.stats['Var_P(X)']));
 }
 assert.deepEqual(evaluate('probability-measures-expectations',[1,0,1]).stats['support values'],[0]);
});
test('conditional expectations preserve every information-event integral and implement the tower property',()=>{
 const weights=[.1,.2,.3,.4],X=[0,2,1,3];for(let mode=0;mode<3;mode++){
  const atoms=finiteAtoms(mode),Y=partitionExpectation(X,weights,atoms).values;for(let mask=0;mask<2**atoms.length;mask++){const event=atoms.flatMap((atom,j)=>(mask>>j)&1?atom:[]);close(event.reduce((s,i)=>s+weights[i]*X[i],0),event.reduce((s,i)=>s+weights[i]*Y[i],0));}
  partitionExpectation(Y,weights,atoms).values.forEach((v,i)=>close(v,Y[i]));const coarse=partitionExpectation(Y,weights,[[0,1,2,3]]).values;coarse.forEach(v=>close(v,1.9));
 }
 const zero=partitionExpectation(X,[.5,.5,0,0],finiteAtoms(0));assert.equal(zero.nullAtoms,1);assert.deepEqual(zero.values,[1,1,0,0]);
});
test('conditional L2 projection has the Pythagorean decomposition for alternative information predictors',()=>{
 const w=[.1,.2,.3,.4],X=[0,2,1,3],Y=partitionExpectation(X,w,finiteAtoms(0)).values,Z=[-1,-1,3,3],sum=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+w[i]*(v-b[i])**2,0);close(sum(X,Z),sum(X,Y)+sum(Y,Z));
});
test('RN ratios change expectations only under absolute continuity and never fabricate missing support',()=>{
 for(const mode of [0,2]){const r=evaluate('radon-nikodym-derivatives',[mode]);assert.equal(r.stats['1{Q absolutely continuous w.r.t. P}'],1);close(Number(r.stats['integral RN dP']),1);close(Number(r.stats['E_Q[φ]']),Number(r.stats['E_P[RN·φ]']));}
 const fail=evaluate('radon-nikodym-derivatives',[1]);assert.equal(fail.stats['RN chosen version'],null);assert.equal(fail.stats['integral RN dP'],null);assert.equal(fail.stats['E_P[RN·φ]'],null);assert.equal(fail.stats['Q mass on P-null points'],.25);
});
test('equal laws need not converge in probability on a fixed coupling and discontinuous CDF limits are permitted',()=>{
 for(let n=1;n<=16;n++){
  const alternating=evaluate('weak-probability-convergence',[1,n,.5]);assert.equal(alternating.stats['CDF sup distance to target law'],0);assert.equal(alternating.stats['P(|X_n−target|>ε)'],n%2);assert.equal(alternating.stats['E|X_n−target|'],2*(n%2));
  const point=evaluate('weak-probability-convergence',[3,n,.5]);assert.equal(point.stats['1{weak convergence of full sequence}'],1);assert.equal(point.stats['CDF sup distance to target law'],1);assert.equal(point.stats['P(|X_n−target|>ε)'],Number(n<2));
  const spike=evaluate('weak-probability-convergence',[2,n,.5]);assert.equal(spike.stats['E|X_n−target|'],1);assert.equal(spike.stats['1{probability convergence to target}'],1);assert.equal(spike.stats['1{L1 convergence to target}'],0);
 }
});
test('differentiated expectations agree for a smooth objective and fail for the moving indicator despite a.e. derivatives',()=>{
 for(const theta of [.25,.5,.75])for(let k=1;k<=8;k++){
  const smooth=evaluate('interchanging-probability-integrals',[0,theta,k]),h=1e-5,numeric=(((theta+h)**2/3)-((theta-h)**2/3))/(2*h);close(Number(smooth.stats['derivative of expectation']),numeric,1e-9);close(Number(smooth.stats['expectation of a.e. derivative']),numeric,1e-9);
  const spike=evaluate('interchanging-probability-integrals',[1,theta,k]);close(Number(spike.stats['difference-quotient width'])*Number(spike.stats['difference-quotient peak']),1);assert.equal(spike.stats['derivative of expectation'],1);assert.equal(spike.stats['expectation of a.e. derivative'],0);assert.equal(spike.stats['expectation of centered quotient'],1);
 }
});
test('partition helpers reject overlaps, empty atoms and zero total weight',()=>{
 assert.throws(()=>partitionExpectation([1,2],[1,1],[[0],[0]]));assert.throws(()=>partitionExpectation([1,2],[0,0],[[0,1]]));assert.throws(()=>partitionExpectation([1,2],[1,1],[[],[0,1]]));assert.throws(()=>finiteAtoms(.5));
});
for(const kind of measureKinds)test(`measure presets, finite geometry and input domains: ${kind}`,()=>{
 const spec=measureLabs[kind];assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.holes?.flat().every(Number.isFinite)??true);assert.ok(r.dots?.flat().every(Number.isFinite)??true);assert.ok(r.table.rows.length>0);}
 assert.throws(()=>evaluate(kind,[]));const p=spec.controls.map(c=>c.initial);p[0]=Infinity;assert.throws(()=>evaluate(kind,p));p[0]=spec.controls[0].min-.5;assert.throws(()=>evaluate(kind,p));
});
