import assert from 'node:assert/strict';
import test from 'node:test';
import { evaluateFoundation as evaluate, foundationKinds, svd2 } from '../src/lib/learning-foundations';
import { foundationLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const vector=(a:number[],b:number[])=>{assert.equal(a.length,b.length);a.forEach((x,i)=>close(x,b[i]));};
for(const matrix of [[3,0,0,1],[-3,0,0,1],[2,2,1,1],[0,0,0,0],[3,1,0,2],[0,1,-2,0],[1,1,0,1e-8]])test(`SVD reconstruction, orthogonality and rank-one error: ${matrix}`,()=>{
 const [a,b,c,d]=matrix;const s=svd2(a,b,c,d);vector(s.reconstructed,matrix);close(s.error,s.singular[1]);assert.ok(s.singular.every(x=>x>=0));close(s.right[0][0]*s.right[1][0]+s.right[0][1]*s.right[1][1],0);close(s.left[0][0]**2+s.left[0][1]**2,1);close(s.left[1][0]**2+s.left[1][1]**2,1,1e-7);
});
test('SVD numerical references distinguish truncation and condition number',()=>{
 const s=svd2(3,0,0,1);vector(s.singular,[3,1]);close(s.error,1);close(svd2(3,0,0,.1).condition!,30);assert.equal(svd2(2,2,1,1).condition,null);
});
test('binomial reference, normalization and deterministic endpoints',()=>{
 const r=evaluate('probability-distributions',[.5,4]);vector(r.curves[0].map(v=>v[1]),[1/16,4/16,6/16,4/16,1/16]);close(r.stats['E[K]'] as number,2);close(r.stats['Var(K)'] as number,1);
 for(const p of [0,.1,.2,.8,1])for(const count of [1,4,20]){const v=evaluate('probability-distributions',[p,count]);close(v.stats['Σ P(K=k)'] as number,1);assert.ok(v.curves[0].every(v=>v[1]>=0));}
});
test('expectation and population variance are independently checked',()=>{
 const r=evaluate('expectation-variance',[1,2,1]);close(r.stats['E[X]'] as number,.25);close(r.stats['E[X²]'] as number,1.25);close(r.stats['Var(X)'] as number,1.1875);close(evaluate('expectation-variance',[0,0,1]).stats['Var(X)'] as number,0);close(evaluate('expectation-variance',[3,1,1]).stats['Var(X)'] as number,1.36);
});
test('likelihood fixes data and handles impossible and boundary observations',()=>{
 const r=evaluate('statistical-estimation',[.5,10,7]);close(r.stats['p̂MLE'] as number,.7);close(r.stats['ℓ(p)'] as number,10*Math.log(.5));assert.ok((r.stats['ℓ(p̂)'] as number)>(r.stats['ℓ(p)'] as number));close(evaluate('statistical-estimation',[1,10,10]).stats['ℓ(p)'] as number,0);assert.equal(evaluate('statistical-estimation',[0,10,7]).stats['ℓ(p)'],-Infinity);
});
test('beta-Bernoulli update and posterior predictive use exact references',()=>{
 const r=evaluate('bayesian-inference',[1,1]);vector(r.stats['αpost, βpost'] as number[],[8,4]);close(r.stats['E[p|D]'] as number,2/3);close(r.stats['Var(p|D)'] as number,2/117);close(evaluate('bayesian-inference',[9,1]).stats['E[p|D]'] as number,.8);close(evaluate('bayesian-inference',[1,9]).stats['E[p|D]'] as number,.4);
 const curve=r.curves[1];const area=curve.slice(1).reduce((sum,v,i)=>sum+(v[0]-curve[i][0])*(v[1]+curve[i][1])/2,0);close(area,1,1e-4);
});
test('entropy decomposition, matching distributions and support mismatch',()=>{
 const r=evaluate('information-theory',[.75,.5]);close(r.stats['H(P) [nat]'] as number,-.75*Math.log(.75)-.25*Math.log(.25));close(r.stats['H(P,Q) [nat]'] as number,Math.log(2));close(evaluate('information-theory',[.75,.75]).stats['D_KL(P‖Q) [nat]'] as number,0);assert.equal(evaluate('information-theory',[.75,0]).stats['H(P,Q) [nat]'],Infinity);close(evaluate('information-theory',[0,0]).stats['H(P,Q) [nat]'] as number,0);close(evaluate('information-theory',[1,1]).stats['H(P,Q) [nat]'] as number,0);
});
test('ridge stabilizes weak direction and labels a minimum-norm choice',()=>{
 vector(evaluate('regularization',[0,.1,1]).stats['wλ'] as number[],[1,10]);vector(evaluate('regularization',[.1,.1,1]).stats['wλ'] as number[],[90/91,10/11]);const r=evaluate('regularization',[0,0,1]);vector(r.stats['wλ'] as number[],[1,0]);assert.equal(r.warning,'rank');
});
test('sampled optimization is reproducible, seed-dependent and has correct mean gradient',()=>{
 const r=evaluate('stochastic-optimization',[.1,4,123]);assert.deepEqual(r,evaluate('stochastic-optimization',[.1,4,123]));assert.notDeepEqual(r.curves[1],evaluate('stochastic-optimization',[.1,4,321]).curves[1]);close(r.curves[0][1][1],3.7);close(r.stats['E[gᵢ(w₀)]'] as number,3);close(r.stats['Var(gᵢ)'] as number,8/3);
});
test('Euler distinguishes exact decay, finite-step error and unstable oscillation',()=>{
 const r=evaluate('differential-equations',[1,.25,1]);close(r.stats['xexact(T)'] as number,Math.exp(-5));close(r.stats['xEuler(T)'] as number,.75**20);assert.equal(evaluate('differential-equations',[3,1,1]).warning,'stability');close(evaluate('differential-equations',[3,1,1]).curves[1][1][1],-2);
 // Fixed final time refinement, independent of the lab's fixed step count.
 const error=(h:number)=>Math.abs((1-h)**Math.round(1/h)-Math.exp(-1));assert.ok(error(.1)/error(.05)>1.9&&error(.1)/error(.05)<2.2);
});
test('velocity-first oscillator ordering and energy error are explicit',()=>{
 const r=evaluate('mechanics-energy',[.1,1,1]);close(r.curves[1][1][1],.99);close(r.stats['E₀ [J]'] as number,.5);assert.ok((r.stats['max |E−E₀| [J]'] as number)>0);assert.equal(evaluate('mechanics-energy',[1,4,1]).warning,'stability');
});
test('partition function, energy, entropy and degeneracy agree independently',()=>{
 const r=evaluate('thermodynamics',[1,1,1]);close(r.stats.Z as number,1+Math.exp(-1));close(r.stats.U as number,1/(1+Math.E));close(r.stats['S/kB'] as number,(r.stats.U as number)-(r.stats.F as number));
 const d=evaluate('thermodynamics',[1,1,3]);close((d.stats['S/kB'] as number)-(d.stats['H(binary occupancy)'] as number),(d.stats['P(E=Δ)'] as number)*Math.log(3));
});
test('Markov stationary existence, periodicity and nonuniqueness remain distinct',()=>{
 const r=evaluate('stochastic-processes',[.2,.3,1]);close(r.stats['π₁'] as number,.4);close(r.curves[0][1][1],.7);close(r.curves[0][2][1],.55);const periodic=evaluate('stochastic-processes',[1,1,1]);vector(periodic.curves[0].slice(0,5).map(v=>v[1]),[1,0,1,0,1]);close(periodic.stats['π₁'] as number,.5);const none=evaluate('stochastic-processes',[0,0,.7]);assert.equal(none.stats['π₁'],null);assert.ok(none.curves[0].every(v=>v[1]===.7));
});
test('inverse-problem amplification, stronger observation and missing contrast',()=>{
 vector(evaluate('inverse-problems',[.1,.1,0]).stats['θ̂'] as number[],[1.5,1.5]);vector(evaluate('inverse-problems',[1,.1,0]).stats['θ̂'] as number[],[1.05,1.95]);vector(evaluate('inverse-problems',[.5,0,0]).stats['θ̂'] as number[],[1,2]);const r=evaluate('inverse-problems',[0,0,0]);assert.equal(r.warning,'rank');assert.equal(r.stats['κ₂(X)'],null);vector(r.curves[1].map(v=>v[1]),[1,1]);
});
test('paired uncertainty uses sample denominator and Student-t rather than a normal critical value',()=>{
 const r=evaluate('research-methods',[.1,.1]);close(r.stats['SE(mean)'] as number,.1/Math.sqrt(6));const ci=r.stats['95% t CI (df=3)'] as number[];assert.ok(ci[0]<0&&ci[1]>0);close(ci[0]+ci[1],.2);close((ci[1]-.1)/(.1/Math.sqrt(6)),3.182446305);assert.equal(r.warning,'uncertainty');
});
for(const kind of foundationKinds)test(`all reference and guided presets produce finite plotted coordinates: ${kind}`,()=>{
 const spec=foundationLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(spec.presets.length,3);assert.equal(learningGuides[kind].experiments.length,3);
 for(const preset of spec.presets){assert.equal(preset.values.length,spec.controls.length);preset.values.forEach((x,i)=>assert.ok(x>=spec.controls[i].min&&x<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.curves.every(curve=>curve.length>=2));}
});
test('invalid shapes and nonfinite inputs are rejected',()=>{
 assert.throws(()=>evaluate('svd',[1,2]));assert.throws(()=>evaluate('svd',[1,2,3,Infinity]));assert.throws(()=>evaluate('probability-distributions',[-1,4]));assert.throws(()=>evaluate('statistical-estimation',[.5,10,11]));assert.throws(()=>evaluate('bayesian-inference',[0,1]));
});

test('cold canonical occupancy retains a representable tiny probability',()=>{
 const r=evaluate('thermodynamics',[4,.05,1]);const probability=r.stats['P(E=Δ)'] as number;assert.ok(probability>0);assert.ok(Math.abs(probability/Math.exp(-80)-1)<1e-12);close((r.stats['S/kB'] as number)/(r.stats['H(binary occupancy)'] as number),1);
});
