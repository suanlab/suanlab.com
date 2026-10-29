import assert from 'node:assert/strict';
import test from 'node:test';
import { evaluateElementary as evaluate, elementaryKinds, gcd } from '../src/lib/learning-elementary';
import { elementaryLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const vector=(a:number[],b:number[])=>{assert.equal(a.length,b.length);a.forEach((x,i)=>close(x,b[i]));};
test('exact fraction reduction, equivalent representations and SI conversion',()=>{
 assert.equal(gcd(-4,6),2);const r=evaluate('numbers-ratios-units',[1,3,150,2]);vector(r.stats['reduced (numerator,denominator)'] as number[],[1,3]);close(r.stats['L [m]'] as number,1.5);close(r.stats['v=L/t [m/s]'] as number,.75);assert.deepEqual(r.curves,evaluate('numbers-ratios-units',[2,6,150,2]).curves);
});
test('zero numerator, zero denominator and zero elapsed time have different meanings',()=>{
 const z=evaluate('numbers-ratios-units',[0,3,0,2]);vector(z.stats['reduced (numerator,denominator)'] as number[],[0,1]);close(z.stats['a/b (≈)'] as number,0);const bad=evaluate('numbers-ratios-units',[1,0,150,0]);assert.equal(bad.stats['a/b (≈)'],null);assert.equal(bad.stats['v=L/t [m/s]'],null);assert.equal(bad.warning,'zero-divisor');assert.deepEqual(bad.curves,[]);
});
for(const [a,b,theta] of [[1,1,90],[0,1,180],[3,-2,-60],[-1,0,0]])test(`complex rotation preserves magnitude: ${a},${b},${theta}`,()=>{
 const r=evaluate('real-complex-numbers',[a,b,theta]);close(r.stats['|z|'] as number,Math.hypot(a,b));close(r.stats['|w|'] as number,Math.hypot(a,b));if(a===1&&b===1)vector(r.stats['w (Re,Im)'] as number[],[-1,1]);
});
test('zero complex number has magnitude zero and no argument',()=>{
 const r=evaluate('real-complex-numbers',[0,0,90]);close(r.stats['|z|'] as number,0);assert.equal(r.stats['arg(z) [°]'],null);assert.equal(r.warning,'undefined-angle');
});
test('linear equations separate unique, absent and all-real solutions',()=>{
 vector(evaluate('equations-inequalities',[0,2,4,1]).stats['real solutions'] as number[],[2]);assert.deepEqual(evaluate('equations-inequalities',[0,0,1,0]).stats['real solutions'],[]);const all=evaluate('equations-inequalities',[0,0,0,1]);assert.equal(all.stats['real solution count'],Infinity);assert.equal(all.stats['real solutions'],null);assert.equal(all.warning,'degenerate-equation');
});
test('quadratic inequality matches completed-square and sign-interval references',()=>{
 const r=evaluate('equations-inequalities',[1,-3,2,1.5]);vector(r.stats['real roots'] as number[],[1,2]);vector(r.stats['solution interval'] as number[],[1,2]);close(r.stats['f(x₀)'] as number,-.25);assert.deepEqual(evaluate('equations-inequalities',[1,0,1,0]).stats['real roots'],[]);vector(evaluate('equations-inequalities',[1,-2,1,1]).stats['solution interval'] as number[],[1,1]);
});
test('cancellation preserves a hole while a polynomial remains defined there',()=>{
 const r=evaluate('polynomial-rational-functions',[0,0,1]);assert.equal(r.stats['f(x₀)'],null);close(r.stats['lim x→1 f(x)'] as number,2);assert.deepEqual(r.holes,[[1,2]]);assert.ok(r.curves[0].every(([x])=>x<1));assert.ok(r.curves[1].every(([x])=>x>1));close(evaluate('polynomial-rational-functions',[2,1,1]).stats['f(x₀)'] as number,0);
});
for(const pole of [-2,0,2])test(`rational branches never connect across a pole: ${pole}`,()=>{
 const r=evaluate('polynomial-rational-functions',[1,pole,pole+.1]);close(r.stats['f(x₀)'] as number,10);assert.ok(r.curves[0].every(([x,y])=>x<pole&&y<0));assert.ok(r.curves[1].every(([x,y])=>x>pole&&y>0));assert.equal(evaluate('polynomial-rational-functions',[1,pole,pole]).stats['f(x₀)'],null);
});
test('log inverse conditions include decreasing bases and exclude base one and input zero',()=>{
 const r=evaluate('exponentials-logarithms',[2,2,4]);close(r.stats['aˣ₀'] as number,4);close(r.stats['logₐ(y₀)'] as number,2);close(r.stats['d(aˣ)/dx at x₀'] as number,4*Math.log(2));close(evaluate('exponentials-logarithms',[.5,2,.25]).stats['logₐ(y₀)'] as number,2);assert.equal(evaluate('exponentials-logarithms',[1,2,1]).stats['logₐ(y₀)'],null);assert.equal(evaluate('exponentials-logarithms',[2,2,0]).warning,'function-domain');
});
test('degree/radian conversion, Euclidean distance and wrap produce independent exact references',()=>{
 const r=evaluate('trigonometry-coordinates',[60,2]);close(r.stats['θ [rad]'] as number,Math.PI/3);vector(r.stats['p=(r cosθ,r sinθ)'] as number[],[1,Math.sqrt(3)]);close(r.stats['‖p−q‖₂'] as number,Math.sqrt(3));vector(evaluate('trigonometry-coordinates',[-300,2]).stats['p=(r cosθ,r sinθ)'] as number[],r.stats['p=(r cosθ,r sinθ)'] as number[]);assert.equal(evaluate('trigonometry-coordinates',[60,0]).stats['arg(p) [°]'],null);
});
test('composition order, two preimages and a restricted inverse are distinct',()=>{
 const r=evaluate('composition-inverse-functions',[2,1,1]);close(r.stats['h(x₀)'] as number,9);close(r.stats['g(f(x₀))'] as number,3);vector(r.stats['h⁻¹({h(x₀)})'] as number[],[-2,1]);close(r.stats['inverse with g(x)≥0'] as number,1);const outside=evaluate('composition-inverse-functions',[2,0,-1]);close(outside.stats['1{g(x₀)≥0}'] as number,0);close(outside.stats['inverse with g(x)≥0'] as number,1);
});
test('composition branch direction handles negative scale and coinciding roots',()=>{
 const r=evaluate('composition-inverse-functions',[-2,1,0]);vector(r.stats['h⁻¹({h(x₀)})'] as number[],[0,1]);close(r.stats['inverse with g(x)≥0'] as number,0);const zero=evaluate('composition-inverse-functions',[2,0,0]);assert.equal(zero.stats['preimage count'],1);const constant=evaluate('composition-inverse-functions',[0,1,1]);assert.equal(constant.stats['preimage count'],Infinity);assert.equal(constant.stats['inverse with g(x)≥0'],null);
});
test('geometric partial sums, tail, empty sum and exceptional zero series',()=>{
 const r=evaluate('sequences-series',[.5,1,10]);close(r.stats.S_N as number,1.998046875);close(r.stats['lim S_k'] as number,2);close(2-(r.stats.S_N as number),1/512);const empty=evaluate('sequences-series',[.5,1,0]);close(empty.stats.S_N as number,0);assert.deepEqual(empty.curves[0],[[0,0]]);close(evaluate('sequences-series',[1.5,0,10]).stats['lim S_k'] as number,0);
});
test('bounded oscillation and finite prefixes do not imply infinite convergence',()=>{
 const r=evaluate('sequences-series',[-1,1,10]);vector(r.curves[0].slice(0,5).map(v=>v[1]),[0,1,0,1,0]);assert.equal(r.stats['lim S_k'],null);assert.equal(r.warning,'infinite-sum');assert.equal(evaluate('sequences-series',[1.1,1,10]).stats['lim S_k'],null);
});
test('Big-O candidate certificates distinguish threshold failure from class failure',()=>{
 const early=evaluate('asymptotic-growth',[2,3,3,1]);assert.equal(early.stats['required n₀ for this C'],3);assert.equal(early.stats['1{bound holds ∀n≥n₀}'],0);const valid=evaluate('asymptotic-growth',[2,3,3,3]);assert.equal(valid.stats['1{bound holds ∀n≥n₀}'],1);const bad=evaluate('asymptotic-growth',[2,3,2,3]);assert.equal(bad.stats['required n₀ for this C'],null);assert.equal(bad.warning,'asymptotic-bound');assert.equal(evaluate('asymptotic-growth',[2,0,2,1]).stats['1{bound holds ∀n≥n₀}'],1);
});
test('overflow is not mislabeled as mathematical infinity or a domain exclusion',()=>{
 const pole=evaluate('polynomial-rational-functions',[1,0,1e-310]);assert.equal(pole.stats['f(x₀)'],null);assert.equal(pole.warning,'numeric-range');const rate=evaluate('numbers-ratios-units',[1,3,150,1e-310]);assert.equal(rate.stats['v=L/t [m/s]'],null);assert.equal(rate.warning,'numeric-range');
});
for(const kind of elementaryKinds)test(`elementary guides and valid finite drawings: ${kind}`,()=>{
 const spec=elementaryLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const preset of spec.presets){assert.equal(preset.values.length,spec.controls.length);preset.values.forEach((x,i)=>assert.ok(x>=spec.controls[i].min&&x<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.holes?.flat().every(Number.isFinite)??true);}learningGuides[kind].experiments.forEach(g=>assert.ok(spec.presets[g.preset]));
});
test('malformed, nonfinite, out-of-domain and fractional mode inputs are rejected',()=>{
 assert.throws(()=>evaluate('real-complex-numbers',[1,2]));assert.throws(()=>evaluate('real-complex-numbers',[1,2,NaN]));assert.throws(()=>evaluate('numbers-ratios-units',[1,-1,1,1]));assert.throws(()=>evaluate('equations-inequalities',[.5,1,1,1]));assert.throws(()=>evaluate('exponentials-logarithms',[0,1,1]));assert.throws(()=>evaluate('sequences-series',[.5,1,1.5]));assert.throws(()=>evaluate('asymptotic-growth',[.2,3,3,1]));
});
