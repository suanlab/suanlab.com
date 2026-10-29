import test from 'node:test';
import assert from 'node:assert/strict';
import { complexKinds, evaluateComplex as evaluate, complexMultiply, complexPower, complexReciprocal, circleQuadrature } from '../src/lib/learning-complex';
import { complexLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
test('scaled complex reciprocals avoid avoidable underflow and power maps obey polar magnitude',()=>{
 vector(complexReciprocal([1e-200,0])!,[1e200,0]);assert.equal(complexReciprocal([0,0]),null);
 for(const radius of [0,.5,1,2])for(const angle of [-180,-45,0,45,135])for(let power=1;power<=4;power++){
  const r=evaluate('complex-polar-maps',[radius,angle,power]);close(r.stats['|w|'] as number,radius**power);if(radius===0)assert.equal(r.stats['arg(w) [degrees]'],null);
 }
 vector(complexMultiply([1,1],[1,-1]),[2,0]);assert.throws(()=>complexPower([1,0],-1));
});
test('CR residuals distinguish real differentiability, point complex differentiability and a holomorphic neighborhood',()=>{
 for(const x of [-1,0,1])for(const y of [-1,0,1]){
  const square=evaluate('cauchy-riemann-holomorphic',[0,x,y]);vector(square.stats['Cauchy–Riemann residuals'] as number[],[0,0]);vector(square.stats['complex derivative'] as number[],[2*x,2*y]);assert.equal(square.stats['1{holomorphic near z₀}'],1);
  const conj=evaluate('cauchy-riemann-holomorphic',[1,x,y]);vector(conj.stats['x-direction quotient limit'] as number[],[1,0]);vector(conj.stats['iy-direction quotient limit'] as number[],[-1,0]);assert.equal(conj.stats['complex derivative'],null);
  const norm=evaluate('cauchy-riemann-holomorphic',[2,x,y]);assert.equal(norm.stats['1{holomorphic near z₀}'],0);if(x===0&&y===0)vector(norm.stats['complex derivative'] as number[],[0,0]);else assert.equal(norm.stats['complex derivative'],null);
 }
});
test('circle quadratures match primitives and winding with orientation and domain boundaries',()=>{
 for(const r of [.25,.5,1,2])for(const direction of [-1,1]){
  vector(circleQuadrature(r,0,0,direction)!,[0,0]);vector(circleQuadrature(r,0,1,direction)!,[0,2*Math.PI*direction]);
 }
 assert.equal(circleQuadrature(0,0,1),null);vector(circleQuadrature(0,0,0)!,[0,0]);
});
test('residues are not pole orders and on-contour poles cannot be rescued by finite sampling',()=>{
 for(const pole of [-2,-.5,0,.5,2])for(const power of [1,2,3]){
  const r=evaluate('residues-winding',[pole,1,power]),expected=Math.abs(pole)<1&&power===1?2*Math.PI:0;vector(r.stats['analytic integral (Re,Im)'] as number[],[0,expected]);vector(r.stats['numeric integral (Re,Im)'] as number[],[0,expected]);assert.equal(r.stats['residue at pole'],power===1?1:0);
 }
 const boundary=evaluate('residues-winding',[1,1,1]);assert.equal(boundary.stats['numeric integral (Re,Im)'],null);assert.equal(boundary.stats['analytic integral (Re,Im)'],null);assert.equal(boundary.stats['winding about pole'],null);
});
test('removable values, poles, essential approach directions and representation overflow remain distinct',()=>{
 vector(evaluate('analytic-functions-singularities',[0,0,45]).stats['f(z) (Re,Im)'] as number[],[1,0]);assert.equal(evaluate('analytic-functions-singularities',[1,0,0]).warning,'function-domain');
 const positive=evaluate('analytic-functions-singularities',[2,.1,0]),negative=evaluate('analytic-functions-singularities',[2,.1,180]);close((positive.stats['f(z) (Re,Im)'] as number[])[0],Math.exp(10),1e-8);close((negative.stats['f(z) (Re,Im)'] as number[])[0],Math.exp(-10));
 const overflow=evaluate('analytic-functions-singularities',[2,.001,0]);assert.equal(overflow.stats['f(z) (Re,Im)'],null);assert.equal(overflow.warning,'numeric-range');
 const tiny=evaluate('analytic-functions-singularities',[1,1e-200,0]);assert.equal(tiny.warning,undefined);close((tiny.stats['f(z) (Re,Im)'] as number[])[0],1e200);
});
test('continued logs exponentiate to the same endpoint while roots flip after odd windings',()=>{
 for(const angle of [-120,0,90])for(const radius of [.25,1,2]){
  const original=evaluate('analytic-continuation-branches',[angle,0,radius]).stats['continued sqrt (Re,Im)'] as number[];
  for(let k=-2;k<=2;k++){const r=evaluate('analytic-continuation-branches',[angle,k,radius]);close(r.stats['exp(log) residual'] as number,0);vector(r.stats['continued sqrt (Re,Im)'] as number[],original.map(x=>k%2===0?x:-x));close(r.stats['log increment (Im)'] as number,2*Math.PI*k);}
 }
});
test('Hermitian positivity and Cauchy–Schwarz differ from bilinear cancellation',()=>{
 for(const angle of [-180,-90,0,90,180])for(const scale of [0,.5,1,2]){
  const r=evaluate('complex-linear-algebra',[angle,scale]);assert.equal(r.stats['v* v'],2);assert.equal(r.stats['vᵀv'],0);assert.ok(Math.hypot(...r.stats['Hermitian v* w'] as number[])<=(r.stats['‖v‖₂‖w‖₂'] as number)+1e-12);if(scale===0)assert.equal(r.stats['normalized |v* w|'],null);
 }
 vector(evaluate('complex-linear-algebra',[-90,1]).stats['Hermitian v* w'] as number[],[0,0]);
});
test('Wirtinger gradients match independent real differences and the exact contraction factor',()=>{
 for(const target of [0,1])for(const [x,y]of [[1,1],[-.5,1],[0,0]])for(const eta of [0,.25,.5,1,1.5]){
  const r=evaluate('wirtinger-gradients',[x,y,eta,target]),loss=(a:number,b:number)=>(a-target)**2+(b-target)**2,h=1e-5;
  vector(r.stats['real gradient (x,y)'] as number[],[(loss(x+h,y)-loss(x-h,y))/(2*h),(loss(x,y+h)-loss(x,y-h))/(2*h)],1e-8);close(r.stats['L after'] as number,(r.stats['L before'] as number)*(1-2*eta)**2);
 }
});
test('phasor interference obeys the cosine law and preserves near cancellation without inventing zero phase',()=>{
 for(const first of [-180,-90,0,90,180])for(const second of [-180,-90,0,90,180])for(const amp of [0,.5,1,2]){
  const r=evaluate('complex-signals-phase',[first,second,amp]);close(r.stats['power amplitude²'] as number,1+amp*amp+2*amp*Math.cos((second-first)*Math.PI/180));
 }
 const zero=evaluate('complex-signals-phase',[0,180,1]);assert.equal(zero.stats['combined amplitude'],0);assert.equal(zero.stats['combined phase [degrees]'],null);assert.ok(zero.curves[1].every(([,y])=>y===0));assert.ok((evaluate('complex-signals-phase',[0,180-1e-8,1]).stats['combined amplitude'] as number)>0);
});
for(const kind of complexKinds)test(`complex presets, guides and finite diagram geometry: ${kind}`,()=>{
 const spec=complexLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const preset of spec.presets){preset.values.forEach((v,i)=>assert.ok(v>=spec.controls[i].min&&v<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.holes?.flat().every(Number.isFinite)??true);}
});
test('complex examples reject malformed inputs, fractional modes and singular contour values explicitly',()=>{
 for(const kind of complexKinds){assert.throws(()=>evaluate(kind,[]));const v=complexLabs[kind].controls.map(c=>c.initial);v[0]=NaN;assert.throws(()=>evaluate(kind,v));}
 assert.throws(()=>evaluate('complex-polar-maps',[1,0,1.5]));assert.throws(()=>evaluate('residues-winding',[0,1,1.5]));assert.throws(()=>evaluate('analytic-continuation-branches',[0,.5,1]));assert.throws(()=>circleQuadrature(1,0,1,0));
});

test('nonzero powers below numeric range are not fabricated as zero or undefined phase',()=>{
 const r=evaluate('complex-polar-maps',[1e-200,45,2]);assert.equal(r.warning,'numeric-range');assert.equal(r.stats['w (Re,Im)'],null);assert.equal(r.stats['|w|'],null);close(r.stats['arg(w) [degrees]'] as number,90);assert.equal(r.curves.length,1);
});
