import test from 'node:test';
import assert from 'node:assert/strict';
import { analysisKinds, evaluateAnalysis as evaluate, expTaylor } from '../src/lib/learning-analysis';
import { analysisLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol,`${a} differs from ${b}`);
test('removable and jump limits keep point values, side limits and excluded markers distinct',()=>{
 const hole=evaluate('limits-continuity-completeness',[0,0,.5]);assert.equal(hole.stats['two-sided limit'],2);assert.equal(hole.stats['1{continuous at 1}'],0);assert.deepEqual(hole.holes,[[1,2]]);
 const continuous=evaluate('limits-continuity-completeness',[0,2,.1]);assert.equal(continuous.stats['1{continuous at 1}'],1);assert.deepEqual(continuous.holes,[]);
 const jump=evaluate('limits-continuity-completeness',[1,.5,.1]);assert.equal(jump.stats['two-sided limit'],null);assert.equal(jump.stats['left limit'],0);assert.equal(jump.stats['right limit'],1);assert.equal(jump.holes!.length,2);
 const small=evaluate('limits-continuity-completeness',[0,0,.01]);assert.ok(small.curves[0].every(([x])=>x>=.99&&x<1));
});
test('derivatives, finite differences and mean-value hypotheses are separate quantities',()=>{
 for(const x of [-1,-.5,0,.5,1])for(const h of [.01,.1,1]){const r=evaluate('differentiability-mean-value',[0,x,h]);close(r.stats['f′(x₀)'] as number,2*x);close(r.stats['forward secant'] as number,2*x+h);}
 assert.equal(evaluate('differentiability-mean-value',[0,.5,0]).stats['forward secant'],null);const corner=evaluate('differentiability-mean-value',[1,0,.1]);assert.equal(corner.stats['f′(x₀)'],null);assert.equal(corner.stats['MVT witness c'],null);assert.equal(corner.curves.length,1);
});
test('Riemann sums obey analytic closed forms and error bounds under refinement',()=>{
 for(const x of [0,.25,1,2])for(const n of [1,2,4,16,64]){
  const r=evaluate('integration-fundamental-theorem',[x,n]);const closed=x**3*(n-1)*(2*n-1)/(6*n*n);close(r.stats['left sum'] as number,closed);assert.ok((r.stats['absolute error'] as number)<=(r.stats['error bound x³/N'] as number)+1e-12);close(r.stats['F′(x)=f(x)'] as number,x*x);
 }
 close(evaluate('integration-fundamental-theorem',[1,4]).stats['left sum'] as number,7/32);
});
test('Taylor values match independent factorial sums and remainder bounds across the example domain',()=>{
 for(let n=0;n<=8;n++)for(let i=-20;i<=20;i++){
  const x=i/10;let independent=0,factorial=1;for(let k=0;k<=n;k++){if(k>0)factorial*=k;independent+=x**k/factorial;}close(expTaylor(x,n),independent);const r=evaluate('taylor-remainders',[x,n]);assert.ok((r.stats['absolute error'] as number)<=(r.stats['Lagrange bound'] as number)+1e-12);
 }
});
test('alternating and integral-comparison tail bounds hold; harmonic terms vanish without a finite sum',()=>{
 for(let n=1;n<=128;n++)for(const mode of [0,1,2]){
  const r=evaluate('series-convergence-tests',[mode,n]);if(mode===0){assert.equal(r.stats['series sum'],null);assert.equal(r.stats['tail bound'],null);}else assert.ok(Math.abs((r.stats.S_N as number)-(r.stats['series sum'] as number))<=(r.stats['tail bound'] as number)+1e-12);
 }
});
test('pointwise limits, endpoint values and nonattained supremum differ from a sampled maximum',()=>{
 for(const n of [1,8,64,128]){
  const r=evaluate('pointwise-uniform-convergence',[0,n,.5]);assert.equal(r.stats['pointwise limit at x₀'],0);assert.equal(r.stats['sup error on [0,1]'],1);assert.equal(evaluate('pointwise-uniform-convergence',[0,n,1]).stats['pointwise error'],0);
  const counterexample=Math.pow(.75,1/n);close(evaluate('pointwise-uniform-convergence',[0,n,counterexample]).stats['pointwise error'] as number,.75,1e-12);
  const uniform=evaluate('pointwise-uniform-convergence',[1,n,1]);assert.equal(uniform.stats['sup error on [0,1]'],1/n);
 }
});
test('open-interval extrema are absent while supremum and infimum remain defined',()=>{
 const open=evaluate('compactness-extreme-values',[1,1]);assert.equal(open.stats['f(x₀)'],null);assert.equal(open.stats.maximum,null);assert.equal(open.stats['sup f'],1);assert.equal(open.holes!.length,2);assert.equal(evaluate('compactness-extreme-values',[0,1]).stats.maximum,1);
});
test('quadratic tangent gaps certify all pairs analytically and do not mistake an equal-point probe',()=>{
 for(const a of [-2,-1,0,1,2])for(const x of [-1,0,1])for(const y of [-1,0,1]){
  const r=evaluate('convexity-first-order',[a,x,y]);close((r.stats['f(y)'] as number)-(r.stats['f(x)+f′(x)(y−x)'] as number),r.stats['exact tangent gap a(y−x)²'] as number);assert.equal(r.stats['1{convex on R}'],a>=0?1:0);
 }
});
test('shrinking spikes retain exact trapezoid area and include their peaks even at narrow support',()=>{
 for(let n=2;n<=64;n++){
  const r=evaluate('limits-integration-interchange',[0,n]),curve=r.curves[0];let area=0;for(let i=1;i<curve.length;i++)area+=(curve[i][0]-curve[i-1][0])*(curve[i][1]+curve[i-1][1])/2;close(area,1);assert.ok(curve.some(([x,y])=>x===1/n&&y===n));assert.equal(r.stats['∫ lim f_n'],0);assert.equal(r.stats['lim ∫ f_n'],1);close(evaluate('limits-integration-interchange',[1,n]).stats['∫ f_n'] as number,1/(2*n));
 }
});
test('implicit and inverse derivatives handle singular Jacobians without declaring missing inverses',()=>{
 close(evaluate('implicit-inverse-functions',[0,.6]).stats['dy/dx'] as number,-.75);assert.equal(evaluate('implicit-inverse-functions',[0,1]).stats['dy/dx'],null);const cube=evaluate('implicit-inverse-functions',[1,0]);assert.equal(cube.stats['1{locally invertible}'],1);assert.equal(cube.stats['inverse derivative at f(x₀)'],null);
});
test('logdet differences respect both perturbed SPD domains and shared-parameter shapes',()=>{
 for(const [a,b,c]of [[2,1,2],[2,0,2],[1,.5,3]]){
  const r=evaluate('matrix-trace-logdet-derivatives',[a,b,c,1e-5]);close(r.stats['central difference'] as number,c/(a*c-b*b),1e-8);assert.equal(r.stats['d tr(AᵀA)/db (shared b)'],4*b);
 }
 const singular=evaluate('matrix-trace-logdet-derivatives',[1,1,1,.01]);assert.equal(singular.stats['log det A (SPD domain)'],null);assert.equal(singular.stats['central difference'],null);assert.equal(evaluate('matrix-trace-logdet-derivatives',[2,1,2,0]).stats['central difference'],null);
});
test('vector arrows preserve the specified field, display scale and constant Jacobian',()=>{
 const p=[0,-1,1,0],r=evaluate('vector-fields-divergence-curl',p);assert.equal(r.stats['div F'],0);assert.equal(r.stats['curl_z F'],2);for(const [s,t]of r.curves){close(t[0]-s[0],-.3*s[1]);close(t[1]-s[1],.3*s[0]);}assert.equal(r.curves.length,9);
});
test('line circulation and flux agree with independent midpoint parametrization',()=>{
 for(const mode of [0,1])for(const R of [0,.25,1,2]){
  let circulation=0,flux=0;const dt=2*Math.PI/256;for(let k=0;k<256;k++){const t=(k+.5)*dt,x=R*Math.cos(t),y=R*Math.sin(t),f=mode===0?[-y,x]:[x,y];circulation+=(f[0]*(-y)+f[1]*x)*dt;flux+=(f[0]*x+f[1]*y)*dt;}
  const r=evaluate('line-surface-volume-integrals',[mode,R]);close(r.stats['∮ F·dr'] as number,circulation);close(r.stats['∮ F·n ds'] as number,flux);
 }
});
test('Gauss and Stokes compare independently accumulated boundary quadratures with analytic interiors',()=>{
 for(const R of [0,.25,1,2])for(const a of [-1,0,1])for(const b of [-1,0,1]){
  const r=evaluate('gauss-stokes-theorems',[R,a,b]),exact=r.stats['∫_ball div F'] as number;close(r.stats['sphere flux (Simpson)'] as number,exact,1e-7*Math.max(1,Math.abs(exact)));close(r.stats['CCW circulation (midpoint)'] as number,r.stats['∫_disk curl·n dA'] as number,1e-10);
 }
});
test('nonlinear Hessian products agree with independent gradient and scalar directional differences',()=>{
 const f=(x:number,y:number)=>.5*x*x*y+Math.sin(y),grad=(x:number,y:number)=>[x*y,.5*x*x+Math.cos(y)];
 for(const [x,y,vx,vy] of [[1,0,1,1],[1,0,1,-1],[-1,.5,.3,-.7],[0,1,0,0]]){
  const h=1e-5,r=evaluate('higher-order-autodiff',[x,y,vx,vy]),gp=grad(x+h*vx,y+h*vy),gm=grad(x-h*vx,y-h*vy),hv=r.stats.Hv as number[];hv.forEach((v,i)=>close(v,(gp[i]-gm[i])/(2*h),1e-8));close(r.stats['vᵀHv'] as number,(f(x+h*vx,y+h*vy)-2*f(x,y)+f(x-h*vx,y-h*vy))/(h*h),1e-5);
 }
});
for(const kind of analysisKinds)test(`analysis presets and guides have valid finite geometry: ${kind}`,()=>{
 const spec=analysisLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const preset of spec.presets){assert.equal(preset.values.length,spec.controls.length);preset.values.forEach((v,i)=>assert.ok(v>=spec.controls[i].min&&v<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.holes?.flat().every(Number.isFinite)??true);}
});
test('invalid shape, nonfinite, fractional modes and out-of-domain controls fail explicitly',()=>{
 for(const kind of analysisKinds){assert.throws(()=>evaluate(kind,[]));const values=analysisLabs[kind].controls.map(c=>c.initial);values[0]=NaN;assert.throws(()=>evaluate(kind,values));}
 assert.throws(()=>evaluate('series-convergence-tests',[.5,8]));assert.throws(()=>evaluate('pointwise-uniform-convergence',[0,0,.5]));assert.throws(()=>evaluate('limits-integration-interchange',[0,1]));assert.throws(()=>evaluate('integration-fundamental-theorem',[-1,4]));assert.throws(()=>expTaylor(1,1.5));
});

// Keep the public bilingual scope synchronized with the design and live reading routes.
import fs from 'node:fs';
import { curriculumDomains, curriculumTopicCount } from '../src/data/learning-curriculum';
test('public curriculum contains every designed topic in both languages and only valid reading links',()=>{
 const design=JSON.parse(fs.readFileSync('docs/online-learning/physics-math-for-ai/curriculum/knowledge-map.json','utf8'));
 assert.equal(curriculumDomains.length,design.domains.length);assert.equal(curriculumTopicCount,509);
 for(const domain of curriculumDomains){const original=design.domains.find((d:{id:string})=>d.id===domain.id);assert.deepEqual(domain.topics.map(t=>t.ko),original.topics);assert.ok(!/[가-힣]/.test(domain.title.en));for(const topic of domain.topics){assert.ok(topic.en.length>0);assert.ok(!/[가-힣]/.test(topic.en));}for(const id of domain.unitIds)assert.ok(learningUnits.some(u=>u.id===id));for(const id of domain.prerequisiteIds)assert.ok(curriculumDomains.some(d=>d.id===id));}
});
