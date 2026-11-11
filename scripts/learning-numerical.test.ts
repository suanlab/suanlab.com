import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {roundBinary,accumulation,stableExponential,conditioning,gaussianSolve,nonlinearSolvers,sparseIteration,interpolationQuadrature,autodiffTape,differenceCheck,refinement,checkpointCounts,checkpointChain,evaluateNumerical,numericalKinds} from '../src/lib/learning-numerical';
import {numericalLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const norm=(v:number[])=>Math.hypot(...v);
test('binary rounding independently finds nearest representable dyadic including ties and binade changes',()=>{
 for(const p of [3,4,6]){const grid=[0,...Array.from({length:12},(_,k)=>k-5).flatMap(e=>Array.from({length:2**(p-1)},(_,j)=>(2**(p-1)+j)*2**(e-p+1)))].sort((a,b)=>a-b);for(let i=0;i<400;i++){const x=.1+i/200,distances=grid.map(v=>Math.abs(v-x)),min=Math.min(...distances),q=roundBinary(x,p);close(Math.abs(q-x),min);close(roundBinary(-x,p),-q);close(roundBinary(q,p),q);assert.ok(Math.abs(q-x)<=2**-p*x+1e-15);}}
 assert.equal(roundBinary(1.0625,4),1);assert.equal(roundBinary(1.1875,4),1.25);assert.equal(roundBinary(1.9375,4),2);assert.equal(roundBinary(0,4),0);
});
test('compensation recovers absorbed increments while input error is separately visible',()=>{
 const a=accumulation(4,1,5,16);close(a.last.naive,1);close(a.last.kahan,1.5);close(a.last.naiveError,.5);close(a.last.kahanError,0);const b=accumulation(4,.1,5,0);assert.ok(b.inputError>0);close(b.last.naiveError,0);close(accumulation(12,1,5,16).last.naiveError,0);
});
test('log-sum-exp agrees with direct safe arithmetic and exact equal-logit limits',()=>{
 for(const offset of [-10,0,10])for(const gap of [0,1,10])for(const T of [.5,1,2]){const r=stableExponential(offset,gap,T),exp=[offset,offset-gap,offset-2*gap].map(v=>Math.exp(v/T)),denom=exp.reduce((a,b)=>a+b,0);close(r.logValue,Math.log(denom));vector(r.probability,exp.map(v=>v/denom));close(r.probability.reduce((a,b)=>a+b),1);}for(const off of [-1000,1000]){const r=stableExponential(off,0,1);vector(r.probability,[1/3,1/3,1/3]);close(r.logValue,off+Math.log(3));}
});
test('raw exponential overflow and underflow remain explicit invalid states rather than zero-filled probabilities',()=>{
 const hi=stableExponential(1000,1,1),lo=stableExponential(-1000,1,1);assert.ok(hi.overflow&&lo.underflow);assert.equal(hi.naive,null);assert.equal(lo.naiveLog,null);vector(hi.probability,lo.probability);assert.ok(stableExponential(-1000,50,.1).probability.some(v=>v===0));
});
test('conditioning separates equal-sized strong and weak data perturbations and arithmetic residual bounds',()=>{
 const a=conditioning(6,6,24,0),b=conditioning(6,6,24,1);close(a.target[1],2);close(a.condition,1e6);close(a.dataForwardError,1/Math.SQRT2);close(b.dataForwardError,1e-6/Math.SQRT2);close(a.relativeDataError,b.relativeDataError);for(const k of [0,3,6,10])for(const q of [2,6,14])for(const p of [3,12,24])for(const d of [0,1]){const r=conditioning(k,q,p,d);assert.ok(r.dataForwardError<=r.condition*r.relativeDataError+1e-10);assert.ok(r.result.numericalError<=r.condition*r.result.backward+1e-10);}
});
test('pivoted Gaussian elimination matches known solutions, row swaps and singular rejection',()=>{
 for(const A of [[[2,1],[1,2]],[[0,2],[1,3]],[[1,2,3],[4,7,8],[2,1,4]]]){const exact=A.map((_,i)=>i-1),b=A.map(row=>row.reduce((s,v,j)=>s+v*exact[j],0));vector(gaussianSolve(A,b),exact);}assert.throws(()=>gaussianSolve([[1,2],[2,4]],[3,6]));
});
test('Newton update solves the local tangent equation and bisection always contains the positive root',()=>{
 for(const a of [.25,1,2,4])for(const start of [.25,1,4]){const r=nonlinearSolvers(a,start,10,1);r.history.forEach((h,i)=>{assert.ok(h.lo<=Math.sqrt(a)&&h.hi>=Math.sqrt(a));assert.ok(h.bisectionError<=(h.hi-h.lo)/2+1e-15);if(i){const prev=r.history[i-1];close((h.newton-prev.newton)*2*prev.newton+prev.newton*prev.newton-a,0);close(h.hi-h.lo,(prev.hi-prev.lo)/2);}});vector(r.linear,[(2*a-1)/3,(2-a)/3]);}const r=nonlinearSolvers(2,1,4,1);close(r.selected,1.4375);close(r.bracketBound!,.0625);close(nonlinearSolvers(1,1,0,0).error,0);
});
test('sparse Poisson reference matches the independent discrete parabola and exact spectral endpoints',()=>{
 for(const N of [2,4,8,16]){const r=sparseIteration(N,0,1,0,0);vector(r.reference,Array.from({length:N},(_,i)=>(i+1)*(N-i)/2));close(r.nonzeros,3*N-2);close(r.lambdaMin,4*Math.sin(Math.PI/(2*(N+1)))**2);close(r.rho,Math.cos(Math.PI/(N+1)));}
});
test('actual weighted Jacobi iterates match an independent dense error-propagation calculation',()=>{
 const N=4,c=.5,w=.8,r=sparseIteration(N,c,w,6,0);let x=Array(N).fill(0);for(const h of r.history){vector(h.jacobi,x);const next=x.map((v,i)=>(1-w)*v+w*(1+(x[i-1]??0)+(x[i+1]??0))/(2+c));x=next;}
});
test('CG solves the symmetric rhs and reports genuine stopping while residual bounds remain valid',()=>{
 const r=sparseIteration(8,0,1,12,1);assert.equal(r.last.updates,4);close(r.error,0);vector(r.last.cg,r.reference);for(const n of [2,7,16])for(const c of [0,.5,2])for(const omega of [.1,1,1.2])for(const method of [0,1]){const s=sparseIteration(n,c,omega,30,method);assert.ok(s.error<=s.errorBound+1e-9);}assert.ok(sparseIteration(16,0,1.2,40,0).rho>1);
});
test('quadrature matches hand weights, exact fourth-power error and derivative-based bounds',()=>{
 const a=interpolationQuadrature(1,1,.25,1);close(a.simpson,5/24);close(a.error,1/120);close(a.bound,1/120);close(a.interpolated,1/32);for(const mode of [0,1])for(const m of [1,2,6,12])for(const method of [0,1]){const r=interpolationQuadrature(m,mode,.37,method);assert.ok(r.error<=r.bound+1e-14);for(let j=0;j<=100;j++){const t=interpolationQuadrature(m,mode,j/100,method);assert.ok(t.queryError<=t.interpolationBound+1e-14);}close(r.n,2*m);close(interpolationQuadrature(m,mode,0,method).queryError,0);close(interpolationQuadrature(m,mode,1,method).queryError,0);}
});
test('interpolation and quadrature converge at their stated asymptotic orders on smooth examples',()=>{
 const a=interpolationQuadrature(4,0,.37,0),b=interpolationQuadrature(8,0,.37,0),c=interpolationQuadrature(4,1,.37,1),d=interpolationQuadrature(8,1,.37,1);assert.ok(a.error/b.error>3.9&&a.error/b.error<4.1);close(c.error/d.error,16,1e-7);
});
test('AD tape agrees with independently differentiated formulas and directional finite differences',()=>{
 for(const x of [-1,.2,1])for(const y of [-1,0,.7])for(const w of [-1,0,1]){const vx=.4,vy=-.3,r=autodiffTape(x,y,vx,vy,w),gx=y*Math.cos(x*y)+2*x,gy=x*Math.cos(x*y);vector(r.vjp,[w*gx,w*gy]);close(r.jvp,gx*vx+gy*vy);close(r.pairingLeft,r.pairingRight);const F=(s:number)=>Math.sin((x+s*vx)*(y+s*vy))+(x+s*vx)**2;close((F(1e-5)-F(-1e-5))/(2e-5),r.jvp,1e-9);}
 const r=autodiffTape(1,0,1,1,1);vector(r.vjp,[2,1]);close(r.jvp,3);assert.equal(r.nodes[4].parents.length,2);
});
test('forward tangent and reverse cotangent seeds alter the correct independent quantities',()=>{
 const a=autodiffTape(1,0,1,1,1),b=autodiffTape(1,0,0,1,1),c=autodiffTape(1,0,1,1,0);close(a.value,b.value);vector(a.vjp,b.vjp);close(b.jvp,1);vector(c.vjp,[0,0]);close(c.jvp,3);
});
test('finite differences expose second-order truncation, vanished steps and a nondifferentiable false pass',()=>{
 const a=differenceCheck(0,.5,-2,0),b=differenceCheck(0,.5,-3,0);assert.ok(a.trueCentralError!/b.trueCentralError!>99);const tiny=differenceCheck(0,1,-18,0);close(tiny.current.plusStep,0);close(tiny.current.central,0);close(tiny.derivative!,Math.E);const kink=differenceCheck(1,0,-4,0);assert.equal(kink.derivative,null);assert.equal(kink.differentiable,false);close(kink.current.forward,1);close(kink.current.central,0);close(kink.checkDiscrepancy,0);assert.equal(kink.trueCentralError,null);
});
test('gradient checks detect an injected wrong derivative and crossing a kink changes the stencil',()=>{
 const r=differenceCheck(0,.5,-4,.25);close(r.checkDiscrepancy,.25,1e-7);close(differenceCheck(1,.1,0,0).current.central,.1);close(differenceCheck(1,.1,-2,0).current.central,1);
});
test('mixed-precision corrections recover a rounded-away rhs while low residuals can falsely vanish',()=>{
 const initial=refinement(3,4,0,0),high=refinement(3,4,1,0),low=refinement(3,4,1,1);vector(initial.last!.x,[2,0]);close(initial.last!.error,1);vector(high.last!.x,[1,1]);close(high.last!.trueResidual,0);vector(low.last!.x,[2,0]);close(low.last!.computedResidual,0);close(low.last!.trueResidual,.125);close(low.last!.error,1);
});
test('rounded singular factors never claim a solution or a zero-error curve',()=>{
 const r=refinement(6,3,2,0);assert.equal(r.valid,false);assert.equal(r.last,null);assert.equal(r.pivot,0);assert.deepEqual(r.history,[]);const view=evaluateNumerical('mixed-precision-refinement',[6,3,2,0]);assert.equal(view.stats['relative solution error'],null);assert.deepEqual(view.curves,[]);
});
test('refinement residuals independently match the original system and its spectral condition number',()=>{
 for(const k of [0,1,3,6])for(const p of [3,6,12]){const r=refinement(k,p,3,0);const delta=2**-k,lambda=(2+delta+Math.hypot(2,delta))/2;close(r.condition,lambda*lambda/delta);for(const h of r.history){const [x,y]=h.x;close(h.trueResidual,Math.hypot(2-x-y,2+delta-x-(1+delta)*y));close(h.error,Math.hypot(x-1,y-1)/Math.SQRT2);}}
});
test('checkpoint schedules match independently enumerated boundaries, interiors, bytes and matvec counts',()=>{
 for(const L of [2,7,16,32])for(const B of [1,3,4,16]){const c=checkpointCounts(L,8,B,4),boundaries=[0];for(let k=B;k<L;k+=B)boundaries.push(k);boundaries.push(L);const lengths=boundaries.slice(1).map((end,i)=>end-boundaries[i]);assert.equal(c.checkpoints,boundaries.length);assert.equal(c.recomputed,lengths.reduce((s,n)=>s+n-1,0));assert.equal(c.checkpointValues,(boundaries.length+Math.max(...lengths)-1+2)*8);assert.equal(c.checkpointMultiplies,(L+c.recomputed+L)*64);}const r=checkpointCounts(16,8,4,4);assert.equal(r.fullBytes,608);assert.equal(r.checkpointBytes,320);assert.equal(r.fullMultiplies,2048);assert.equal(r.checkpointMultiplies,2816);assert.equal(checkpointCounts(16,8,1,4).recomputed,0);
});
test('checkpoint recomputation preserves full reverse gradients and matches independent scalar recurrences',()=>{
 for(const L of [2,7,16])for(const W of [2,5])for(const B of [1,3,16]){const a=1.2,x=.4,r=checkpointChain(L,W,B,a,x);close(r.gradientDifference,0);close(r.parameterDifference,0);assert.equal(r.recomputed,checkpointCounts(L,W,B,8).recomputed);let h=x*(W+1)/(2*W),da=0,dx=(W+1)/(2*W);for(let l=0;l<L;l++){const out=Math.tanh(a*h),d=1-out*out;da=d*(h+a*da);dx=d*a*dx;h=out;}close(r.value,h);close(r.full.parameterGradient,da);close(r.inputParameterDerivative,dx);}
});
test('checkpoint alpha and initial-state gradients match independent outer finite differences',()=>{
 const L=7,W=4,B=3,a=1.1,x=.4,h=1e-5,r=checkpointChain(L,W,B,a,x);const value=(gain:number,input:number)=>{let state=input*(W+1)/(2*W);for(let k=0;k<L;k++)state=Math.tanh(gain*state);return state;};close((value(a+h,x)-value(a-h,x))/(2*h),r.full.parameterGradient,1e-8);close((value(a,x+h)-value(a,x-h))/(2*h),r.inputParameterDerivative,1e-8);
});
test('numerical model domains reject malformed, fractional and nonfinite controls',()=>{
 for(const kind of numericalKinds){const s=numericalLabs[kind],initial=s.controls.map(c=>c.initial);assert.throws(()=>evaluateNumerical(kind,[]));for(const [i,c] of s.controls.entries())for(const v of [NaN,Infinity,c.min-1,c.max+1,...(c.step==='any'?[]:[c.min+.5])]){const bad=[...initial];bad[i]=v;assert.throws(()=>evaluateNumerical(kind,bad));}}
});
for(const kind of numericalKinds)test(`${kind}: meaningful guides and finite geometry across parameter boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=numericalLabs[kind],initial=spec.controls.map(c=>c.initial);for(const g of learningGuides[kind].experiments)assert.ok(spec.presets[g.preset]);for(const p of [...spec.presets.map(p=>p.values),...spec.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))]){const r=evaluateNumerical(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.bounds!.every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(Number.isFinite(v));}
});
for(const kind of numericalKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed bilingual theory, equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});

test('binary64 precision preserves represented neighbors just below binade boundaries',()=>{
 for(const e of [-20,-1,0,1,20,100,900]){const power=2**e,below=power-2**(e-53);assert.equal(roundBinary(below,53),below);assert.equal(roundBinary(-below,53),-below);}
 assert.equal(roundBinary(Number.MAX_VALUE,53),Number.MAX_VALUE);
});
test('finite exponentials with an overflowing sum do not become a zero probability vector',()=>{
 const r=stableExponential(709,0,1);assert.ok(r.scaled.every(x=>Number.isFinite(Math.exp(x))));assert.equal(r.overflow,true);assert.equal(r.naive,null);vector(r.probability,[1/3,1/3,1/3]);
});
test('relative logits and leading-class loss retain a small tail before absolute-logit cancellation',()=>{
 const r=stableExponential(1000,50,1),expected=Math.log1p(Math.exp(-50)+Math.exp(-100));assert.equal(r.logValue-1000,0);assert.equal(r.loss[0],expected);assert.ok(r.loss[0]>0);
 const tiny=stableExponential(1000,1e-14,1);assert.ok(tiny.scaled.every(x=>x===1000));assert.deepEqual(tiny.probability,stableExponential(0,1e-14,1).probability);assert.ok(tiny.probability[0]>tiny.probability[2]);
});
test('centered data and residual errors retain requested perturbations rather than rounded target differences',()=>{
 const r=conditioning(0,14,24,1),expected=1e-14/Math.SQRT2;assert.equal(r.dataForwardError,expected);assert.equal(r.result.residual,1e-14);assert.equal(r.result.numericalError,1e-14/Math.hypot(...r.target));assert.ok(r.result.numericalError<=r.condition*r.result.backward*(1+1e-14));
});

test('new class-loss readout preserves the indexed conceptual reference statistics',()=>{
 const r=evaluateNumerical('stable-exponentials-logsumexp',[1000,1,1]);assert.deepEqual(Object.keys(r.stats).slice(0,4),['stable log-sum-exp','naive log-sum-exp when finite','1{naive exponential sum overflowed}','1{all naive exponentials underflowed}']);assert.equal(Object.keys(r.stats).at(-1),'first-class stable cross entropy');
});
