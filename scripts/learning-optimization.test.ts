import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {optimizationKinds,evaluateOptimization as evaluate,finiteSumRun,smoothRun,adaptiveRun,saddleRun,minibatchDistribution,varianceReduced,localDrift,bilevelRun,cubicRoot,implicitSensitivity,stationaryRun} from '../src/lib/learning-optimization';
import {optimizationLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
test('GD has the independent closed-form error and exact noiseless SGD agreement',()=>{
 for(const eta of [0,.2,1,2,2.2])for(const steps of [0,1,10,20]){const r=finiteSumRun(eta,-1,steps,0,0);close(r.gd-1,-2*(1-eta)**steps);close(r.expected,-2*(1-eta)**steps);close(r.sgd,r.gd);close(r.variance,0);assert.deepEqual(r,finiteSumRun(eta,-1,steps,0,0));}
});
test('iid mean variance and MSE match exhaustive independent paths rather than the displayed seed',()=>{
 for(const eta of [.2,1,2]){let errors=[-2];for(let steps=1;steps<=6;steps++){errors=errors.flatMap(e=>[-2,-1,1,2].map(z=>(1-eta)*e+eta*z));const m=errors.reduce((s,e)=>s+e,0)/errors.length,v=errors.reduce((s,e)=>s+(e-m)**2,0)/errors.length,mse=errors.reduce((s,e)=>s+e*e,0)/errors.length,r=finiteSumRun(eta,-1,steps,1,0);close(r.expected,m);close(r.variance,v);close(r.mse,mse);}}
});
test('constant-step noise floor and the noncontracting boundary have separately calculated limits',()=>{
 const r=finiteSumRun(.2,-1,20,1,0);close(r.variance,(2.5*.2/(2-.2))*(1-.8**40));close(finiteSumRun(2,-1,4,1,0).variance,40);assert.notEqual(finiteSumRun(.2,-1,10,1,0).sgd,finiteSumRun(.2,-1,10,1,1).sgd);
});
test('quadratic eigendirections and descent bounds match independently expanded exact formulas',()=>{
 for(const kappa of [1,3,8,12])for(const eta of [0,.1,2/kappa,.3,2.5]){const r=smoothRun(kappa,eta,8);for(const h of r.history){vector(h.x,[(1-eta)**h.k,(1-eta*kappa)**h.k]);close(h.value,(h.x[0]**2+kappa*h.x[1]**2)/2);assert.ok(h.value<=h.upper+1e-10*(1+Math.abs(h.upper)));}close(r.rho,Math.max(Math.abs(1-eta),Math.abs(1-eta*kappa)));}
});
test('a stable sign flip, boundary oscillation and an unstable steep mode remain distinct',()=>{
 assert.ok(smoothRun(8,.2,8).rho<1);close(smoothRun(8,.25,8).x[1],1);close(smoothRun(8,.3,8).rho,1.4);assert.ok(smoothRun(8,.3,8).history.at(-1)!.value>4.5);
});
test('heavy-ball equals GD at beta zero and satisfies the independent second-order recurrence',()=>{
 for(const beta of [0,.5,.95]){const r=adaptiveRun(0,8,.1,beta,.9,10);for(let k=2;k<r.history.length;k++)for(let i=0;i<2;i++)close(r.history[k].x[i],(1+beta-.1*[1,8][i])*r.history[k-1].x[i]-beta*r.history[k-2].x[i]);if(beta===0)r.history.forEach(h=>close(h.value,h.gdValue));}
});
test('heavy-ball stability flag agrees with independent characteristic-root magnitudes',()=>{
 for(const eta of [0,.1,.25,.5,1])for(const beta of [0,.5,.95]){const factors=[1,8].map(lambda=>{const a=1+beta-eta*lambda,D=a*a-4*beta;return D<0?Math.sqrt(beta):Math.max(Math.abs((a+Math.sqrt(D))/2),Math.abs((a-Math.sqrt(D))/2));});const stable=factors.every(v=>v<1-1e-12);assert.equal(adaptiveRun(0,8,eta,beta,.9,1).heavyBallStable,stable);}
});
test('first Adam bias correction recovers the gradient and squared gradient across beta values',()=>{
 for(const beta of [0,.8,.95])for(const b2 of [.5,.9,.999]){const r=adaptiveRun(1,8,.1,beta,b2,1),h=r.history[1];vector(h.mhat,[1,8]);vector(h.vhat,[1,64]);vector(r.x,[1-.1/1.001,1-.8/8.001]);}
});
test('Adam multi-step moments equal independently weighted gradient histories',()=>{
 const b1=.8,b2=.9,r=adaptiveRun(1,3,.1,b1,b2,8);for(let k=1;k<=8;k++)for(let j=0;j<2;j++){let m=0,v=0;for(let t=1;t<=k;t++){const g=r.history[t-1].x[j]*[1,3][j];m+=(1-b1)*b1**(k-t)*g;v+=(1-b2)*b2**(k-t)*g*g;}close(r.history[k].m[j],m);close(r.history[k].v[j],v);close(r.history[k].mhat[j],m/(1-b1**k));close(r.history[k].vhat[j],v/(1-b2**k));}
});
test('exact cosine saddle traps GD while an explicit perturbation escapes along negative curvature',()=>{
 const a=saddleRun(0,0,0,.2,30);close(a.gradientNorm,0);close(a.excess,2);vector(a.initialEigenvalues,[1,-1]);const b=saddleRun(0,0,.01,.2,30);close(b.y,1.821714780828057);assert.ok(b.excess<a.excess);for(let k=1;k<b.history.length;k++)assert.ok(b.history[k].value<=b.history[k-1].value+1e-12);
});
test('quartic flat minimum has no negative curvature and positive iterates decrease without a fake saddle',()=>{
 const r=saddleRun(1,0,.2,.2,40);vector(saddleRun(1,0,0,.2,1).initialEigenvalues,[1,0]);assert.ok(r.y>0&&r.y<.2);for(let k=1;k<r.history.length;k++){const old=r.history[k-1].y;close(r.history[k].y,old-.2*old**3);assert.ok(r.history[k].value<=r.history[k-1].value);}close(r.excess,r.y**4/4);
});
test('exact batch distributions normalize, center and match finite-population variance formulas',()=>{
 for(let B=1;B<=4;B++)for(const mode of [0,1])for(const sigma of [0,.3,2]){const r=minibatchDistribution(B,mode,.7,sigma);close(r.rows.reduce((s,v)=>s+v.probability,0),1);close(r.expected,.7);close(r.actualVariance,5*sigma*sigma/B*(mode===0?1:(4-B)/3));assert.equal(r.total,mode===0?4**B:[4,12,24,24][B-1]);}
});
test('zero noise pushes every batch onto one probability-one atom',()=>{
 for(let B=1;B<=4;B++)for(const mode of [0,1]){const r=minibatchDistribution(B,mode,.5,0);assert.deepEqual(r.rows,[{offset:0,gradient:.5,count:r.total,probability:1}]);close(r.actualVariance,0);}
});
test('batch PMF at B two has independently counted zero-mass probabilities',()=>{
 const a=minibatchDistribution(2,0,0,1),b=minibatchDistribution(2,1,0,1);close(a.rows.find(r=>r.gradient===0)!.probability,4/16);close(b.rows.find(r=>r.gradient===0)!.probability,4/12);close(minibatchDistribution(4,0,0,1).actualVariance,1.25);close(minibatchDistribution(4,1,0,1).actualVariance,0);
});
test('SGD and SVRG uniform means match the independently summed component derivatives',()=>{
 for(const x of [-2,.7,2])for(const anchor of [-2,.7,2]){const r=varianceReduced(x,anchor,.1,0),h=[1,2,3,4],target=[-1,1,2,0],G=h.reduce((s,v,i)=>s+v*(x-target[i]),0)/4;close(r.sgdMean,G);close(r.svrgMean,G);close(r.svrgVariance,1.25*(x-anchor)**2);}
});
test('zero estimator variance can occur away from optimum and stale anchors can increase variance',()=>{
 close(varianceReduced(.7,.7,.1,0).sgdVariance,6.575);close(varianceReduced(.7,.7,.1,0).svrgVariance,0);close(varianceReduced(2,2,.1,0).full,3.25);close(varianceReduced(.7,-1,.1,0).svrgVariance,3.6125);assert.ok(varianceReduced(-2,2,.1,0).svrgVariance>varianceReduced(-2,2,.1,0).sgdVariance);
});
test('local round trajectories equal an independently derived affine closed form',()=>{
 for(const kappa of [1,4,8])for(const eta of [.1,.3])for(const E of [1,4,8]){const r=localDrift(kappa,eta,E,12,-.3),a1=(1-eta)**E,a2=(1-eta*kappa)**E,A=(a1+a2)/2,B=(a1-a2)/2;close(r.mapA,A);close(r.mapB,B);for(const h of r.history)close(h.local,A===1?-.3+h.k*B:A**h.k*(-.3)+B*(1-A**h.k)/(1-A),1e-8);}
});
test('one local step matches centralized GD while heterogeneous multi-step averaging has bias',()=>{
 const r=localDrift(4,.1,1,8,0);r.history.forEach(h=>close(h.local,h.central));close(r.fixed!,.6);close(r.fixed!-r.optimum,0);const q=localDrift(4,.1,4,8,0);close(q.fixed!,5265/12143);assert.ok(Math.abs(q.gradient)>.4);assert.ok(Math.abs(q.mapA)<1);const z=localDrift(4,0,4,8,.3);assert.equal(z.fixed,null);close(z.local,.3);assert.ok(Math.abs(localDrift(8,.3,8,12,0).mapA)>1);
});
test('unrolled ridge sensitivities match an independent analytic finite-step solution',()=>{
 for(const lambda of [.2,1,3.8])for(const eta of [.1,.25,.5])for(const K of [0,1,8,20]){const r=bilevelRun(lambda,.5,eta,K),q=1-eta*(1+lambda),w=2/(1+lambda)*(1-q**K),derivative=K===0?0:-2/(1+lambda)**2*(1-q**K)+2/(1+lambda)*K*eta*q**(K-1);close(r.w,w);close(r.sensitivity,derivative);close(r.unrolled,(w-.5)*derivative);}
});
test('hypergradients match finite differences of the exact objective actually differentiated',()=>{
 const h=1e-5;for(const K of [1,8])for(const lambda of [.2,1,3.8]){const V=(l:number)=>{const q=1-.25*(1+l),w=2/(1+l)*(1-q**K);return (w-.5)**2/2;};close(bilevelRun(lambda,.5,.25,K).unrolled,(V(lambda+h)-V(lambda-h))/(2*h),1e-8);}const r=bilevelRun(1,.5,.25,1);close(r.unrolled,0);close(r.exactHyper,-.25);close(bilevelRun(1,.5,.25,8).unrolled,-.23932647705078125);
});
test('nonlinear roots agree with independent interval bisection over regular parameter values',()=>{
 for(const delta of [0,.1,1,2])for(const theta of [-1,-.125,0,.125,1]){let lo=-2,hi=2;for(let j=0;j<100;j++){const mid=(lo+hi)/2;if(mid**3+delta*mid>theta)hi=mid;else lo=mid;}const w=cubicRoot(delta,theta);close(w,(lo+hi)/2);close(w**3+delta*w,theta);}
});
test('implicit derivatives match finite differences only away from the singular Jacobian',()=>{
 for(const delta of [.1,1,2])for(const theta of [-.7,0,.7]){const r=implicitSensitivity(delta,theta,.3),h=1e-6,fd=(cubicRoot(delta,theta+h)-cubicRoot(delta,theta-h))/(2*h);close(r.derivative!,fd,1e-8);close(r.hyper!,(r.w-.3)*r.derivative!);}
 const singular=implicitSensitivity(0,0,1);assert.equal(singular.derivative,null);assert.equal(singular.hyper,null);assert.equal(singular.singular,true);close(singular.finiteDifference,(1e-4)**(-2/3));close(implicitSensitivity(0,.125,0).derivative!,4/3);
});
test('finite derivative overflow is distinct from singularity and keeps graph geometry finite',()=>{
 const r=evaluate('implicit-solution-sensitivity',[Number.MIN_VALUE,0,1]);assert.equal(r.numericNote,true);assert.equal(r.stats['implicit dw/dtheta'],null);assert.equal(r.stats['1{implicit-function Jacobian nonsingular}'],1);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.equal(implicitSensitivity(0,0,1).numericNote,false);
});
test('small steps zero gradients and objective error are independent stopping indicators',()=>{
 const max=stationaryRun(1,0,.2,20);close(max.gradient,0);close(max.value,.25);close(max.hessian,-1);assert.equal(max.exactMinimizer,false);const zero=stationaryRun(0,1,0,20);close(zero.history.at(-1)!.step,0);close(zero.gradient,1);const q=stationaryRun(0,1,.2,10);close(q.gradient,.8**10);close(q.value,.8**20/2);
});
test('exponential iterates increase while objectives decrease without an attained finite minimum',()=>{
 const r=stationaryRun(2,0,.2,30);assert.equal(r.attained,false);assert.equal(r.exactMinimizer,false);for(let k=1;k<r.history.length;k++){assert.ok(r.history[k].x>r.history[k-1].x);assert.ok(r.history[k].value<r.history[k-1].value);close(r.history[k].gradient,r.history[k].value);}
});
test('optimization model domains reject malformed, nonfinite and fractional discrete controls',()=>{
 for(const kind of optimizationKinds){const spec=optimizationLabs[kind],initial=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));for(let i=0;i<initial.length;i++){for(const v of [NaN,Infinity,spec.controls[i].min-1,spec.controls[i].max+1]){const bad=[...initial];bad[i]=v;assert.throws(()=>evaluate(kind,bad));}if(spec.controls[i].step!=='any'){const bad=[...initial];bad[i]=spec.controls[i].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
});
for(const kind of optimizationKinds)test(`${kind}: meaningful guides and finite geometry across control boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=optimizationLabs[kind],initial=spec.controls.map(c=>c.initial);for(const g of learningGuides[kind].experiments)assert.ok(spec.presets[g.preset]);const values=[...spec.presets.map(p=>p.values),...spec.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))];for(const p of values){const r=evaluate(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite),`${kind} ${p}`);assert.ok(r.bounds!.every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of optimizationKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed theory, valid equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});

test('centered SGD errors retain noise smaller than the parameter rounding scale',()=>{
 const r=finiteSumRun(.5,1,1,1e-20,0);assert.equal(r.sgd,1);assert.equal(r.sgdError,-1e-20);assert.equal(r.gdError,0);
 assert.equal(evaluate('finite-sum-gd-sgd',[.5,1,1,1e-20,0]).stats['illustrative SGD final error x-1'],-1e-20);
 const quiet=finiteSumRun(.2,-1,20,0,0);quiet.history.forEach(h=>assert.equal(h.sgdError,h.gdError));
});
test('SVRG variance is not lost when all displayed estimators round to the same value',()=>{
 const r=varianceReduced(0,1e-20,.1,0);assert.ok(r.svrg.every(x=>x===r.svrg[0]));
 assert.ok(Math.abs(r.svrgVariance/1.25e-40-1)<1e-15);
});
test('exact-solution hypergradients retain a small outer residual near the validation target',()=>{
 const r=bilevelRun(1e-20,2,.25,20);assert.equal(r.exact,2);assert.equal(r.exactHyper,4e-20);
 assert.equal(Math.abs(bilevelRun(0,2,.25,20).exactHyper),0);
});
test('strict quadratic and local-round stability survives a factor rounded to one',()=>{
 const r=smoothRun(8,1e-20,1);assert.equal(r.rho,1);assert.equal(r.contracts,true);
 assert.equal(evaluate('smoothness-learning-rate',[8,1e-20,1]).stats['1{all error modes strictly contract}'],1);
 const local=localDrift(4,1e-20,4,1,0);assert.equal(local.mapA,1);assert.equal(local.contracts,true);
 assert.equal(localDrift(4,0,4,1,0).contracts,false);
});
