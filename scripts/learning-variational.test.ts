import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {variationalKinds,evaluateVariational as evaluate,functionalVariation,eulerBoundary,constrainedVariation,bernoulliKL,bernoulliKLShift,fisherExample,naturalStep,simplexProjection,mirrorRun,bregman,bregmanExample,circleRun,adjointObjective,functionMetricRun} from '../src/lib/learning-variational';
import {variationalLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
function integrate(f:(x:number)=>number,N=4000){let s=f(0)+f(1);for(let i=1;i<N;i++)s+=(i%2?4:2)*f(i/N);return s/(3*N);}
function solve(A:number[][],b:number[]){const M=A.map((r,i)=>[...r,b[i]]);for(let j=0;j<b.length;j++){const pivot=M[j][j];for(let k=j;k<=b.length;k++)M[j][k]/=pivot;for(let i=0;i<b.length;i++)if(i!==j){const factor=M[i][j];for(let k=j;k<=b.length;k++)M[i][k]-=factor*M[j][k];}}return M.map(r=>r[b.length]);}
test('functional variations match independently integrated fields and derivative products',()=>{
 const a=.7,b=-.3,c=.2,d=.8,alpha=2,r=functionalVariation(a,b,c,d,.1,alpha),y=(t:number)=>a*Math.sin(Math.PI*t)+b*Math.sin(2*Math.PI*t),h=(t:number)=>c*Math.sin(Math.PI*t)+d*Math.sin(2*Math.PI*t),yp=(t:number)=>a*Math.PI*Math.cos(Math.PI*t)+2*b*Math.PI*Math.cos(2*Math.PI*t),hp=(t:number)=>c*Math.PI*Math.cos(Math.PI*t)+2*d*Math.PI*Math.cos(2*Math.PI*t);close(r.value,integrate(t=>(yp(t)**2+alpha*y(t)**2)/2));close(r.first,integrate(t=>yp(t)*hp(t)+alpha*y(t)*h(t)));close(r.second,integrate(t=>hp(t)**2+alpha*h(t)**2));close(r.variedValue,integrate(t=>((yp(t)+.1*hp(t))**2+alpha*(y(t)+.1*h(t))**2)/2));
});
test('orthogonal variations have a nonzero quadratic cost while epsilon zero leaves the quotient undefined',()=>{
 const r=functionalVariation(1,0,0,1,.1,0);close(r.first,0);close(r.difference,.01*Math.PI**2);assert.equal(functionalVariation(1,0,1,0,0,0).quotient,null);const tiny=functionalVariation(1,0,1,0,1e-200,0);assert.ok(tiny.difference>0);close(tiny.quotient!,Math.PI**2/2);close(functionalVariation(.2,.3,0,0,.5,2).second,0);
});
test('a perturbation cancelling the entire curve has exactly zero directly evaluated energy',()=>{
 for(const a of [.1,.2,.3,.4])for(const alpha of [0,1,4])assert.equal(functionalVariation(a,0,1,0,-a,alpha).variedValue,0);
});
test('both Euler boundary energies and residual norms match independent quadrature',()=>{
 for(const mode of [0,1])for(const load of [0,2,4]){const a=1.3,b=.2,r=eulerBoundary(mode,load,a,b),yp=(t:number)=>a*(mode===0?1-2*t:1-t)+b*Math.PI*Math.cos(Math.PI*t);close(r.value,integrate(t=>yp(t)**2/2-load*r.y(t)));close(r.residualNorm**2,integrate(t=>r.residual(t)**2));close(r.excess,r.value-r.optimum);close(r.y(0),0);if(mode===0)close(r.y(1),0);else close(r.boundaryResidual!,yp(1));}
});
test('fixed and natural endpoint solutions are distinct minima with matching first-order conditions',()=>{
 const fixed=eulerBoundary(0,2,1,0),free=eulerBoundary(1,2,2,0);close(fixed.value,-1/6);close(free.value,-2/3);close(fixed.residualNorm,0);close(free.residualNorm,0);close(fixed.y(1),0);close(free.y(1),1);assert.equal(fixed.boundaryResidual,null);close(free.boundaryResidual!,0);vector(fixed.coefficientGradient,[0,0]);vector(free.coefficientGradient,[0,0]);
});
test('coefficient gradients include the free-end boundary contribution and match finite differences',()=>{
 for(const mode of [0,1]){const a=1.3,b=.2,h=1e-5,r=eulerBoundary(mode,2,a,b);vector(r.coefficientGradient,[(eulerBoundary(mode,2,a+h,b).value-eulerBoundary(mode,2,a-h,b).value)/(2*h),(eulerBoundary(mode,2,a,b+h).value-eulerBoundary(mode,2,a,b-h).value)/(2*h)],1e-8);}
});
test('area constraint and directional area changes agree with independently integrated sine modes',()=>{
 const A=.5,b=.2,v=-.3,c=.7,r=constrainedVariation(A,b,v,c);close(r.area,integrate(r.y));close(r.areaVariation,integrate(t=>Math.sin(2*Math.PI*t)+c*Math.sin(Math.PI*t)));const yp=(t:number)=>6*A*(1-2*t)+2*Math.PI*b*Math.cos(2*Math.PI*t)+Math.PI*v*Math.cos(Math.PI*t),hp=(t:number)=>2*Math.PI*Math.cos(2*Math.PI*t)+c*Math.PI*Math.cos(Math.PI*t);close(r.value,integrate(t=>yp(t)**2/2));close(r.first,integrate(t=>yp(t)*hp(t)));close(r.lagrangianVariation,r.first+r.multiplier*r.areaVariation);assert.equal(r.certifiedExcess,null);
});
test('constrained minima need only vanish in admissible directions and certify feasible energy gaps',()=>{
 for(const A of [-1,0,.5]){const r=constrainedVariation(A,0,0,1);close(r.first,24*A/Math.PI);close(r.lagrangianVariation,0);assert.equal(r.admissible,false);close(r.multiplier,-12*A);const q=constrainedVariation(A,.2,0,0);assert.equal(q.feasible,true);close(q.certifiedExcess!,.04*Math.PI**2);close(q.value-6*A*A,q.certifiedExcess!);}
});
test('Bernoulli scores and Fisher are obtained by explicitly summing both model outcomes',()=>{
 for(const p of [.02,.2,.5,.98]){const r=fisherExample(p,1,.001,10),scores=[-1/(1-p),1/p],weights=[1-p,p];close(dot(weights,scores),0);close(dot(weights,scores.map(x=>x*x)),r.information);close(p*(1-p),r.logitInformation);close(r.logitInformation,(p*(1-p))**2*r.information);close(r.totalInformation,10*r.information);close(fisherExample(p,0,.001,1).outerProduct,1/(1-p)**2);}
});
test('KL matches its logarithmic definition away from equality and the Fisher quadratic locally',()=>{
 for(const p of [.02,.2,.5,.98])for(const delta of [-.01,.01]){const q=p+delta,actual=p*Math.log(p/q)+(1-p)*Math.log((1-p)/(1-q));close(bernoulliKL(p,q),actual);assert.ok(bernoulliKL(p,q)>=0);}for(const p of [.1,.3,.7]){const h=1e-5,expected=1/(p*(1-p));close((bernoulliKL(p,p+h)+bernoulliKL(p,p-h))/(h*h),expected,1e-7);close(bernoulliKL(p,p),0);}assert.ok(bernoulliKL(.2,1e-17)>0);assert.throws(()=>bernoulliKL(.2,0));
});
test('Fisher diagrams use a local query window and the actual KL vertical scale',()=>{
 const r=evaluate('bernoulli-fisher-coordinates',[.5,1,.01,1]);assert.ok(r.bounds![1]-r.bounds![0]<.03);assert.ok(r.bounds![3]>2e-4&&r.bounds![3]<3e-4);assert.ok(r.bounds![0]>.48);assert.ok(r.bounds![1]<.52);
});
test('one observed score is not the model Fisher and iid counts add information',()=>{
 const r=fisherExample(.2,0,.01,10);close(r.outerProduct,1.5625);close(r.information,6.25);close(r.totalInformation,62.5);assert.notEqual(r.outerProduct,r.information);close(fisherExample(.2,1,.01,10).kl,r.kl);
});
test('natural tangent velocities agree under the logit chain rule but finite updates differ',()=>{
 for(const p of [.1,.2,.5,.9])for(const target of [.1,.8,.9]){const r=naturalStep(p,target,.1,0);close(p*(1-p)*r.logitVelocity,r.velocity);close(r.directional,r.gradient*r.velocity);assert.ok(r.directional<=0);}const r=naturalStep(.2,.8,.1,0);close(r.probabilityEuler,.26);close(r.logitEuler,.26672661861925906);close(r.exactFlow,.8-.6*Math.exp(-.1));assert.ok(Math.abs(r.logitEuler-r.probabilityEuler)>.006);
});
test('finite coordinate discrepancy has a second-order leading term and ordinary GD can exit the domain',()=>{
 const gap=(h:number)=>{const r=naturalStep(.2,.8,h,0);return Math.abs(r.logitEuler-r.probabilityEuler);};assert.ok(gap(.001)/gap(.002)>.249&&gap(.001)/gap(.002)<.251);const r=naturalStep(.1,.9,.5,0);assert.equal(r.ordinaryInterior,false);assert.equal(r.ordinaryExcess,null);close(r.probabilityEuler,.5);for(const mode of [0,1])close(naturalStep(.3,.3,.5,mode).excess,0);
});
test('simplex projection satisfies independent variational inequalities and exact face examples',()=>{
 for(const v of [[.2,.3,.5],[-2,1,3],[0,0,0],[5,-3,1]]){const q=simplexProjection(v);close(q.reduce((s,x)=>s+x,0),1);assert.ok(q.every(x=>x>=0));for(let i=0;i<=10;i++)for(let j=0;j<=10-i;j++){const z=[i/10,j/10,1-i/10-j/10];assert.ok(dot(q.map((x,k)=>x-v[k]),z.map((x,k)=>x-q[k]))>=-1e-10);}}vector(simplexProjection([0,0,0]),[1/3,1/3,1/3]);assert.throws(()=>simplexProjection([1,2]));
});
test('entropic mirror weights match an independent repeated multiplicative calculation',()=>{
 const costs=[0,1,2],r=mirrorRun(costs,1,4);let q=[.2,.3,.5];for(let k=1;k<=4;k++){const weights=q.map((v,i)=>v*Math.exp(-costs[i])),sum=weights.reduce((s,x)=>s+x,0);q=weights.map(v=>v/sum);vector(r.history[k].mirror,q);close(q.reduce((s,x)=>s+x,0),1);assert.ok(q.every(x=>x>0));}vector(r.projected,[1,0,0]);close(r.history.at(-1)!.mirrorGap,.028348174693407793);
});
test('mirror steps preserve cost-shift invariance and initial proportions on a tied optimal face',()=>{
 const a=mirrorRun([-1,0,1],1,4),b=mirrorRun([0,1,2],1,4);vector(a.mirror,b.mirror);const t=mirrorRun([0,0,1],4,12);close(t.mirror[1]/t.mirror[0],1.5);assert.ok(t.mirror[2]>0&&t.mirror[2]<1e-20);vector(mirrorRun([1,1,1],4,12).mirror,[.2,.3,.5]);vector(mirrorRun([0,1,2],0,12).mirror,[.2,.3,.5]);
});
test('all Bregman generators match independently expanded definitions and the three-point identity',()=>{
 for(const mode of [0,1,2])for(const x of [.1,.7,2.8])for(const y of [.2,1.3,3]){const phi=(t:number)=>mode===0?t*t/2:mode===1?t*Math.log(t)-t:-Math.log(t),dp=(t:number)=>mode===0?t:mode===1?Math.log(t):-1/t,r=bregmanExample(mode,x,y,.5);close(r.forward,phi(x)-phi(y)-dp(y)*(x-y));assert.ok(r.forward>=0);close(r.left,r.right);}
});
test('Bregman asymmetry and triangle failure are concrete while tiny gaps remain positive',()=>{
 close(bregman(1,2,1),2*Math.log(2)-1);close(bregman(1,1,2),1-Math.log(2));assert.ok(bregman(0,3,1)>bregman(0,3,2)+bregman(0,2,1));for(const mode of [0,1,2]){const x=1+1e-12,expected=(x-1)**2/2,value=bregman(mode,x,1);assert.ok(value>0);assert.ok(Math.abs(value/expected-1)<1e-10);close(bregman(mode,1,1),0);}
});
test('circle retraction agrees with independently normalized ambient tangent steps',()=>{
 for(const theta of [-2,-.7,0,.8,2])for(const eta of [.1,.5,2]){const x=[Math.cos(theta),Math.sin(theta)],v=[Math.sin(theta)**2,-Math.cos(theta)*Math.sin(theta)],raw=x.map((v0,i)=>v0+eta*v[i]),norm=Math.hypot(...raw),expected=raw.map(v=>v/norm),r=circleRun(0,theta,eta,1);vector(r.last.point,expected);close(r.last.tangency,0);close(Math.hypot(...r.last.point),1);}
});
test('circle exponential matches independent rotation and retraction disagreement is cubic locally',()=>{
 const theta=.8,eta=.5,angle=-eta*Math.sin(theta),x=[Math.cos(theta),Math.sin(theta)],expected=[Math.cos(angle)*x[0]-Math.sin(angle)*x[1],Math.sin(angle)*x[0]+Math.cos(angle)*x[1]];vector(circleRun(1,theta,eta,1).last.point,expected);const a=circleRun(0,theta,.01,1).oneStepDifference,b=circleRun(0,theta,.02,1).oneStepDifference;assert.ok(a/b>.124&&a/b<.126);vector(circleRun(0,Math.PI/2,.5,1).last.point,[1/Math.sqrt(5),2/Math.sqrt(5)]);
});
test('tiny circle-step discrepancies retain their analytic cubic remainder without angle cancellation',()=>{
 const eta=1e-8,theta=.8,r=circleRun(0,theta,eta,1),leading=(eta*Math.sin(theta))**3/3;assert.ok(r.oneStepDifference>0);assert.ok(Math.abs(r.oneStepDifference/leading-1)<1e-12);
});
test('the exact antipodal maximum is stationary and does not escape through floating sin pi',()=>{
 for(const mode of [0,1]){const r=circleRun(mode,Math.PI,.5,20);vector(r.last.point,[-1,0]);close(r.last.gradient,0);close(r.last.excess,2);assert.ok(r.history.every(h=>h.theta===Math.PI));close(circleRun(mode,0,2,20).last.excess,0);}
});
test('adjoint and sensitivity solves match independent elimination and outer finite differences',()=>{
 for(const lambda of [.1,1,2.8])for(const coupling of [-.8,0,.8]){const r=adjointObjective(lambda,coupling,.2,8),H=[[1+lambda,coupling],[coupling,2+lambda]],y=solve(H,[1,-1]),v=solve(H,[y[0]-.2,y[1]]),s=solve(H,y.map(x=>-x));vector(r.y,y);vector(r.exactAdjoint,v);vector(r.sensitivity,s);close(r.exactHyper,dot([y[0]-.2,y[1]],s));const h=1e-5,V=(l:number)=>{const w=solve([[1+l,coupling],[coupling,2+l]],[1,-1]);return ((w[0]-.2)**2+w[1]**2)/2;};close(r.exactHyper,(V(lambda+h)-V(lambda-h))/(2*h),1e-8);}
});
test('approximate adjoint uses actual Richardson steps and its residual bound contains the error',()=>{
 for(const lambda of [0,1,3])for(const coupling of [-.8,0,.8])for(const K of [0,1,8]){const r=adjointObjective(lambda,coupling,.2,K);assert.ok(Math.abs(r.last.hyper-r.exactHyper)<=r.errorBound+1e-12);for(let k=1;k<=K;k++){const old=r.history[k-1].adjoint,Hv=r.H.map(row=>dot(row,old));vector(r.history[k].adjoint,old.map((v,i)=>v+([r.y[0]-.2,r.y[1]][i]-Hv[i])/6));assert.ok(r.history[k].residual<=r.history[k-1].residual+1e-12);}}
 close(adjointObjective(0,0,0,0).exactHyper,-1.125);close(adjointObjective(0,0,0,0).last.hyper,0);
});
test('function-space gradients satisfy independent mass-matrix Riesz pairing and energy quadrature',()=>{
 const r=functionMetricRun(1,4,.1,.2,.3,8),a=r.coefficients,h=[.3,-.7],wave=[Math.PI,4*Math.PI],u=(t:number)=>a[0]*Math.sin(wave[0]*t)+a[1]*Math.sin(wave[1]*t),target=(t:number)=>Math.sin(wave[0]*t)+.5*Math.sin(wave[1]*t),up=(t:number)=>a[0]*wave[0]*Math.cos(wave[0]*t)+a[1]*wave[1]*Math.cos(wave[1]*t),v=(t:number)=>h[0]*Math.sin(wave[0]*t)+h[1]*Math.sin(wave[1]*t),vp=(t:number)=>h[0]*wave[0]*Math.cos(wave[0]*t)+h[1]*wave[1]*Math.cos(wave[1]*t),differential=integrate(t=>(u(t)-target(t))*v(t)+.1*up(t)*vp(t));close(differential,dot(r.L2gradient,h)/2);close(differential,r.metricGradient.reduce((s,g,i)=>s+r.M[i]*g*h[i]/2,0));close(r.directional,-r.normSquared);
 const energy=(coeff:number[])=>integrate(t=>{const val=coeff[0]*Math.sin(wave[0]*t)+coeff[1]*Math.sin(wave[1]*t),der=coeff[0]*wave[0]*Math.cos(wave[0]*t)+coeff[1]*wave[1]*Math.cos(wave[1]*t);return ((val-target(t))**2+.1*der*der)/2;});close(energy(a)-energy(r.optimum),r.history.at(-1)!.excess);
});
test('metric choices preserve the minimizer and match independent modal error powers',()=>{
 for(const mode of [0,1])for(const frequency of [2,4,8]){const r=functionMetricRun(mode,frequency,.1,.2,.3,12);vector(r.coefficients,r.optimum.map((a,i)=>a*(1-(1-.3*r.A[i]/r.M[i])**12)));vector(r.optimum,functionMetricRun(1-mode,frequency,.1,.2,.3,12).optimum);}assert.ok(functionMetricRun(1,4,.1,.2,.3,12).rho<1);assert.ok(functionMetricRun(0,4,.1,.2,.3,12).rho>1);vector(functionMetricRun(1,4,.1,0,.3,12).coefficients,functionMetricRun(0,4,.1,.2,.3,12).coefficients);
});
test('variational domains reject malformed, nonfinite and fractional discrete controls',()=>{
 for(const kind of variationalKinds){const spec=variationalLabs[kind],initial=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));for(let i=0;i<initial.length;i++){for(const v of [NaN,Infinity,spec.controls[i].min-1,spec.controls[i].max+1]){const bad=[...initial];bad[i]=v;assert.throws(()=>evaluate(kind,bad));}if(spec.controls[i].step!=='any'){const bad=[...initial];bad[i]=spec.controls[i].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
});
for(const kind of variationalKinds)test(`${kind}: meaningful guides and finite geometry across parameter boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=variationalLabs[kind],initial=spec.controls.map(c=>c.initial);for(const g of learningGuides[kind].experiments)assert.ok(spec.presets[g.preset]);for(const p of [...spec.presets.map(p=>p.values),...spec.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))]){const r=evaluate(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.bounds!.every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of variationalKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed bilingual theory, valid equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});

test('requested Bernoulli displacements retain KL below the probability coordinate ulp',()=>{
 for(const p of [.02,.2,.5,.98])for(const delta of [-1e-20,1e-20]){const expected=delta*delta/(2*p*(1-p)),actual=bernoulliKLShift(p,delta);assert.equal(p+delta,p);assert.ok(actual>0);assert.ok(Math.abs(actual/expected-1)<2e-14);assert.equal(fisherExample(p,0,delta,1).kl,actual);}
 for(const p of [.02,.2,.5,.98]){assert.equal(bernoulliKLShift(p,0),0);for(const d of [-.01,.01])close(bernoulliKLShift(p,d),bernoulliKL(p,p+d));}
 for(const [p,d] of [[0,0],[1,0],[.2,-.2],[.2,.8],[.5,NaN],[.5,Infinity]])assert.throws(()=>bernoulliKLShift(p,d));
});
test('zero-time natural flow returns the input probability without target subtraction cancellation',()=>{
 for(const p of [.1,.2,.3,.7,.9])for(const r of [.1,.2,.8,.9])assert.equal(naturalStep(p,r,0,0).exactFlow,p);
});
