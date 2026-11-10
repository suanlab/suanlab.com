import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {transportKinds,evaluateTransport as evaluate,twoPlan,mongeExample,dualTransport,monotoneTransport,quantileExample,scalarKL,sinkhorn,entropicTwoPlan,entropicExample,dynamicGaussian,aggregateAtoms,displacementExample,jkoGaussian,dyadicCoupling,transportGridError} from '../src/lib/learning-transport';
import {transportLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const sum=(x:number[])=>x.reduce((s,v)=>s+v,0);
const cost=(P:number[][],C:number[][])=>sum(P.flatMap((row,i)=>row.map((v,j)=>v*C[i][j])));
function marginals(P:number[][],a:number[],b:number[],exact=false){P.flat().forEach(v=>assert.ok(v>=0&&Number.isFinite(v)));P.forEach((row,i)=>exact?assert.equal(sum(row),a[i]):close(sum(row),a[i]));b.forEach((v,j)=>exact?assert.equal(sum(P.map(row=>row[j])),v):close(sum(P.map(row=>row[j])),v));}
function integrate(f:(x:number)=>number,a=0,b=1,N=2000){let s=f(a)+f(b);for(let i=1;i<N;i++)s+=(i%2?4:2)*f(a+(b-a)*i/N);return s*(b-a)/(3*N);}
function bruteTransport(C:number[][],a:number[],b:number[],Q:number){const rows=[...a],cols=[...b],P=C.map(row=>row.map(()=>0));let best=Infinity;function search(index:number){if(index===a.length*b.length){if(rows.every(v=>v===0)&&cols.every(v=>v===0))best=Math.min(best,cost(P,C)/Q);return;}const i=Math.floor(index/b.length),j=index%b.length;for(let n=0;n<=Math.min(rows[i],cols[j]);n++){P[i][j]=n;rows[i]-=n;cols[j]-=n;search(index+1);rows[i]+=n;cols[j]+=n;}}search(0);return best;}
test('the entire two-by-two coupling family preserves both marginals including zero masses',()=>{
 for(const a of [0,.2,.5,1])for(const b of [0,.4,.5,1])for(const t of [0,.3,1]){const r=twoPlan(a,b,t);marginals(r.plan,[a,1-a],[b,1-b]);marginals(r.upper,[a,1-a],[b,1-b]);marginals(r.lower,[a,1-a],[b,1-b]);}
});
test('atomic Monge infeasibility and a strict relaxation benefit have independent map costs',()=>{
 assert.equal(mongeExample(1,.5,1,1).maps.length,0);const r=mongeExample(.4,.6,1,1);assert.equal(r.maps.length,1);vector(r.maps[0].map,[1,0]);close(r.mongeCost!,2.2);close(r.optimum,.6);close(mongeExample(.5,.5,1,1).mongeCost!,.5);for(const mode of [0,1])for(let k=0;k<=100;k++){const q=mongeExample(.4,.6,mode,k/100);assert.ok(q.queryCost>=q.optimum-1e-12);}
});
test('transport dual potentials match direct objectives, inequalities and complementary slackness',()=>{
 for(const a of [0,.2,.8,1])for(const b of [0,.4,.8,1])for(const L of [.5,1,2.7])for(const delta of [-1,0,1]){const r=dualTransport(a,b,L,.3,delta),direct=a*r.u[0]+(1-a)*r.u[1]+b*r.v[0]+(1-b)*r.v[1];close(r.dual,direct);close(r.queryCost-r.dual,r.stableGap);close(r.queryCost-r.optimum,r.excess);r.C.forEach((row,i)=>row.forEach((c,j)=>close(c-r.u[i]-r.v[j],r.slack[i][j])));if(delta<=0){assert.ok(r.slack.flat().every(v=>v>=0));assert.ok(r.certifiedGap!>=r.excess-1e-12);}else assert.equal(r.certifiedGap,null);}
});
test('negative algebraic gaps from infeasible potentials never become certificates',()=>{
 const r=dualTransport(.4,.6,2,1,1);close(r.queryCost-r.dual,-.4);assert.equal(r.dualFeasible,false);assert.equal(r.certifiedGap,null);close(dualTransport(.4,.6,2,1,-1).certifiedGap!,.4);close(dualTransport(.4,.6,2,1,0).certifiedGap!,0);
});
test('monotone quantile transport matches exhaustive integer plans on a three-by-three example',()=>{
 const x=[-1,0,2],y=[-2,.5,1],a=[1,2,1],b=[2,1,1],Q=4;for(const p of [1,2]){const C=x.map(u=>y.map(v=>Math.abs(u-v)**p)),r=monotoneTransport(x,a.map(v=>v/Q),y,b.map(v=>v/Q),p);close(r.cost,bruteTransport(C,a,b,Q));marginals(r.plan,a.map(v=>v/Q),b.map(v=>v/Q));close(r.distance,monotoneTransport(y,b.map(v=>v/Q),x,a.map(v=>v/Q),p).distance);}
});
test('Wasserstein cost powers, equal means, unused atoms and triangle inequality remain distinct',()=>{
 close(quantileExample(.25,.75,0,1,1).distance,1);close(quantileExample(.25,.75,0,1,2).distance,Math.SQRT2);close(quantileExample(.5,.5,0,2,2).distance,1);close(quantileExample(1,1,0,1,2).distance,0);close(monotoneTransport([-1,-1,2],[.2,.3,.5],[-1,2],[.5,.5],2).distance,0);const ab=monotoneTransport([0],[1],[1],[1],2),bc=monotoneTransport([1],[1],[2],[1],2),ac=monotoneTransport([0],[1],[2],[1],2);assert.ok(ac.distance<=ab.distance+bc.distance);assert.ok(ac.cost>ab.cost+bc.cost);
});
test('generalized scalar KL matches its definition and preserves tiny positive remainders',()=>{
 for(const r of [0,.1,.5,2])for(const g of [.1,.5,2])close(scalarKL(r,g),r===0?g:r*Math.log(r/g)-r+g);const r=1+1e-12;assert.ok(scalarKL(r,1)>0);assert.ok(Math.abs(scalarKL(r,1)/((r-1)**2/2)-1)<1e-10);assert.equal(scalarKL(.5,.5),0);assert.throws(()=>scalarKL(1,0));
});
test('log Sinkhorn equals independently repeated ordinary kernel scalings in a well-scaled case',()=>{
 const C=[[0,4],[1,1]],a=[.4,.6],b=[.6,.4],epsilon=.5,r=sinkhorn(C,a,b,epsilon,8),K=C.map(row=>row.map(v=>Math.exp(-v/epsilon)));let u=[1,1],v=[1,1];for(let k=1;k<=8;k++){u=a.map((mass,i)=>mass/sum(K[i].map((x,j)=>x*v[j])));v=b.map((mass,j)=>mass/sum(K.map((row,i)=>row[j]*u[i])));const P=K.map((row,i)=>row.map((x,j)=>u[i]*x*v[j]));vector(r.history[k].plan.flat(),P.flat());assert.ok(r.history[k].columnError<1e-12);assert.ok(r.history[k].dual>=r.history[k-1].dual-1e-12);}
});
test('positive entropic reference satisfies both marginals and independently differentiated cross-ratio stationarity',()=>{
 for(const a of [.1,.5,.9])for(const b of [.1,.6,.9])for(const L of [.5,1,3])for(const epsilon of [.05,.5,2]){const r=entropicTwoPlan(a,b,L,epsilon),P=r.plan;marginals(P,[a,1-a],[b,1-b]);assert.ok(P.flat().every(x=>x>0));close(-2*L+epsilon*(Math.log(P[0][0])+Math.log(P[1][1])-Math.log(P[0][1])-Math.log(P[1][0])),0,1e-9);}
 close(entropicTwoPlan(.5,.5,1,.5).s,1/(2*(1+Math.exp(2))));assert.ok(entropicTwoPlan(.5,.5,3,.05).s>0);
});
test('entropic Fenchel certificates match direct primal minus dual without certifying an infeasible raw plan',()=>{
 for(const steps of [0,1,20]){const r=entropicExample(.4,.6,2,.5,steps);close(r.gap,r.reference.objective-r.last.dual,1e-9);assert.ok(r.gap>=0);assert.ok(r.last.dual<=r.reference.objective+1e-12);}const initial=entropicExample(.4,.6,2,.5,0);assert.ok(initial.last.rowError>.5);assert.ok(initial.last.rawObjective<initial.reference.objective);const symmetric=entropicExample(.5,.5,1,.5,1);close(symmetric.last.rowError,0);close(symmetric.reference.transportCost,.11920292202211755);assert.ok(symmetric.reference.objective<0);
});
test('Gaussian continuity uses independently finite-differenced density and flux derivatives',()=>{
 const h=1e-6;for(const mode of [0,1])for(const t of [.2,.5,.8])for(const x of [-1,0,1]){const r=dynamicGaussian(-.7,.6,.8,1.2,t,mode,x),plus=dynamicGaussian(-.7,.6,.8,1.2,t+h,mode,x),minus=dynamicGaussian(-.7,.6,.8,1.2,t-h,mode,x),dt=(plus.query.density-minus.query.density)/(2*h),dx=(r.evaluate(x+h).flux-r.evaluate(x-h).flux)/(2*h);close(r.query.dt,dt,1e-8);close(r.query.dxFlux,dx,1e-8);close(r.query.residual,0);close(dt+dx,0,1e-8);}
});
test('Gaussian mass, kinetic energy and reparameterized action match independent quadrature',()=>{
 const r=dynamicGaussian(-.7,.6,.8,1.2,.3,0,0);close(integrate(x=>r.evaluate(x).density,-12,12,4000),1);close(integrate(x=>r.evaluate(x).density*r.evaluate(x).velocity**2,-12,12,4000),r.kinetic);for(const mode of [0,1]){const action=integrate(t=>dynamicGaussian(-1,1,.5,1,t,mode,0).kinetic);close(action,mode===0?4.25:17/3);}close(dynamicGaussian(0,0,1,1,.5,0,0).action,0);
});
test('Gaussian particle velocity agrees with a fixed-standardized-coordinate path derivative',()=>{
 const t=.4,z=.7,h=1e-6,r=dynamicGaussian(-.7,.6,.8,1.2,t,1,0),plus=dynamicGaussian(-.7,.6,.8,1.2,t+h,1,0),minus=dynamicGaussian(-.7,.6,.8,1.2,t-h,1,0),x=r.m+r.s*z;close(r.evaluate(x).velocity,((plus.m+plus.s*z)-(minus.m+minus.s*z))/(2*h),1e-8);
});
test('atomic representations merge coincident positions and remove zero masses',()=>{
 assert.deepEqual(aggregateAtoms([0,0,1,2],[.2,.3,.5,0]),{position:[0,1],mass:[.5,.5]});for(const t of [0,1]){const r=displacementExample(.25,.75,0,1,t,1);vector(r.displacement.position,[-1,1]);vector(r.displacement.mass,t===0?[.25,.75]:[.75,.25]);assert.equal(r.mixture.position.length,2);}
});
test('optimal displacement paths obey independently recomputed geodesic distances',()=>{
 for(const a of [0,.25,.8,1])for(const b of [0,.75,1])for(const t of [0,.2,.5,1]){const r=displacementExample(a,b,.7,1.3,t,0);close(r.fromSource.distance,t*r.optimal.distance);close(r.toTarget.distance,(1-t)*r.optimal.distance);close(r.displacementMean,r.mixtureMean);close(sum(r.displacement.mass),1);}
});
test('mixture and displacement differ at a Dirac midpoint and product coupling need not be geodesic',()=>{
 const dirac=displacementExample(1,1,2,1,.5,0);close(dirac.displacementVariance,0);close(dirac.mixtureVariance,1);close(dirac.fromSource.distance,1);close(dirac.mixtureFromSource.distance,Math.SQRT2);assert.equal(dirac.displacement.mass.length,1);const product=displacementExample(.5,.5,0,1,.5,1);close(product.optimal.distance,0);close(product.couplingCost,2);close(product.fromSource.distance,Math.SQRT1_2);vector(product.displacement.mass,[.25,.5,.25]);
});
test('Gaussian JKO steps satisfy independent proximal first-order equations and energy competition',()=>{
 for(const T of [0,.2,2])for(const m0 of [-2,0,1])for(const s0 of [.3,1,2])for(const h of [.05,.5,1]){const r=jkoGaussian(m0,s0,T,h,12);for(let k=1;k<r.history.length;k++){const old=r.history[k-1],now=r.history[k];close((now.m-old.m)/h+now.m,0);close((now.s-old.s)/h+now.s-T/now.s,0);assert.ok(now.s>0);assert.ok(now.energy+now.movement<=old.energy+1e-10);}assert.ok(r.dissipation<=r.energyDrop+1e-10);}
});
test('the JKO Gaussian minimizes independently sampled mean-scale competitors',()=>{
 const m0=1,s0=.6,T=1,h=.5,r=jkoGaussian(m0,s0,T,h,1),objective=(m:number,s:number)=>((m-m0)**2+(s-s0)**2)/(2*h)+(m*m+s*s)/2-T*Math.log(s),best=objective(r.last.m,r.last.s);for(let i=-20;i<=20;i++)for(let j=1;j<=40;j++)assert.ok(objective(i/10,j/20)>=best-1e-12);
});
test('JKO matches hand calculations, equilibrium and fixed-physical-time refinement',()=>{
 const r=jkoGaussian(1,1,1,.5,2);close(r.last.m,4/9);close(r.last.s,1);close(r.reference.m,Math.exp(-1));close(r.distance,.07656500327300209);close(jkoGaussian(1,1,0,.5,2).last.s,4/9);close(jkoGaussian(0,1,1,.5,4).dissipation,0);const zero=jkoGaussian(1,.3,2,.5,0);close(zero.distance,0);assert.equal(zero.last.meanResidual,null);const errors=[1,.5,.25,.125].map(h=>jkoGaussian(1,1,1,h,Math.round(1/h)).distance);errors.slice(1).forEach((e,i)=>assert.ok(e<errors[i]));
});
test('integer capacity correction has exactly binary-feasible marginals for arbitrary nonnegative matrices',()=>{
 for(const N of [2,4,8,16])for(const mode of [0,1,2]){const P=Array.from({length:N},(_,i)=>Array.from({length:N},(_,j)=>mode===0?0:mode===1?Math.exp(-Math.abs(i-j)):(i+1)*(j+2)/3)),r=dyadicCoupling(P),mass=Array(N).fill(1/N);marginals(r,mass,mass,true);assert.ok(r.flat().every(v=>Number.isInteger(v*2**24)));}assert.throws(()=>dyadicCoupling([[1,0,0],[0,1,0],[0,0,1]]));
});
test('uniform midpoint quantization and exact translated-pair distances have independent integral answers',()=>{
 for(let k=1;k<=4;k++){const r=transportGridError(k,.5,.1,1),N=2**k;let squared=0;for(let i=0;i<N;i++){const mid=(i+.5)/N;squared+=((i/N+1/N-mid)**3-(i/N-mid)**3)/3;}close(r.quantization,Math.sqrt(squared));close(monotoneTransport(r.x,Array(N).fill(1/N),r.y,Array(N).fill(1/N),2).cost,.25);assert.ok(r.pairDistanceBound>0);}
});
test('feasible grid plans obey entropy primal-dual identities and the regularization-plus-gap bound',()=>{
 for(const k of [1,3,4])for(const shift of [-1,0,.5])for(const epsilon of [.05,.3,1])for(const steps of [0,1,20]){const r=transportGridError(k,shift,epsilon,steps),h=r.last,N=r.N;marginals(h.feasible,Array(N).fill(1/N),Array(N).fill(1/N),true);close(h.objective-h.dual,h.gap,1e-9);assert.ok(h.gap>=0);assert.ok(h.excess>=0&&h.excess<=h.costBound+1e-10);const C=r.x.map(x=>r.y.map(y=>(x-y)**2));close(cost(h.feasible,C)-shift*shift,h.excess);const entropy=-sum(h.feasible.flat().map(v=>v===0?0:v*Math.log(v)));assert.ok(entropy>=Math.log(N)-1e-10&&entropy<=2*Math.log(N)+1e-10);}
});
test('zero-iteration feasibility correction is distinct from a solved regularized objective',()=>{
 const a=transportGridError(3,.5,.1,0),b=transportGridError(3,.5,.1,20);assert.ok(a.last.rowError>1);assert.ok(a.last.gap>1);assert.ok(b.last.gap<a.last.gap);assert.ok(a.quantization>0);close(a.exactSquaredDistance,.25);close(transportGridError(4,.5,.1,20).quantization,a.quantization/2);
});
test('transport model domains reject malformed, nonfinite and fractional discrete controls',()=>{
 for(const kind of transportKinds){const spec=transportLabs[kind],initial=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));for(let i=0;i<initial.length;i++){for(const v of [NaN,Infinity,spec.controls[i].min-1,spec.controls[i].max+1]){const bad=[...initial];bad[i]=v;assert.throws(()=>evaluate(kind,bad));}if(spec.controls[i].step!=='any'){const bad=[...initial];bad[i]=spec.controls[i].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
});
for(const kind of transportKinds)test(`${kind}: meaningful guides and finite geometry across parameter boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=transportLabs[kind],initial=spec.controls.map(c=>c.initial);for(const g of learningGuides[kind].experiments)assert.ok(spec.presets[g.preset]);for(const p of [...spec.presets.map(p=>p.values),...spec.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))]){const r=evaluate(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.bounds!.every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}if(r.flow)assert.ok(r.flow.plan.flat().every(v=>v>=0&&Number.isFinite(v)));for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of transportKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed bilingual theory, valid equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});
