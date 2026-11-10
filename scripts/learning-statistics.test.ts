import test from 'node:test';
import assert from 'node:assert/strict';
import { statisticsKinds, evaluateStatistics as evaluate, finiteSamplingLaw, empiricalQuantile, normalQuantile } from '../src/lib/learning-statistics';
import { normalCdf } from '../src/lib/learning-probability';
import { statisticsLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const integrate=(f:(x:number)=>number,lo:number,hi:number,N=8000)=>{const h=(hi-lo)/N;let sum=f(lo)+f(hi);for(let i=1;i<N;i++)sum+=(i%2?4:2)*f(lo+i*h);return sum*h/3;};
test('estimators keep fixed observed data separate from the parameter and repeated-sampling law',()=>{
 const a=evaluate('samples-parameters-estimators',[4,.5,3]),b=evaluate('samples-parameters-estimators',[4,.8,3]);assert.deepEqual(a.stats['observed sample'],[1,1,0,0]);assert.equal(a.stats['observed estimate'],b.stats['observed estimate']);assert.notEqual(a.stats['E[estimator]'],b.stats['E[estimator]']);assert.equal(a.stats['probability of ordered sample'],1/16);assert.equal(a.stats['probability of observed count'],6/16);
 for(let n=2;n<=12;n++){const r=evaluate('samples-parameters-estimators',[n,.5,4095]);assert.equal(r.stats['observed count'],n);assert.equal(r.stats['observed estimate'],1);}
});
test('shrinkage risk agrees with independent enumeration of binary samples',()=>{
 for(let n=1;n<=7;n++)for(const p of [0,.2,.5,.8,1])for(const lambda of [0,.5,1]){let mean=0,second=0,risk=0;for(let mask=0;mask<2**n;mask++){let k=0;for(let j=0;j<n;j++)k+=(mask>>j)&1;const probability=p**k*(1-p)**(n-k),T=(1-lambda)*k/n+lambda/2;mean+=probability*T;second+=probability*T*T;risk+=probability*(T-p)**2;}const r=evaluate('bias-variance-consistency',[n,p,lambda]);close(Number(r.stats['E[Tλ]']),mean);close(Number(r.stats.variance),second-mean*mean);close(Number(r.stats.MSE),risk);}
 assert.equal(evaluate('bias-variance-consistency',[30,.8,.5]).stats['1{consistent for every p at fixed λ}'],0);
 const point=evaluate('bias-variance-consistency',[10,.8,1]).curves[1];assert.equal(point.length,1);close(point[0][1],1);
});
test('MLE and MAP lie at actual profile peaks and posterior means agree with numerical Beta integration',()=>{
 for(const [k,a,b] of [[7,2,2],[0,1,1],[10,1,1],[2,4,1]]){const r=evaluate('likelihood-mle-map',[k,a,b]);close(Math.max(...r.curves[0].map(v=>v[1])),1);close(Math.max(...r.curves[1].map(v=>v[1])),1);const f=(p:number)=>p**(k+a-1)*(1-p)**(10-k+b-1),norm=integrate(f,0,1),mean=integrate(p=>p*f(p),0,1)/norm;close(Number(r.stats['posterior mean']),mean,1e-8);}
 const r=evaluate('likelihood-mle-map',[0,1,1]);assert.equal(r.stats.MLE,0);assert.equal(r.stats.MAP,0);close(Number(r.stats['posterior mean']),1/12);
});
test('sufficiency conditional masses normalize and remain parameter-independent where conditioning is possible',()=>{
 for(let t=0;t<=4;t++)for(const p of [0,.2,.5,.8,1]){const r=evaluate('exponential-families-sufficiency',[t,p]);close(r.table.rows.reduce((s,row)=>s+Number(row[2]),0),1);if(Number(r.stats['P(T=t)'])>0){close(r.table.rows.reduce((s,row)=>s+Number(row[3]),0),1);for(const row of r.table.rows)if(row[1]===t)close(Number(row[3]),1/Number(r.stats['ordered samples with T=t']));}else{assert.equal(r.stats['conditional mass of each matching sample'],null);assert.ok(r.table.rows.every(row=>row[3]==='—'));}}
 for(const p of [.2,.5,.8]){const r=evaluate('exponential-families-sufficiency',[2,p]),h=1e-6;let information=0;for(const row of r.table.rows){const k=Number(row[1]),log=(q:number)=>k*Math.log(q)+(4-k)*Math.log1p(-q),score=(log(p+h)-log(p-h))/(2*h);information+=Number(row[2])*score*score;}close(information,Number(r.stats['Fisher information in p']),1e-6);}
});
test('known-variance confidence coverage agrees with normal integration and tests match interval exclusion',()=>{
 for(const mode of [0,1]){const reference=evaluate('confidence-intervals-tests',[16,.5,1,mode]),critical=Number(reference.stats['critical |z|']),coverage=integrate(x=>Math.exp(-x*x/2)/Math.sqrt(2*Math.PI),-critical,critical);close(coverage,Number(reference.stats['confidence level']),1e-7);
  for(const n of [2,16,64])for(const mean of [-1,0,.3,1]){const r=evaluate('confidence-intervals-tests',[n,mean,1,mode]),[lo,hi]=r.stats['confidence interval'] as number[];assert.equal(r.stats['1{reject H0}'],Number(lo>0||hi<0));}}
 assert.equal(evaluate('confidence-intervals-tests',[16,0,1,0]).stats['two-sided p-value (approx.)'],1);
});
test('bootstrap enumeration agrees with an independent convolution and quantile convention',()=>{
 const r=evaluate('bootstrap-resampling',[0,0,0]),expected=[1,3,6,7,6,3,1];assert.deepEqual(r.table.rows.map(row=>row[1]),expected);r.table.rows.forEach((row,i)=>close(Number(row[0]),i/3));close(Number(r.stats['bootstrap standard error']),Math.sqrt(2/9));assert.deepEqual(r.stats['percentile interval'],[1/3,5/3]);
 const med=evaluate('bootstrap-resampling',[1,1,1]);close(Number(med.stats['bootstrap bias estimate']),7/9);assert.equal(evaluate('bootstrap-resampling',[2,0,1]).stats['bootstrap standard error'],0);
 assert.equal(empiricalQuantile([0,1,2,3],.5),1);assert.equal(empiricalQuantile([0,1,2,3],0),0);assert.equal(empiricalQuantile([0,1,2,3],1),3);
});
test('OLS satisfies normal equations, projection leverage and correct rank degrees of freedom',()=>{
 for(let change=-2;change<=4;change++)for(let mode=0;mode<3;mode++){const r=evaluate('regression-residuals-diagnostics',[change,mode]);for(const v of r.stats['Xᵀ residual'] as number[])close(v,0);close(r.table.rows.reduce((s,row)=>s+Number(row[5]),0),Number(r.stats['design rank']));assert.equal(r.stats['residual degrees of freedom'],3-Number(r.stats['design rank']));close(r.table.rows.reduce((s,row)=>s+Number(row[4])**2,0),Number(r.stats.SSE));}
 close(Number(evaluate('regression-residuals-diagnostics',[0,0]).stats.SSE),1/6);assert.equal(evaluate('regression-residuals-diagnostics',[0,1]).stats['residual sum'],4);assert.equal(evaluate('regression-residuals-diagnostics',[0,2]).stats['1{coefficients identifiable}'],0);
});
test('nonidentified regression coefficients differ along a null direction without changing fitted values',()=>{
 const r=evaluate('regression-residuals-diagnostics',[0,2]),[a,b]=r.stats['coefficients (intercept,slope)'] as number[];for(const t of [-2,0,2])for(const row of r.table.rows)close((a+t)+(b-t)*Number(row[1]),Number(row[3]));
});
test('delta exact moments agree with integration of the specified Gaussian while first-order degeneracy remains visible',()=>{
 for(const n of [1,10,64])for(const mu of [0,1])for(const mode of [0,1]){const r=evaluate('asymptotic-delta-method',[n,mu,mode]),g=(x:number)=>mode===0?x*x:Math.exp(x),phi=(z:number)=>Math.exp(-z*z/2)/Math.sqrt(2*Math.PI),mean=integrate(z=>g(mu+z/Math.sqrt(n))*phi(z),-9,9),second=integrate(z=>g(mu+z/Math.sqrt(n))**2*phi(z),-9,9);close(mean,Number(r.stats['exact expected transformed estimate']),1e-6);close(second-mean*mean,Number(r.stats['exact finite-n variance']),1e-6);}
 const zero=evaluate('asymptotic-delta-method',[10,0,0]);assert.equal(zero.stats['first-order delta variance'],0);assert.equal(zero.stats['exact finite-n variance'],.02);
});
test('power matches independent tail integration and Bonferroni bounds the independent all-null error',()=>{
 for(const correction of [0,1])for(const m of [1,10,20]){const r=evaluate('power-multiple-comparisons',[16,.5,m,correction]),critical=Number(r.stats['critical z']),power=integrate(z=>Math.exp(-((z-2)**2)/2)/Math.sqrt(2*Math.PI),critical,12);close(power,Number(r.stats['power for one alternative']),1e-6);if(correction)assert.ok(Number(r.stats['all-null FWER under independence'])<=.05+1e-12);}
 const nullTest=evaluate('power-multiple-comparisons',[16,0,10,1]);close(Number(nullTest.stats['power for one alternative']),.005,1e-6);
 close(normalQuantile(.975),1.959963984540054,1e-6);for(const p of [.05,.5,.95,.995])close(normalCdf(normalQuantile(p)),p,1e-6);
});
test('design-aware variances equal direct covariance-matrix sums for pairs and clusters',()=>{
 for(const n of [2,10,30])for(const rho of [0,.5,1])for(const k of [1,4,8]){let sum=0;for(let i=0;i<k;i++)for(let j=0;j<k;j++)sum+=i===j?1:rho;const cluster=evaluate('experimental-design-error',[n,rho,k,2]);close(Number(cluster.stats['error variance']),sum/(n*k*k));if(rho===1)close(Number(cluster.stats['effective sample size for cluster mean']),n);const paired=evaluate('experimental-design-error',[n,rho,k,0]);close(Number(paired.stats['error variance']),(1+1-2*rho)/n);}
});
test('finite law and quantile helpers reject invalid data instead of silently renormalizing zero mass',()=>{
 assert.throws(()=>finiteSamplingLaw([0],[0]));assert.throws(()=>finiteSamplingLaw([0],[NaN]));assert.throws(()=>empiricalQuantile([], .5));assert.throws(()=>empiricalQuantile([1,2],1.1));assert.throws(()=>normalQuantile(0));assert.throws(()=>normalQuantile(1));assert.deepEqual(finiteSamplingLaw([1,1],[.5,.5]),[[1,1]]);
});
for(const kind of statisticsKinds)test(`statistics presets, explanations and finite diagram geometry: ${kind}`,()=>{
 const spec=statisticsLabs[kind];assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.table.rows.length>0);}assert.throws(()=>evaluate(kind,[]));const values=spec.controls.map(c=>c.initial);values[0]=NaN;assert.throws(()=>evaluate(kind,values));values[0]=spec.controls[0].min-.5;assert.throws(()=>evaluate(kind,values));
});
test('tiny positive probabilities do not turn possible sufficient-statistic conditions into null events',()=>{
 const r=evaluate('exponential-families-sufficiency',[4,1e-200]);assert.equal(r.stats['P(T=t)'],0);assert.equal(r.stats['conditional mass of each matching sample'],1);assert.equal(r.warning,'numeric-range');
 const overflow=evaluate('exponential-families-sufficiency',[0,5e-324]);assert.equal(overflow.stats['Fisher information in p'],null);assert.equal(overflow.warning,'numeric-range');
});
