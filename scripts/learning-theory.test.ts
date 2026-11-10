import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {theoryLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
import type {TheoryKind} from '../src/lib/learning-theory';
import {thresholdRisk,errorDecomposition,thresholdCapacity,posteriorBound,ridgeStability,implicitLinear,minimumNormFit,interpolationRisk,calibrationExperiment,groupShift,binomialProbabilities,binomialTail,sampleComplexity,scalingExperiment,evaluateTheory,theoryKinds} from '../src/lib/learning-theory';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0);
const multiply=(A:number[][],B:number[][])=>A.map(row=>B[0].map((_,j)=>sum(row.map((v,k)=>v*B[k][j]))));
const transpose=(A:number[][])=>A[0].map((_,j)=>A.map(row=>row[j]));
const matrixClose=(A:number[][],B:number[][],tol=1e-9)=>A.forEach((row,i)=>row.forEach((v,j)=>close(v,B[i][j],tol)));
test('finite population threshold risks match independent joint-distribution enumeration',()=>{
 for(const noise of [0,.1,.45])for(const seed of [1,7,999]){const r=thresholdRisk(32,noise,seed,2);for(let t=0;t<=5;t++){let risk=0;for(let x=0;x<5;x++)for(const y of [0,1]){const q=x>=2?1-noise:noise;risk+=(y?q:1-q)/5*Number(Number(x>=t)!==y);}close(r.population[t],risk);close(r.empirical[t],r.samples.filter(s=>Number(s.x>=t)!==s.y).length/32);}assert.equal(r.oracle,2);close(r.population[2],noise);assert.equal(r.erm,r.empirical.indexOf(Math.min(...r.empirical)));}
});
test('seeded ERM samples are reproducible and preserve prefixes without changing true risk',()=>{
 const a=thresholdRisk(8,.2,7,0),b=thresholdRisk(16,.2,7,5);assert.deepEqual(a.samples,b.samples.slice(0,8));assert.deepEqual(a.population,b.population);const n=thresholdRisk(1,0,7,2);assert.ok(n.empirical.filter(x=>x===Math.min(...n.empirical)).length>1);assert.equal(n.erm,n.empirical.indexOf(Math.min(...n.empirical)));
});
test('population approximation, estimation and signed optimization effects sum to exact excess risk',()=>{
 for(const mode of [0,1])for(const n of [2,8,32])for(const seed of [1,7,19])for(const steps of [0,1,40]){const r=errorDecomposition(n,1,.5,mode,steps,.8,seed);close(r.approximation+r.estimation+r.optimizationEffect,r.excess);assert.ok(r.approximation>=-1e-12&&r.estimation>=-1e-12&&r.empiricalGap>=-1e-12);close(r.population(r.oracle)-r.bayes,mode===0?4/9:1/3);for(let k=1;k<r.path.length;k++)close(r.path[k],r.path[k-1]-.8*(r.A*r.path[k-1]-r.B));}
});
test('early optimization can improve population risk relative to the sample ERM',()=>{
 let found=false;for(let seed=1;seed<=100;seed++){const r=errorDecomposition(2,0,1,1,1,.5,seed);if(r.optimizationEffect<-.01)found=true;}assert.equal(found,true);
});
test('threshold growth and exact Rademacher complexities match independent subset maxima',()=>{
 for(let n=1;n<=10;n++){const r=thresholdCapacity(n,1023);assert.equal(r.patterns,n+1);assert.equal(r.shattered,n===1);let expected=0;for(let mask=0;mask<2**n;mask++){const signs=Array.from({length:n},(_,i)=>((mask>>i)&1)?1:-1);let best=-Infinity;for(let t=0;t<=n;t++){let s=0;for(let i=0;i<n;i++)s+=signs[i]*(i<t?-1:1);best=Math.max(best,s);}expected+=best/(n*2**n);}close(r.rademacher,expected);assert.ok(r.rademacher<=r.finiteClassBound+1e-12);}close(thresholdCapacity(1,0).rademacher,1);close(thresholdCapacity(2,1).rademacher,.75);assert.equal(thresholdCapacity(2,1).minMistakes,1);assert.equal(thresholdCapacity(2,2).minMistakes,0);
});
test('PAC-Bayes posterior, KL and grid penalty use their actual finite definitions',()=>{
 for(const tilt of [-1,0,1])for(const beta of [0,5,20])for(const k of [0,5,7]){const r=posteriorBound(32,.1,7,beta,k,.05,tilt);close(sum(r.prior),1);close(sum(r.posterior),1);if(beta===0){r.prior.forEach((v,i)=>close(v,r.posterior[i]));close(r.KL,0);}const unnorm=r.prior.map((v,i)=>v*Math.exp(-beta*r.risk.empirical[i])),z=sum(unnorm);unnorm.forEach((v,i)=>close(v/z,r.posterior[i]));close(r.confidence,Math.log(160));close(r.rawUpper,r.empirical+(r.KL+Math.log(160))/2**k+2**k/(8*32));assert.ok(r.upper<=1);assert.ok(r.KL>=-1e-12);}
});
test('Hoeffding moment and finite change-of-measure inequalities independently support the bound derivation',()=>{
 for(const R of [0,.1,.3,.9,1])for(const n of [1,8,64])for(const lambda of [1,32,128]){const mgf=n*Math.log(R*Math.exp(lambda*(R-1)/n)+(1-R)*Math.exp(lambda*R/n));assert.ok(mgf<=lambda*lambda/(8*n)+1e-9);}const r=posteriorBound(16,.2,19,8,4,.1,.5),values=r.risk.population.map((v,i)=>16*(v-r.risk.empirical[i])),max=Math.max(...values),logPriorMGF=max+Math.log(sum(values.map((v,i)=>r.prior[i]*Math.exp(v-max))));assert.ok(sum(values.map((v,i)=>r.posterior[i]*v))-r.KL<=logPriorMGF+1e-12);
});
test('replace-one ridge endpoint supremum matches a dense independent test-label sweep',()=>{
 for(const N of [1,2,8,32])for(const lambda of [0,1,4])for(const a of [-1,0,1])for(const b of [-1,.3,1]){const r=ridgeStability(N,lambda,a,b,.3);close(r.w,sum(r.data)/(N*(1+lambda)));close(r.v,sum(r.neighbor)/(N*(1+lambda)));let maximum=0;for(let j=0;j<=200;j++){const y=-1+j/100,difference=((r.v-y)**2-(r.w-y)**2)/2;maximum=Math.max(maximum,Math.abs(difference));}close(r.observedBeta,maximum);assert.ok(r.observedBeta<=r.uniformBeta+1e-12);assert.ok(r.predictionChange<=r.predictionBound+1e-12);}close(ridgeStability(2,1,-1,1,0).observedBeta,.5625);close(ridgeStability(1,0,-1,1,1).observedBeta,2);
});
test('linear GD preserves the initial null component and converges to the appropriate solution',()=>{
 for(const a of [.5,1,2])for(const b of [.5,1,2])for(const start of [[0,0],[1,-1]]){const eta=Math.min(1,.8/(a*a+b*b)),r=implicitLinear(a,b,1,start[0],start[1],eta,40);for(const w of r.path)close(r.nullProjection(w),r.initialNull);r.path.at(-1)!.forEach((v,i)=>close(v,r.limit[i]));if(start[0]===0)r.path.at(-1)!.forEach((v,i)=>close(v,r.minimum[i]));for(const w of r.constraint)close(a*w[0]+b*w[1],1);assert.equal(r.convergent,true);}const r=implicitLinear(1,1,1,0,0,1,4);assert.equal(r.convergent,false);close(r.contraction,-1);assert.ok(r.loss(r.path.at(-1)!)>0);
});
test('minimum-norm fits match hand solutions including rank deficiency and explicit truncation',()=>{
 let r=minimumNormFit([[1,1]],[1]);close(r.weights[0],.5);close(r.weights[1],.5);assert.equal(r.rank,1);r=minimumNormFit([[1,0],[0,1],[1,1]],[1,2,3]);close(r.weights[0],1);close(r.weights[1],2);r=minimumNormFit([[1,1],[2,2]],[1,2]);close(r.weights[0],.5);close(r.weights[1],.5);r=minimumNormFit([[1,0],[0,1e-11]],[1,1]);assert.equal(r.rank,1);close(r.weights[1],0);r=minimumNormFit([[0,0],[0,0]],[1,2]);assert.equal(r.rank,0);assert.equal(r.condition,null);assert.deepEqual(r.weights,[0,0]);
});
test('Jacobi pseudoinverses satisfy all four independent Moore-Penrose identities',()=>{
 for(const X of [[[1,1]],[[1,2],[2,4]],[[1,0],[0,1],[1,1]],[[1,2,0],[0,1,1]],[[0,0],[0,0]]]){const n=X.length,P=transpose(Array.from({length:n},(_,j)=>minimumNormFit(X,Array.from({length:n},(_,i)=>Number(i===j))).weights)),XP=multiply(X,P),PX=multiply(P,X);matrixClose(multiply(XP,X),X);matrixClose(multiply(PX,P),P);matrixClose(XP,transpose(XP));matrixClose(PX,transpose(PX));}
});
test('nested Gaussian-design fits have actual rank, residual and exact conditional test risk',()=>{
 for(const N of [2,4,8])for(const sigma of [0,.3,1])for(const seed of [1,7,19,999]){const r=interpolationRisk(N,sigma,seed,12);for(const level of r.levels){assert.equal(level.rank,Math.min(N,level.p));const residual=r.X.map((row,i)=>sum(row.slice(0,level.p).map((v,j)=>v*level.weights[j]))-r.y[i]);close(level.training,sum(residual.map(v=>v*v))/N,1e-7);const risk=Array.from({length:12},(_,j)=>(level.weights[j]??0)-Number(j===0)).reduce((s,v)=>s+v*v,sigma*sigma);close(level.population,risk);assert.ok(level.population>=sigma*sigma-1e-10);assert.ok(Math.max(0,...level.normal.map(Math.abs))<1e-6*(1+Math.max(...r.y.map(Math.abs))));if(level.p>=N)assert.ok(level.training<1e-10);}const again=interpolationRisk(N,sigma,seed,3);assert.deepEqual(r.X,again.X);assert.deepEqual(r.y,again.y);}
});
test('Brier decomposition and bin-mean variance are exact Bernoulli expectations',()=>{
 const r=calibrationExperiment(.15,.5,.85,1,20,7,0);close(r.brier,r.reliability+r.intrinsic);assert.equal(r.reliability,0);assert.equal(r.exactECE,0);assert.deepEqual(r.predictions,[.15,.5,.85]);close(r.brier,(.1275+.25+.1275)/3);for(let i=0;i<3;i++){const p=r.predictions[i],q=r.truth[i];close(q*(1-p)**2+(1-q)*p*p,(p-q)**2+q*(1-q));close(r.meanEstimatorVariance[i],q*(1-q)/20);}close(r.empiricalBrier,sum(r.labels.flatMap((ys,i)=>ys.map(y=>(r.predictions[i]-y)**2)))/(3*20));
});
test('merged bins hide opposite calibration errors and temperature changes confidence without changing accuracy',()=>{
 const fine=calibrationExperiment(.85,.5,.15,1,20,7,0),merged=calibrationExperiment(.85,.5,.15,1,20,7,1);close(fine.exactECE,1.4/3);close(merged.selectedECE,0);close(fine.brier,merged.brier);assert.deepEqual(fine.labels,merged.labels);assert.equal(merged.points.length,1);const cold=calibrationExperiment(.15,.5,.85,.5,20,7,0),warm=calibrationExperiment(.15,.5,.85,3,20,7,0);close(cold.accuracy,warm.accuracy);assert.notEqual(cold.brier,warm.brier);
});
test('total-variation robust risks match an independent dense ambiguity-interval search',()=>{
 for(const alpha of [0,.2,.5,.9,1])for(const radius of [0,.2,1]){const r=groupShift(alpha,.7,radius,16,7,0);for(let h=0;h<3;h++){let maximum=0;for(let j=0;j<=100;j++)maximum=Math.max(maximum,r.risk(h,r.lo+(r.hi-r.lo)*j/100));close(r.worst[h],maximum);}assert.equal(r.robust,r.worst.indexOf(Math.min(...r.worst)));close(r.hi-r.lo,Math.min(1,alpha+radius)-Math.max(0,alpha-radius));}assert.equal(groupShift(.9,.1,.8,16,7,1).robust,2);
});
test('importance weights recover target risk in expectation only under target support',()=>{
 for(const alpha of [.1,.4,.9])for(const beta of [0,.2,1])for(const h of [0,1,2]){const r=groupShift(alpha,beta,.2,8,7,h);close(sum(r.P.map((p,i)=>p*r.weights[i]!*r.losses[h][i])),r.target);assert.equal(r.support,true);}const missing=groupShift(1,0,.5,8,7,0);assert.equal(missing.support,false);assert.equal(missing.importance,null);assert.equal(missing.weights[0],null);assert.equal(groupShift(1,1,0,8,7,0).support,true);
});
test('binomial probabilities match factorial enumeration and remain normalized at extreme p',()=>{
 for(const N of [1,2,8,32,128])for(const p of [0,.000001,.1,.5,.9,.999999,1]){const pmf=binomialProbabilities(N,p);close(sum(pmf),1,1e-11);if(N<=8){let factorial=1;for(let k=1;k<=N;k++)factorial*=k;for(let k=0;k<=N;k++){let a=1,b=1;for(let j=1;j<=k;j++)a*=j;for(let j=1;j<=N-k;j++)b*=j;close(pmf[k],factorial/(a*b)*p**k*(1-p)**(N-k));}}}
});
test('inclusive tail boundaries and nonmonotone finite-n behavior are not smoothed away',()=>{
 close(binomialTail(2,.5,.5).tail,.5);close(binomialTail(3,.5,.2).tail,.25);close(binomialTail(4,.5,.2).tail,.625);assert.equal(binomialTail(4,.5,.25).lo,1);assert.equal(binomialTail(4,.5,.25).hi,3);for(const N of [1,8,128]){close(binomialTail(N,0,.1).tail,0);close(binomialTail(N,1,.1).tail,0);}
});
test('Hoeffding bounds dominate exact finite tails and sufficient counts meet the requested confidence',()=>{
 for(const p of [0,.1,.5,.9,1])for(const epsilon of [.01,.2,.5]){const r=sampleComplexity(128,p,epsilon,.05);for(const row of r.levels)assert.ok(row.tail<=row.bound+1e-11);assert.ok(2*Math.exp(-2*r.sufficient*epsilon*epsilon)<=.05+1e-12);}assert.equal(sampleComplexity(8,.5,.2,.05).sufficient,47);
});
test('nested repetition curves are actual squared estimator errors plus a declared analytic floor',()=>{
 for(const floor of [0,1]){const r=scalingExperiment(5,20,1,.2,7,floor);for(const row of r.levels){const empirical=sum(r.paths.map(path=>(path[row.n-1]/row.n+.2)**2+floor))/20;close(row.observed,empirical);close(row.theory,.04+1/row.n+floor);}const x=r.levels.map(row=>Math.log(row.n)),y=r.levels.map(row=>Math.log(row.observed)),xm=sum(x)/x.length,ym=sum(y)/y.length,slope=sum(x.map((v,i)=>(v-xm)*(y[i]-ym)))/sum(x.map(v=>(v-xm)**2));close(r.slope!,slope);}const small=scalingExperiment(3,10,1,0,7,0),large=scalingExperiment(3,20,1,0,7,0);assert.deepEqual(small.paths,large.paths.slice(0,10));
});
test('Rademacher sample means have independently enumerated sigma-squared over n risk',()=>{
 for(const n of [2,4,8]){let expectation=0;for(let mask=0;mask<2**n;mask++){let s=0;for(let i=0;i<n;i++)s+=((mask>>i)&1)?1:-1;expectation+=(s/n)**2/2**n;}close(expectation,1/n);}const zero=scalingExperiment(7,2,0,0,7,0);assert.equal(zero.slope,null);assert.equal(zero.validPoints,0);assert.equal(scalingExperiment(1,20,1,.2,7,0).slope,null);const floor=scalingExperiment(7,20,1,1,7,1);assert.ok(floor.levels.at(-1)!.theory>2);
});
const initial:Record<TheoryKind,number[]>={'finite-risk-empirical-minimization':[12,.1,7,2],'approximation-estimation-optimization-risk':[8,1,.5,1,10,.5,7],'vc-threshold-rademacher':[4,5],'finite-posterior-pac-bayes':[32,.1,7,10,5,.05,0],'replace-one-ridge-stability':[8,1,-1,1,0],'linear-nullspace-implicit-bias':[1,1,1,0,0,.25,16],'minimum-norm-interpolation-risk':[4,.2,7,4],'calibration-brier-bin-uncertainty':[.15,.5,.85,1,20,7,0],'group-shift-total-variation-dro':[.9,.1,.8,16,7,1],'binomial-sample-complexity':[32,.5,.2,.05],'finite-window-learning-scaling':[7,20,1,0,7,0]};
for(const kind of theoryKinds)test(`learning-theory result and invalid-parameter contract: ${kind}`,()=>{const p=initial[kind],r=evaluateTheory(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const value of Object.values(r.stats).flat())assert.ok(value===null||Number.isFinite(value));assert.ok(!/[가-힣]/.test(r.statement??''));if(r.weights){assert.ok(r.weights.path.flat().every(Number.isFinite));assert.ok(r.weights.constraint.flat().every(Number.isFinite));}assert.throws(()=>evaluateTheory(kind,[]));const bad=p.slice();bad[0]=NaN;assert.throws(()=>evaluateTheory(kind,bad));});

for(const id of theoryKinds)test(`theory guides, presets and parameter boundaries: ${id}`,()=>{
 const spec=theoryLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);const vectors=[spec.controls.map(c=>c.initial),...spec.presets.map(p=>p.values)];for(let i=0;i<spec.controls.length;i++)for(const v of [spec.controls[i].min,spec.controls[i].max]){const p=spec.controls.map(c=>c.initial);p[i]=v;vectors.push(p);}for(const p of vectors){const r=evaluateTheory(id,p);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const v of Object.values(r.stats).flat())assert.ok(v===null||Number.isFinite(v));assert.ok(!/[가-힣]/.test(r.statement??''));if(r.labels){assert.equal(r.labels.length,p[0]+1);assert.ok(r.labels.every(row=>row.length===p[0]&&row.every(v=>v===1||v===-1)));}if(r.weights)assert.ok([...r.weights.path,...r.weights.constraint,r.weights.minimum].flat().every(Number.isFinite));if(r.bounds){const [lx,hx,ly,hy]=r.bounds;assert.ok(hx>lx&&hy>ly);assert.ok(r.curves.flat().every(([x,y])=>x>=lx&&x<=hx&&y>=ly&&y<=hy));}}const index=spec.controls.findIndex(c=>c.step===1);if(index>=0){const p=spec.controls.map(c=>c.initial);p[index]=spec.controls[index].min+.5;assert.throws(()=>evaluateTheory(id,p));}
});
for(const id of theoryKinds)for(const lang of ['ko','en'] as const)test(`substantive theory manuscript and rendered equations: ${id}.${lang}`,async()=>{
 const raw=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),content=await getLearningContent(id);assert.ok((raw.match(/^## /gm)||[]).length>=5);assert.ok(raw.includes('$$'));assert.ok(raw.includes('https://'));assert.ok(raw.includes(lang==='en'?'Exercise:':'직접 풀기:'));if(lang==='en'){assert.ok(raw.split(/\s+/).length>350);assert.ok(!/[가-힣]/.test(raw));}else assert.ok((raw.match(/[가-힣]/g)||[]).length>700);assert.ok(content[lang].includes('katex-display'));assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].length>4000);
});

test('zero-feature baseline predicts zero and reveals a computed interpolation peak without prescribing a curve',()=>{const r=interpolationRisk(4,.5,7,0);assert.equal(r.current.p,0);assert.deepEqual(r.current.predictions,[0,0,0,0]);close(r.current.training,sum(r.y.map(v=>v*v))/4);close(r.current.population,1.25);assert.equal(r.current.rank,0);assert.equal(r.current.condition,null);assert.ok(r.levels[0].population>r.levels[1].population);assert.ok(r.levels[4].population>r.levels[3].population);assert.ok(r.levels[4].population>r.levels[5].population);});
