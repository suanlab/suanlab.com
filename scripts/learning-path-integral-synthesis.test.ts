import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {calculateBrownianBridge as bridge} from '../src/lib/learning-brownian-bridge';
import {calculateEuclideanKernel as euclidean} from '../src/lib/learning-euclidean-kernel';
import {calculateSaddleFluctuation as saddle} from '../src/lib/learning-saddle-fluctuation';
import {buildRealTimeChainExact as real,realTimeFractionText,buildRealTimePacketExact} from '../src/lib/learning-real-time-phase-exact';
import {Rational as Q} from '../src/lib/learning-graph-gauge-validation-math';
import {getLearningContent} from '../src/lib/learning-content';
import {euclideanKernelDefault} from '../src/lib/learning-euclidean-kernel-input';
import {saddleFluctuationDefault} from '../src/lib/learning-saddle-fluctuation-input';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
const root='.runtime/learning/path-integral-topic-audit';mkdirSync(root,{recursive:true});
const python=spawnSync('python3',['scripts/learning/verify-path-integral-synthesis.py'],{encoding:'utf8',timeout:60000,maxBuffer:2000000});assert.equal(python.status,0,python.stderr);writeFileSync(root+'/independent-fraction-references.json',python.stdout);
type CommonInput={intervals:number;time:string;mass:string;hbar:string;start:string;end:string;omegaSquared:string;probe:number};
const fixture=JSON.parse(python.stdout) as {cases:{input:CommonInput;diffusion:string;mean:string[];precision:string[][];covariance:string[][];action:string;physicalAction:string;determinant:string;prefactorSquaredTimesPi:string}[]};
const le=(a:Q,b:Q)=>a.n*b.d<=b.n*a.d;
const contains=(range:{lower:string;upper:string},exact:string)=>assert(le(Q.read(range.lower),Q.read(exact))&&le(Q.read(exact),Q.read(range.upper)),`${exact} outside ${JSON.stringify(range)}`);
test('16shared finite free models agree with an independent dense incidence/Fraction inverse',()=>{
 let cells=0;
 for(const c of fixture.cases){const b=bridge({intervals:c.input.intervals,time:c.input.time,diffusion:c.diffusion,start:c.input.start,end:c.input.end,probe:c.input.probe,source:'0'}),e=euclidean(c.input),s=saddle({mode:'chain',...c.input,quartic:'0'}),r=real({mode:'chain',...c.input,damping:'0'});assert.equal(s.kind,'chain');if(s.kind!=='chain')throw Error('wrong mode');
  assert.deepEqual(b.mean,c.mean);assert.deepEqual(e.mean,c.mean);assert.deepEqual(s.stationary.candidate,c.mean);assert.deepEqual(r.stationary!.map(realTimeFractionText),c.mean.slice(1,-1));assert.deepEqual(b.precision,c.precision);assert.deepEqual(e.precision,c.precision);assert.deepEqual(b.covariance,c.covariance);
  const values=e.covariance.values.flat();for(let i=0;i<=c.input.intervals;i++)for(let j=0;j<=c.input.intervals;j++){assert.equal(values[e.covariance.indices[i][j]],c.covariance[i][j]);contains(s.gaussian.covariance[i][j],c.covariance[i][j]);if(i>0&&i<c.input.intervals&&j>0&&j<c.input.intervals)assert.equal(Q.read(realTimeFractionText(r.inverseEntry(i-1,j-1))).mul(Q.read(c.input.hbar)).text(),c.covariance[i][j]);cells++;}
  assert.equal(b.kernel.baselineNegativeLogExponential,c.action);assert.equal(e.kernel.minimumAction,c.action);assert.equal(realTimeFractionText(r.stationaryAction!),c.physicalAction);assert.equal(s.stationary.candidateAction,c.physicalAction);assert.equal(e.factorization.determinant,c.determinant);assert.equal(b.kernel.prefactorSquaredTimesPi,c.prefactorSquaredTimesPi);assert.equal(e.kernel.prefactorSquaredTimesPi,c.prefactorSquaredTimesPi);contains(s.gaussian.prefactorSquaredTimesPi,c.prefactorSquaredTimesPi);assert.deepEqual(s.ratio,{status:'exact-quadratic',lower:'1',upper:'1'});assert.equal(e.inverseTemperature,Q.read(c.input.time).div(Q.read(c.input.hbar)).text());
 }
 writeFileSync(root+'/free-cross-checks.json',JSON.stringify({models:fixture.cases.length,covarianceEntries:cells,independentDenseFraction:true},null,2)+'\n');
});
test('same midpoint marginal does not make trapezoidal area or action expectation grid independent',()=>{
 const base={intervals:4,time:'1',diffusion:'1/2',start:'0',end:'1',probe:2,source:'0'},coarse=bridge(base),fine=bridge({...base,intervals:8,probe:4});assert.equal(coarse.covariance[2][2],'1/4');assert.equal(fine.covariance[4][4],'1/4');assert.equal(coarse.area.variance,'5/64');assert.equal(fine.area.variance,'21/256');assert.equal(Q.read(fine.area.variance).sub(Q.read(coarse.area.variance)).text(),'1/256');assert.equal(coarse.area.analyticLimitingVariance,'1/12');assert.equal(coarse.kernel.baselineNegativeLogExponential,'1/2');assert.equal(coarse.increments.expectedAction,'2');
});
test('the N2harmonic Euclidean precision andreal-action Hessian are different integrals',()=>{
 const input={...euclideanKernelDefault},e=euclidean(input),r=real({mode:'chain',...input,damping:'0'});assert.equal(e.precision[0][0],'9/2');assert.equal(e.mean[1],'4/9');assert.equal(e.covariance.values.flat()[e.covariance.indices[1][1]],'2/9');assert.equal(e.kernel.minimumAction,'49/72');assert.equal(realTimeFractionText(r.diagonal),'7/2');assert.equal(realTimeFractionText(r.stationary![0]),'4/7');assert.equal(realTimeFractionText(r.inverseEntry(0,0)),'2/7');assert.equal(realTimeFractionText(r.stationaryAction!),'17/56');assert.equal(e.endpointIntegral.conservesMass,false);assert.equal(e.trace.status,'finite');
});
test('a certified quartic stationary coordinate andlocal width do not certify actual moments or a shifted ratio',()=>{
 const s=saddle(saddleFluctuationDefault);assert.equal(s.kind,'chain');if(s.kind!=='chain')throw Error('wrong mode');assert.equal(s.stationary.candidate[1],'1/2');contains(s.gaussian.hessian[0][0],'39/8');contains(s.gaussian.covariance[1][1],'8/39');assert.equal(s.scope.stationaryMeanInterpretation,'not-a-mean-calculation');assert.equal(s.scope.gaussianCovarianceIsExactQuarticCovariance,false);assert.equal(s.ratio.status,'not-applicable-shifted-cubic');assert.equal(s.ratio.lower,null);assert.equal(s.ratio.upper,null);const q=saddle({...saddleFluctuationDefault,quartic:'0'});assert.equal(q.kind,'chain');if(q.kind==='chain')assert.equal(q.stationary.candidate[1],'37/72');
});
test('a harmonic caustic acts finitely on a packet while its chain has no ordinary inverse',()=>{
 const c=real(realTimePhaseExamples.find(x=>x.id==='caustic-compatible')!.input),p=buildRealTimePacketExact(realTimePhaseExamples.find(x=>x.id==='harmonic-parity')!.input);assert.equal(c.stationaryStatus,'affine-family');assert.equal(c.inertia.zero,1);assert.throws(()=>c.inverseEntry(0,0),/Singular/);assert.deepEqual(p.finalCovariance.map(r=>r.map(realTimeFractionText)),p.initialCovariance.map(r=>r.map(realTimeFractionText)));assert.deepEqual(p.map.total.map(r=>r.map(realTimeFractionText)),[['-1','0'],['0','-1']]);
});
test('four-chapter synthesis is bilingual,has worked comparisons andkeeps theoretical limits explicit',async()=>{
 const id='real-time-phase-and-classical-limit',content=await getLearningContent(id),raw=readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`,'utf8');assert.equal(content.outline.en.length,34);assert.equal(content.outline.ko.length,34);assert.equal((raw.match(/\*\*Worked comparison \d:/g)||[]).length,4);for(const id of['brownian-bridge-time-slicing-measure','euclidean-quantum-kernels-and-time-slicing','saddle-fluctuation-and-nongaussian-approximation'])assert(raw.includes(`/book/online/physics-math-for-ai/${id}/`));assert(!/[가-힣]/.test(content.en));assert(raw.includes('does not')||raw.includes('do not'));assert(raw.includes('37/72')&&raw.includes('1/256')&&raw.includes('39/8'));
});
