import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateBayes as evaluate, bayesKinds, betaCountPredictive, logBetaInteger, mixtureEm } from '../src/lib/learning-bayes';
import { bayesLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-8)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const integrate=(f:(x:number)=>number,lo:number,hi:number,N=6000)=>{const h=(hi-lo)/N;let sum=f(lo)+f(hi);for(let i=1;i<N;i++)sum+=(i%2?4:2)*f(lo+i*h);return sum*h/3;};
const number=(r:ReturnType<typeof evaluate>,key:string)=>{const v=r.stats[key];assert.equal(typeof v,'number');return v as number;};
const factorial=(n:number)=>Array.from({length:n},(_,i)=>i+1).reduce((a,b)=>a*b,1),choose=(n:number,k:number)=>factorial(n)/(factorial(k)*factorial(n-k));
test('finite Bayesian update normalizes direct joint masses and averages predictions',()=>{
 for(let k=0;k<=4;k++)for(let mode=0;mode<=2;mode++){const r=evaluate('finite-bayes-updates',[k,mode]),prior=mode===0?[1/3,1/3,1/3]:mode===1?[.8,.1,.1]:[.5,.5,0],theta=[.2,.5,.8],joint=theta.map((t,i)=>prior[i]*t**k*(1-t)**(4-k)),Z=joint.reduce((s,v)=>s+v,0),posterior=r.stats['posterior masses'] as number[];close(number(r,'ordered-data evidence'),Z);close(posterior.reduce((s,v)=>s+v,0),1);posterior.forEach((v,i)=>close(v,joint[i]/Z));close(number(r,'posterior predictive P(next=1)'),posterior.reduce((s,v,i)=>s+v*theta[i],0));if(mode===2)assert.equal(posterior[2],0);}
 close(number(evaluate('finite-bayes-updates',[3,0]),'posterior predictive P(next=1)'),.6681260945709282);
});
test('Beta-binomial predictive masses agree with independently integrated conditional binomials',()=>{
 for(const [k,a,b] of [[4,1,1],[8,1,1],[0,2,4],[3,8,8]]){const A=a+k,B=b+8-k,f=(p:number)=>p**(A-1)*(1-p)**(B-1),norm=integrate(f,0,1),law=betaCountPredictive(4,A,B);for(let j=0;j<=4;j++)close(law[j],integrate(p=>choose(4,j)*p**j*(1-p)**(4-j)*f(p),0,1)/norm);close(law.reduce((s,v)=>s+v,0),1);const r=evaluate('conjugate-posterior-predictive',[k,a,b]),mean=law.reduce((s,v,j)=>s+j*v,0),variance=law.reduce((s,v,j)=>s+v*(j-mean)**2,0);close(number(r,'future count mean'),mean);close(number(r,'future count variance'),variance);assert.ok(variance>number(r,'plug-in count variance'));}
 assert.deepEqual(betaCountPredictive(0,1,1),[1]);
});
test('integrating a shared Beta parameter creates the displayed future-pair covariance',()=>{
 const f=(p:number)=>p**4*(1-p)**4,norm=integrate(f,0,1),mean=integrate(p=>p*f(p),0,1)/norm,cov=integrate(p=>p*p*f(p),0,1)/norm-mean*mean;close(number(evaluate('conjugate-posterior-predictive',[4,1,1]),'future-pair covariance'),cov);
});
test('partial pooling agrees with direct integration of normal prior times likelihood',()=>{
 for(const [n,y,tau] of [[4,2,1],[4,2,.25],[20,-1,.1]]){const f=(x:number)=>Math.exp(-x*x/(2*tau*tau)-n*(y-x)**2/2),norm=integrate(f,-15,15),mean=integrate(x=>x*f(x),-15,15)/norm,variance=integrate(x=>(x-mean)**2*f(x),-15,15)/norm,r=evaluate('hierarchical-partial-pooling',[n,y,-1,tau]);close((r.stats['posterior means'] as number[])[0],mean);close(number(r,'conditional posterior variance'),variance);assert.equal(r.stats['conditional posterior covariance'],0);assert.equal(r.stats['new-group prior variance'],tau*tau);}
});
test('ordered-data evidence equals integration and every ordering has the same final product',()=>{
 for(const [a,b] of [[1,1],[2,3],[8,8]]){const prior=(p:number)=>p**(a-1)*(1-p)**(b-1),norm=integrate(prior,0,1);let total=0;for(let k=0;k<=8;k++){const r=evaluate('marginal-likelihood-evidence',[k,a,b]),Z=integrate(p=>p**k*(1-p)**(8-k)*prior(p),0,1)/norm;close(number(r,'ordered-data evidence'),Z);close(number(r,'log ordered-data evidence'),number(r,'sum log sequential predictions'));assert.ok(number(r,'log likelihood minus log evidence')>=-1e-12);total+=choose(8,k)*Z;close(r.curves[1].at(-1)![1],number(r,'max log likelihood'));}close(total,1);
 for(let mask=0;mask<256;mask++){let A=a,B=b,product=1,k=0;for(let i=0;i<8;i++){const bit=(mask>>i)&1;product*=bit?A/(A+B):B/(A+B);if(bit){A++;k++;}else B++;}close(product,number(evaluate('marginal-likelihood-evidence',[k,a,b]),'ordered-data evidence'));}}
});
test('exact mean-only EM has monotone observed likelihood and normalized responsibilities',()=>{
 for(const separation of [0,.01,.5,1,3]){const r=mixtureEm(separation,12);for(let i=1;i<r.history.length;i++)assert.ok(r.history[i]>=r.history[i-1]-1e-12);r.responsibilities.forEach(v=>assert.ok(v>=0&&v<=1));const sum=r.responsibilities.reduce((s,v)=>s+v,0),updated=r.responsibilities.reduce((s,v,i)=>s+v*r.data[i],0)/sum;close(updated,r.means[0],2e-6);}
 const start=mixtureEm(1,0),r=mixtureEm(1,1),weights=start.data.map(x=>{const a=Math.exp(-((x+1)**2)/2),b=Math.exp(-((x-1)**2)/2);return a/(a+b);});close(r.means[0],weights.reduce((s,v,i)=>s+v*start.data[i],0)/weights.reduce((s,v)=>s+v,0));
});
test('symmetric EM initialization remains stationary below a separated solution',()=>{
 const stuck=mixtureEm(0,8),separated=mixtureEm(1,8);assert.deepEqual(stuck.means,[0,0]);assert.ok(separated.history.at(-1)!>stuck.history.at(-1)!);assert.ok(stuck.responsibilities.every(v=>v===.5));
});
test('ELBO and directed KL agree with direct joint and entropy sums including zero-q endpoints',()=>{
 for(const r of [0,.2,.5,.7,1])for(const pi of [.1,.7,.9]){const out=evaluate('variational-elbo',[r,pi]),q=[1-r,r],joint=[.4*(1-pi),.4*pi],elbo=q.reduce((s,v,i)=>s+(v===0?0:v*Math.log(joint[i]/v)),0);close(number(out,'ELBO'),elbo);close(number(out,'ELBO')+number(out,'KL(q || posterior)'),Math.log(.4));assert.ok(number(out,'KL(q || posterior)')>=-1e-12);}
 assert.equal(evaluate('variational-elbo',[.7,.7]).stats['KL(q || posterior)'],0);close(number(evaluate('variational-elbo',[1,.7]),'KL(q || posterior)'),Math.log(1/.7));
});
test('ordinary importance identity holds while a finite self-normalized ratio is biased',()=>{
 for(const a of [.1,1/3,.8]){const q=[a,(1-a)/2,(1-a)/2],p=[.2,.3,.5];close(q.reduce((s,v,i)=>s+v*(p[i]/v)*i,0),1.3);let expectation=0;for(let i=0;i<3;i++)for(let j=0;j<3;j++)expectation+=q[i]*q[j]*((p[i]/q[i]*i+p[j]/q[j]*j)/(p[i]/q[i]+p[j]/q[j]));assert.ok(Math.abs(expectation-1.3)>.001);}
});
test('importance ESS can be maximal with wrong estimates and support failure is withheld',()=>{
 const r=evaluate('importance-sampling',[1/3,2]);close(number(r,'weight ESS out of six'),6);close(number(r,'self-normalized estimate'),2);
 for(const q of [0,1])for(const pattern of [0,1,2]){const out=evaluate('importance-sampling',[q,pattern]);assert.equal(out.stats['1{proposal covers target}'],0);assert.equal(out.stats['self-normalized estimate'],null);assert.equal(out.stats['weight ESS out of six'],null);}
 assert.equal(evaluate('importance-sampling',[0,1]).stats['1{displayed draw possible}'],1);assert.equal(evaluate('importance-sampling',[0,0]).stats['1{displayed draw possible}'],0);
});
test('extreme positive proposal probabilities retain stable ratios and distinguish numeric overflow from missing support',()=>{
 for(const q of [1e-200,1e-320]){const r=evaluate('importance-sampling',[q,0]);assert.equal(r.stats['1{proposal covers target}'],1);assert.ok(Number.isFinite(number(r,'self-normalized estimate')));close(number(r,'weight ESS out of six'),2);}
 const r=evaluate('importance-sampling',[1e-320,1]);close(number(r,'self-normalized estimate'),1.625);assert.equal(r.warning,'numeric-range');
});
test('Laplace curvature matches finite differences and exact Beta moments match quadrature',()=>{
 for(const [k,a,b] of [[4,2,2],[0,2,2],[8,2,6],[4,6,6]]){const r=evaluate('laplace-posterior-approximation',[k,a,b]),A=k+a,B=8-k+b,mode=number(r,'posterior mode'),log=(p:number)=>(A-1)*Math.log(p)+(B-1)*Math.log1p(-p),h=1e-5,H=-(log(mode+h)-2*log(mode)+log(mode-h))/(h*h),f=(p:number)=>p**(A-1)*(1-p)**(B-1),norm=integrate(f,0,1),mean=integrate(p=>p*f(p),0,1)/norm,variance=integrate(p=>(p-mean)**2*f(p),0,1)/norm;close(number(r,'negative log curvature'),H,3e-4);close(number(r,'posterior mean'),mean);close(number(r,'exact posterior variance'),variance);close(number(r,'Laplace variance')*number(r,'negative log curvature'),1);assert.ok(r.curves.flat().every(v=>v.every(Number.isFinite)));}
});
test('untruncated Laplace support leakage equals an independent Gaussian tail integral',()=>{
 const r=evaluate('laplace-posterior-approximation',[0,2,2]),mean=number(r,'posterior mode'),sd=Math.sqrt(number(r,'Laplace variance')),f=(p:number)=>Math.exp(-.5*((p-mean)/sd)**2)/(sd*Math.sqrt(2*Math.PI)),leak=integrate(f,-3,0)+integrate(f,1,4);close(number(r,'Gaussian mass outside [0,1] (approx.)'),leak,1e-6);assert.ok(leak>.1);
});
test('Bayes factors agree with integrated alternative and posterior odds include model priors',()=>{
 for(const k of [0,4,8])for(const prior of [.1,.5,.9]){const r=evaluate('bayesian-model-comparison',[k,prior,1]),Z=integrate(p=>p**k*(1-p)**(8-k),0,1),BF=Z/(.5**8),post=prior*Z/(prior*Z+(1-prior)*.5**8);close(number(r,'Bayes factor M1/M0'),BF);close(number(r,'posterior P(M1)'),post);close(number(r,'posterior model odds'),BF*prior/(1-prior));close((choose(8,k)*Z)/(choose(8,k)*.5**8),BF);}
 close(number(evaluate('bayesian-model-comparison',[8,.5,1]),'Bayes factor M1/M0'),256/9);
});
test('uncertainty decomposition and calibration scores match independent moment integration and all fifty labels',()=>{
 for(const [a,b] of [[2,2],[12,12],[1,8]]){const r=evaluate('predictive-uncertainty-calibration',[a,b,0]),f=(p:number)=>p**(a-1)*(1-p)**(b-1),norm=integrate(f,0,1),mean=integrate(p=>p*f(p),0,1)/norm,aleatoric=integrate(p=>p*(1-p)*f(p),0,1)/norm,epistemic=integrate(p=>(p-mean)**2*f(p),0,1)/norm;close(number(r,'expected conditional variance'),aleatoric);close(number(r,'variance of conditional mean'),epistemic);close(number(r,'total predictive variance'),aleatoric+epistemic);}
 for(const mode of [0,1]){const r=evaluate('predictive-uncertainty-calibration',[12,12,mode]);let loss=0,ece=0;for(const row of r.table.rows){const p=Number(row[0]),positives=Number(row[1]);for(let i=0;i<10;i++)loss+=(p-Number(i<positives))**2;ece+=Math.abs(p-positives/10)/5;}close(number(r,'Brier score (fifty labels)'),loss/50);close(number(r,'empirical calibration error (five bins)'),ece);}
 close(number(evaluate('predictive-uncertainty-calibration',[12,12,1]),'empirical calibration error (five bins)'),.112);
});
test('Bayesian helpers and controls reject nonfinite, fractional and out-of-domain inputs',()=>{
 for(const f of [()=>logBetaInteger(0,2),()=>logBetaInteger(2.5,2),()=>logBetaInteger(100,101),()=>betaCountPredictive(2.5,2,2),()=>mixtureEm(-1,2),()=>mixtureEm(1,1.5)])assert.throws(f);
 for(const kind of bayesKinds){const values=bayesLabs[kind].controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[NaN,...values.slice(1)]));assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[Infinity,...values.slice(1)]));}
});
for(const kind of bayesKinds)test(`${kind} has four concept steps and three finite guided examples`,()=>{
 const spec=bayesLabs[kind],guide=learningGuides[kind];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat().every(v=>v.every(Number.isFinite)));for(const value of Object.values(r.stats)){if(typeof value==='number')assert.ok(Number.isFinite(value));if(Array.isArray(value))assert.ok(value.every(Number.isFinite));}assert.ok(!/[가-힣]/.test(r.statement??''));}
});
