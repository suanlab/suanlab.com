import test from 'node:test';
import assert from 'node:assert/strict';
import {instrumentalMoments,mediationEffects,gaussianDiscovery,threeNodeDAGs,mechanismTransport,offPolicyConfounding,evaluateCausalResearch} from '../src/lib/learning-causal-research';
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0),mean=(a:number[])=>sum(a)/a.length;
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const bits=(n:number)=>Array.from({length:2**n},(_,mask)=>Array.from({length:n},(_,j)=>((mask>>j)&1)?1:-1));
test('IV and OLS moments match independent sixteen-state centered-noise integration',()=>{
 for(const gamma of [0,.1,1])for(const alpha of [0,1])for(const beta of [-1,1])for(const delta of [0,1])for(const kappa of [-.5,0,.5])for(const rho of [-.5,0,.5])for(const sigma of [0,1]){
  const r=instrumentalMoments(gamma,alpha,beta,delta,kappa,rho,sigma),rows=bits(4).map(([h,v,ex,ey])=>{const z=h,u=rho*h+Math.sqrt(1-rho*rho)*v,x=gamma*z+alpha*u+sigma*ex,y=beta*x+delta*u+kappa*z+ey;return {z,u,x,y};});
  const zx=mean(rows.map(v=>v.z*v.x)),zy=mean(rows.map(v=>v.z*v.y)),xx=mean(rows.map(v=>v.x*v.x)),xy=mean(rows.map(v=>v.x*v.y));close(r.covZX,zx);close(r.covZY,zy);close(r.varX,xx);close(r.covXY,xy);
  if(r.covZX===0)assert.equal(r.iv,null);else close(r.iv!,zy/zx);if(r.varX===0)assert.equal(r.ols,null);else close(r.ols!,xy/xx);
 }
});
test('IV validity, weak-denominator amplification and independence failure remain distinct',()=>{
 const valid=instrumentalMoments(1,1,1,1,0,0,1);close(valid.iv!,1);close(valid.ols!,4/3);
 close(instrumentalMoments(.1,1,1,1,.1,0,1).iv!,2);close(instrumentalMoments(1,1,1,1,0,.5,1).iv!,4/3);
 assert.equal(instrumentalMoments(0,1,1,1,0,0,1).iv,null);assert.equal(instrumentalMoments(0,0,1,1,0,0,0).ols,null);
});
test('weak IV plots omit out-of-window values and keep nonfinite machine ratios unavailable',()=>{
 const r=evaluateCausalResearch('instrumental-moment-assumptions',[.01,1,1,1,.5,0,1]);assert.ok(Number(r.stats['plot IV points omitted'])>0);assert.ok(r.curves.flat().every(([,y])=>y>=-5.05&&y<=5.05));close(Number(r.stats['population IV ratio']),51);
 const extreme=evaluateCausalResearch('instrumental-moment-assumptions',[Number.MIN_VALUE,1,1,1,.5,0,1]);assert.equal(extreme.stats['population IV ratio'],null);assert.ok(extreme.curves.flat(2).every(Number.isFinite));
});
test('mediation effects equal independent analytic formulas and statewise telescoping identities',()=>{
 for(const a of [-1,.5,1])for(const b of [-1,.5,1])for(const c of [-1,.2,1])for(const d of [-1,0,.5,1])for(const m of [-2,1,2]){
  const r=mediationEffects(a,b,c,d,m);close(r.TE,c+a*b+a*d);close(r.PNDE,c);close(r.TNIE,a*(b+d));close(r.PNIE,a*b);close(r.TNDE,c+a*d);close(r.CDE,c+d*m);close(r.TE,r.PNDE+r.TNIE);close(r.TE,r.PNIE+r.TNDE);
  for(const v of r.atoms){close(v.Y11-v.Y00,(v.Y10-v.Y00)+(v.Y11-v.Y10));close(v.Y11-v.Y00,(v.Y01-v.Y00)+(v.Y11-v.Y01));}
 }
});
test('mediator interaction separates reference indirect effects and controlled versus natural direct effects',()=>{
 const r=mediationEffects(1,.5,.2,.5,-1);close(r.TE,1.2);close(r.PNDE,.2);close(r.TNIE,1);close(r.PNIE,.5);close(r.TNDE,.7);close(r.CDE,-.3);
 const simple=mediationEffects(1,.5,.2,0,-1);close(simple.TNIE,simple.PNIE);close(simple.PNDE,simple.TNDE);close(simple.CDE,simple.PNDE);
});
test('Gaussian covariance and partial correlations match separate structural state integration and residualization',()=>{
 for(let mode=0;mode<4;mode++)for(const a of [.25,1])for(const b of [.25,1])for(const c of [-1,0,1]){
  const r=gaussianDiscovery(mode,a,b,c,1),rows=bits(3).map(([ea,eb,ec])=>mode===0?[ea,a*ea+eb,b*(a*ea+eb)+ec]:mode===1?[a*eb+ea,eb,b*eb+ec]:mode===2?[ea,a*ea+b*ec+eb,ec]:[ea,a*ea+eb,c*ea+b*(a*ea+eb)+ec]);
  const cov=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>mean(rows.map(v=>v[i]*v[j]))));for(let i=0;i<3;i++)for(let j=0;j<3;j++)close(r.covariance[i][j],cov[i][j]);
  const ra=rows.map(v=>v[0]-cov[0][1]/cov[1][1]*v[1]),rc=rows.map(v=>v[2]-cov[2][1]/cov[1][1]*v[1]);close(r.partial,mean(ra.map((v,i)=>v*rc[i]))/Math.sqrt(mean(ra.map(v=>v*v))*mean(rc.map(v=>v*v))));
 }
});
test('all 25 three-node DAG candidates distinguish Markov and faithful compatibility',()=>{
 assert.equal(threeNodeDAGs.length,25);assert.equal(new Set(threeNodeDAGs.map(e=>e.map(v=>v.join(',')).sort().join(';'))).size,25);
 for(const mode of [0,1]){const r=gaussianDiscovery(mode,1,1,0,1);assert.equal(r.candidates.filter(v=>v.faithful).length,3);assert.equal(r.candidates.filter(v=>v.markov).length,9);assert.equal(r.trueFaithful,true);close(r.partial,0);}
 const collider=gaussianDiscovery(2,1,1,0,1);assert.equal(collider.candidates.filter(v=>v.faithful).length,1);assert.equal(collider.candidates.filter(v=>v.markov).length,7);close(collider.marginal,0);close(collider.partial,-.5);
 const complete=gaussianDiscovery(3,.5,.5,.5,0);assert.equal(complete.candidates.filter(v=>v.faithful).length,6);assert.equal(complete.trueFaithful,true);
});
test('unfaithful triangle cancellation mimics a collider CI pattern without changing its generating graph',()=>{
 const r=gaussianDiscovery(3,1,1,-1,0);assert.equal(r.edges.length,3);assert.equal(r.trueFaithful,false);close(r.marginal,0);close(r.partial,-1/Math.sqrt(3));assert.equal(r.candidates.filter(v=>v.faithful).length,1);assert.equal(r.candidates.filter(v=>v.markov).length,7);
});
test('transport uses observed source cell ratios only on supported target-positive strata',()=>{
 for(const source of [0,.2,1])for(const target of [0,.8,1])for(const e0 of [0,.5,1])for(const e1 of [0,.5,1])for(const shift of [-.1,0,.1]){
  const r=mechanismTransport(source,target,e0,e1,shift);close(sum(r.cells.map(v=>v.mass)),1);close(r.sourceATE,.1+.2*source);close(r.targetATE,.1+.2*target+shift);
  const supported=[0,1].every(z=>r.Q[z]===0||(r.P[z]>0&&r.e[z]>0&&r.e[z]<1));assert.equal(r.transported!==null,supported);
  if(supported){close(r.transported!,.1+.2*target);close(r.transported!-r.targetATE,-shift);}for(let z=0;z<2;z++)for(let x=0;x<2;x++){const mass=r.P[z]*(x?r.e[z]:1-r.e[z]);if(mass===0)assert.equal(r.rates[z][x],null);else close(r.rates[z][x]!,r.q(z,x));}
 }
});
test('transport distinguishes composition shift, outcome-mechanism shift and absent source groups',()=>{
 const r=mechanismTransport(.2,.8,.5,.5,0);close(r.sourceATE,.14);close(r.targetATE,.26);close(r.transported!,.26);assert.equal(r.invariant,true);
 const changed=mechanismTransport(.2,.8,.5,.5,.1);close(changed.targetATE,.36);close(changed.transported!,.26);assert.equal(changed.invariant,false);
 assert.equal(mechanismTransport(0,1,.5,.5,0).transported,null);close(mechanismTransport(0,0,.5,0,0).transported!,.1);
});
test('off-policy means agree with independently enumerated weighted binary outcomes when support holds',()=>{
 for(const pi of [0,.5,1])for(const e0 of [0,.1,.5,1])for(const e1 of [0,.5,.9,1])for(const policy of [0,.3,1]){
  const r=offPolicyConfounding(pi,e0,e1,policy);let truth=0,naive=0,conditional=0,logging=0;
  for(let u=0;u<2;u++)for(let a=0;a<2;a++)for(let y=0;y<2;y++){const pu=u?pi:1-pi,e=u?e1:e0,pa=a?e:1-e,pt=a?policy:1-policy,q=.1+.6*u+.2*a,py=y?q:1-q,mass=pu*pa*py;truth+=pu*pt*py*y;logging+=mass*y;if(pt>0&&r.marginals[a]>0)naive+=mass*pt/r.marginals[a]*y;if(pt>0&&pa>0)conditional+=mass*pt/pa*y;}
  close(r.truth,truth);close(r.logging,logging);if(r.marginalSupport)close(r.naive!,naive);else assert.equal(r.naive,null);if(r.conditionalSupport)close(r.conditional!,conditional);else assert.equal(r.conditional,null);
 }
});
test('hidden-confounding bias, randomization and marginal-without-conditional overlap are explicit',()=>{
 const r=offPolicyConfounding(.5,.1,.9,1);close(r.truth,.6);close(r.naive!,.84);close(r.conditional!,.6);
 const randomized=offPolicyConfounding(.5,.5,.5,1);close(randomized.naive!,randomized.truth);
 const missing=offPolicyConfounding(.5,0,1,1);assert.equal(missing.marginalSupport,true);assert.equal(missing.conditionalSupport,false);close(missing.naive!,.9);assert.equal(missing.conditional,null);
 const unused=offPolicyConfounding(.5,0,0,0);assert.equal(unused.marginalSupport,true);assert.equal(unused.conditionalSupport,true);close(unused.naive!,.4);close(unused.conditional!,.4);assert.equal(offPolicyConfounding(.5,0,0,1).naive,null);
});
