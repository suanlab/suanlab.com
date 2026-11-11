import test from 'node:test';
import assert from 'node:assert/strict';
import {causalLogBit,normalizeCausalLogs} from '../src/lib/learning-causal-probability';
import {binarySCM,binaryCounterfactual,binaryConfounding,observationPolicy} from '../src/lib/learning-causality';
import {mediationEffects,instrumentalMoments,mechanismTransport,offPolicyConfounding} from '../src/lib/learning-causal-research';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-11,`${a} != ${b}`);
test('log conditioning distinguishes unrepresentable positive evidence from structural zero',()=>{
 const r=binaryCounterfactual(1e-200,0,1e-200,0,1,0,0);assert.equal(r.evidence,0);assert.ok(r.logEvidence!==null&&r.logEvidence<-900);assert.ok(r.posterior);assert.equal(r.selected,1);assert.equal(r.means[1],0);
 const impossible=binaryCounterfactual(0,0,0,0,1,0,0);assert.equal(impossible.posterior,null);assert.equal(impossible.logEvidence,null);
 close(binarySCM(Number.MIN_VALUE,0,.5,0,-1).conditionalOne!,.5);
});
test('normalized log weights retain support, zero atoms and scale invariance',()=>{
 for(const offset of [0,-1000,-2000]){const r=normalizeCausalLogs([Math.log(.25)+offset,Math.log(.75)+offset,-Infinity]);assert.ok(r.probabilities);close(r.probabilities[0],.25);close(r.probabilities[1],.75);assert.equal(r.probabilities[2],0);}
 assert.equal(normalizeCausalLogs([-Infinity]).probabilities,null);assert.throws(()=>normalizeCausalLogs([NaN]));assert.throws(()=>causalLogBit(.5,2));assert.throws(()=>causalLogBit(-1,0));
});
test('backdoor and transport condition on positive factored cells even when their products underflow',()=>{
 const b=binaryConfounding(1e-200,.5,1e-200,-.1);assert.ok(b.adjusted.every(v=>v!==null));close(b.adjusted[1]!,.1);assert.equal(b.cells.find(v=>v.z===1&&v.x===1&&v.y===1)!.mass,0);
 const t=mechanismTransport(1e-200,1,.5,1e-200,0);close(t.transported!,.3);assert.equal(mechanismTransport(0,1,.5,1e-200,0).transported,null);assert.equal(mechanismTransport(1e-200,1,.5,0,0).transported,null);
});
test('rare marginal action support survives products below the floating-point range',()=>{
 const r=offPolicyConfounding(1e-200,0,1e-200,1);assert.equal(r.marginals[1],0);assert.equal(r.marginalSupport,true);close(r.naive!,.9);assert.equal(r.conditionalSupport,false);assert.equal(r.conditional,null);
 assert.equal(offPolicyConfounding(0,0,1e-200,1).marginalSupport,false);
});
test('analytic mediation preserves small cross-world effects while retaining rounded atom differences',()=>{
 const r=mediationEffects(1e-20,.5,1e-20,.25,0);assert.equal(r.expanded.TE,0);assert.equal(r.PNDE,1e-20);assert.ok(Math.abs(r.TNIE/7.5e-21-1)<1e-15);assert.ok(Math.abs(r.TE/1.75e-20-1)<1e-15);assert.equal(r.CDE,1e-20);
});
test('factored IV retains a valid small effect without changing zero or nonfinite denominator handling',()=>{
 const r=instrumentalMoments(1e-200,0,1e-200,0,0,0,1);assert.equal(r.expandedIV,0);assert.equal(r.iv,1e-200);assert.equal(instrumentalMoments(0,0,1,0,0,0,1).iv,null);assert.equal(instrumentalMoments(Number.MIN_VALUE,0,1,0,.5,0,1).iv,null);
});

test('rare observed-policy conditionals retain positive evidence while adjustment still enforces strata',()=>{const r=observationPolicy(1e-200,0,1e-200,1,1);assert.equal(r.queryMass,0);assert.ok(r.logQueryMass!==null&&r.logQueryMass<-900);assert.ok(r.posterior);close(r.observed!,.9);assert.equal(r.adjusted,null);assert.equal(r.supported,false);});

test('IV exact dyadic moments retain a residual after large cancellation and a nonzero underflowed denominator',()=>{const r=instrumentalMoments(1e-200,1,-1,0,-.5,-.5,1);assert.equal(r.factoredIV,0);assert.ok(Math.abs(r.iv!/2e-200-1)<1e-15);const q=instrumentalMoments(0,1e-200,.5,0,0,1e-200,1);assert.equal(q.covZX,0);assert.equal(q.iv,.5);});

test('IV rational rounding handles binary midpoint ties, subnormals and overflow',()=>{
 const half=.5,ulp=2**-53;
 assert.equal(instrumentalMoments(1,0,half,0,ulp/2,0,1).iv,half);
 assert.equal(instrumentalMoments(1,0,half+ulp,0,ulp/2,0,1).iv,half+2*ulp);
 assert.equal(instrumentalMoments(1,0,-half,0,-ulp/2,0,1).iv,-half);
 const tiny=Number.MIN_VALUE;
 assert.equal(instrumentalMoments(1,0,0,tiny,0,.5,1).iv,0);
 assert.equal(instrumentalMoments(1,0,tiny,tiny,0,.5,1).iv,2*tiny);
 assert.ok(Object.is(instrumentalMoments(1,0,0,tiny,0,-.5,1).iv,-0));
 assert.equal(instrumentalMoments(2**-1024,0,0,0,.5,0,1).iv,2**1023);
 assert.equal(instrumentalMoments(2**-1025,0,0,0,.5,0,1).iv,null);
});
