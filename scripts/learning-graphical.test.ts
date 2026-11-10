import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateGraphical as evaluate, graphicalKinds, multiplyFactors, sumOut, eliminateBinary, binaryBeliefPropagation, isingJoint, hmmForwardBackward } from '../src/lib/learning-graphical';
import { graphicalLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`),num=(r:ReturnType<typeof evaluate>,key:string)=>r.stats[key] as number;
const bits=(m:number,n:number)=>Array.from({length:n},(_,i)=>(m>>i)&1);
test('BN normalization and scaled MRF agree for every joint assignment and boundary parameter',()=>{
 for(const p of [0,.3,.5,1])for(const r of [0,.5,.8,1]){const bn=evaluate('bayesian-markov-factorization',[0,p,r,2]);close(num(bn,'raw normalizer'),1);for(const scale of [1,2,4]){const mrf=evaluate('bayesian-markov-factorization',[1,p,r,scale]);close(num(mrf,'raw normalizer'),scale*scale);mrf.table.rows.forEach((row,i)=>close(Number(row[2]),Number(bn.table.rows[i][2])));close(num(mrf,'P(C=1)'),p*(r*r+(1-r)**2)+(1-p)*2*r*(1-r));}}
});
test('factor graph product table restores probability from three independently multiplied functions',()=>{
 for(const [h,J] of [[.4,.7],[0,0],[-2,2]]){const out=evaluate('factor-graphs-products',[h,J,7]);let Z=0;for(const row of out.table.rows){const x=String(row[0]).split('').map(Number),s=x.map(v=>2*v-1),w=Math.exp(h*s[0]+J*s[0]*s[1]+J*s[1]*s[2]);close(Number(row[4]),w);close(Number(row[1])*Number(row[2])*Number(row[3]),w);Z+=w;}close(Z,num(out,'partition function Z'));close(out.table.rows.reduce((s,row)=>s+Number(row[5]),0),1);}
});
test('collider conditioning opens dependence while chain and fork conditioning block it',()=>{
 for(const shape of [0,1]){const unconditional=evaluate('d-separation-colliders',[shape,0,.1]);assert.ok(num(unconditional,'I(A;C | selected event) (bits)')>0);for(const condition of [1,2]){const r=evaluate('d-separation-colliders',[shape,condition,.1]);assert.equal(r.stats['1{graph d-separates A,C}'],1);assert.equal(r.stats['I(A;C | selected event) (bits)'],0);r.table.rows.forEach(row=>close(Number(row[2]),Number(row[3])));}}
 const closed=evaluate('d-separation-colliders',[2,0,.1]),open=evaluate('d-separation-colliders',[2,2,.1]);assert.equal(closed.stats['I(A;C | selected event) (bits)'],0);assert.ok(num(open,'I(A;C | selected event) (bits)')>0);close(num(open,'conditioning event probability'),.7);[1,9,9,9].forEach((v,i)=>close(Number(open.table.rows[i][2]),v/28));
});
test('active paths do not force dependence at unfaithful or degenerate settings',()=>{
 for(const condition of [1,2]){const r=evaluate('d-separation-colliders',[2,condition,.5]);assert.equal(r.stats['1{graph d-separates A,C}'],0);assert.equal(r.stats['I(A;C | selected event) (bits)'],0);}
 const deterministic=evaluate('d-separation-colliders',[2,1,0]);assert.equal(deterministic.stats['I(A;C | selected event) (bits)'],0);assert.equal(deterministic.table.rows[0][2],1);
});
test('factor multiplication respects arbitrary scope ordering and sum-out agrees with direct sums',()=>{
 const f={scope:[1,0],values:[1,2,3,4]},g={scope:[1,2],values:[5,6,7,8]},p=multiplyFactors([f,g]);assert.deepEqual(p.scope,[0,1,2]);for(let mask=0;mask<8;mask++){const [a,b,c]=bits(mask,3);assert.equal(p.values[mask],f.values[b+2*a]*g.values[b+2*c]);}const result=sumOut(p,1);assert.deepEqual(result.scope,[0,2]);for(let a=0;a<2;a++)for(let c=0;c<2;c++)assert.equal(result.values[a+2*c],p.values[a+4*c]+p.values[a+2+4*c]);assert.deepEqual(multiplyFactors([]),{scope:[],values:[1]});
});
test('elimination orders return the same exact posterior with different measured peak scopes',()=>{
 for(const r of [0,.2,.8,1])for(const evidence of [0,1,2]){const end=evaluate('variable-elimination-orders',[0,r,evidence]),middle=evaluate('variable-elimination-orders',[1,r,evidence]);close(num(end,'P(A=1 | evidence)'),num(middle,'P(A=1 | evidence)'));assert.equal(end.stats['peak multiplied factor entries'],4);assert.equal(middle.stats['peak multiplied factor entries'],8);let Z=0,one=0;for(let mask=0;mask<16;mask++){const x=bits(mask,4);if(evidence!==0&&x[3]!==evidence-1)continue;const w=.5*x.slice(1).reduce((s,v,i)=>s*(v===x[i]?r:1-r),1);Z+=w;one+=w*x[0];}close(num(end,'evidence probability'),Z);close(num(end,'P(A=1 | evidence)'),one/Z);}
});
test('junction-tree separator calibration matches independent four-cycle enumeration',()=>{
 for(const J of [-1,.7])for(const h of [0,.4])for(const evidence of [0,1,2]){const out=evaluate('junction-tree-calibration',[J,h,evidence]),weights=Array(4).fill(0) as number[];let Z=0;for(let mask=0;mask<16;mask++){const x=bits(mask,4),s=x.map(v=>2*v-1);if(evidence!==0&&x[2]!==evidence-1)continue;const w=Math.exp(h*s[0]+J*(s[0]*s[1]+s[1]*s[2]+s[2]*s[3]+s[3]*s[0]));Z+=w;weights[x[0]+2*x[2]]+=w;}close(num(out,'partition/evidence normalizer'),Z,1e-8);out.table.rows.forEach((row,i)=>{close(Number(row[3]),weights[i]/Z);close(Number(row[4]),weights[i]/Z);});assert.ok(num(out,'maximum numerical separator mismatch')<1e-12);}
});
test('tree messages become exact after the diameter and all outgoing messages normalize',()=>{
 for(const J of [-1.5,0,1])for(const h of [-.7,0,.7]){const bp=binaryBeliefPropagation(4,[[0,1],[1,2],[2,3]],J,h,3);assert.ok(bp.history.at(-1)!.error<1e-12);for(const m of bp.messages.values())close(m[0]+m[1],1);}
 const early=binaryBeliefPropagation(4,[[0,1],[1,2],[2,3]],1,.7,1);assert.ok(early.history.at(-1)!.error>.1);assert.equal(early.beliefs[3],.5);
});
test('HMM forward/backward agrees with every latent path and every four-bit observation',()=>{
 for(const stay of [0,.8,1])for(const correct of [0,.5,.9,1])for(let observed=0;observed<16;observed++){const obs=bits(observed,4),r=hmmForwardBackward(stay,correct,observed);let Z=0;const one=[0,0,0,0],prefixZ=[0,0,0,0],prefixOne=[0,0,0,0];for(let t=0;t<4;t++)for(let mask=0;mask<2**(t+1);mask++){const x=bits(mask,t+1);let w=.5;for(let i=0;i<=t;i++){if(i>0)w*=x[i]===x[i-1]?stay:1-stay;w*=x[i]===obs[i]?correct:1-correct;}prefixZ[t]+=w;prefixOne[t]+=w*x[t];if(t===3){Z+=w;for(let i=0;i<4;i++)one[i]+=w*x[i];}}close(r.evidence,Z);assert.equal(r.possible,Z>0);for(let t=0;t<4;t++){if(Z===0)assert.equal(r.smoothed[t],null);else close(r.smoothed[t]!,one[t]/Z);if(prefixZ[t]===0)assert.equal(r.filtered[t],null);else close(r.filtered[t]!,prefixOne[t]/prefixZ[t]);}}
});
test('positive HMM evidence below numeric range remains possible in log arithmetic',()=>{
 const r=hmmForwardBackward(1,1e-320,5);assert.equal(r.evidence,0);assert.equal(r.possible,true);assert.ok(Number.isFinite(r.logZ));close(r.logZ,2*Math.log(1e-320),1e-10);r.smoothed.forEach(v=>close(v!,.5));const out=evaluate('hmm-forward-backward',[1,1e-320,5]);assert.equal(out.warning,'numeric-range');assert.equal(out.stats['1{evidence possible}'],1);const impossible=hmmForwardBackward(1,1,5);assert.equal(impossible.possible,false);assert.equal(impossible.logZ,-Infinity);
});
test('scalar Gaussian state-space worked update agrees with density integration',()=>{
 const N=12000,lo=-10,hi=10,step=(hi-lo)/N;let Z=0,first=0,second=0;for(let i=0;i<=N;i++){const x=lo+i*step,w=Math.exp(-x*x/2-(1-x)**2/2)*(i===0||i===N?1:i%2?4:2);Z+=w;first+=x*w;second+=x*x*w;}close(first/Z,.5);close(second/Z-(first/Z)**2,.5);
});
test('BIC scores use MLE counts, respond to sample size and ignore the smoothing control',()=>{
 const small=evaluate('graph-structure-parameter-learning',[1,1,1]),large=evaluate('graph-structure-parameter-learning',[1,8,1]);assert.equal(small.stats['winning model (0 independent,1 edge)'],0);assert.equal(large.stats['winning model (0 independent,1 edge)'],1);for(const pattern of [0,1,2]){const out=evaluate('graph-structure-parameter-learning',[pattern,1,1]),other=evaluate('graph-structure-parameter-learning',[pattern,1,0]);assert.equal(out.stats['BIC independent'],other.stats['BIC independent']);assert.equal(out.stats['BIC edge'],other.stats['BIC edge']);let LL0=0,LL1=0;for(const row of out.table.rows){const n=Number(row[1]);if(n>0){LL0+=n*Math.log(Number(row[3]));LL1+=n*Math.log(Number(row[2]));}}close(num(out,'BIC independent'),-2*LL0+2*Math.log(16));close(num(out,'BIC edge'),-2*LL1+3*Math.log(16));close(out.table.rows.reduce((s,row)=>s+Number(row[4]),0),1);}
});
test('Boltzmann normalization, entropy and log-partition gradient agree with direct enumeration and finite differences',()=>{
 for(const [J,h,T] of [[1,.3,1],[0,0,1],[1,0,.25],[-2,1,3]]){const out=evaluate('energy-partition-functions',[J,h,T]);let Z=0,energy=0,entropy=0;for(const row of out.table.rows){const E=Number(row[1]),prob=Number(row[3]);Z+=Math.exp(-E/T);energy+=prob*E;if(prob>0)entropy-=prob*Math.log(prob);}close(Math.log(Z),num(out,'log partition Z'));close(energy,num(out,'expected energy'));close(entropy,num(out,'entropy (nats)'));if(Math.abs(h)<.99){const eps=1e-5,plus=evaluate('energy-partition-functions',[J,h+eps,T]),minus=evaluate('energy-partition-functions',[J,h-eps,T]);close((num(plus,'log partition Z')-num(minus,'log partition Z'))/(2*eps),num(out,'expected total spin')/T,1e-7);}}
});
test('loopy residual can shrink despite nonzero marginal bias, and matching marginals need not recover correlations',()=>{
 const out=evaluate('exact-approximate-inference',[1,.7,16,1]);assert.ok(num(out,'last message-change residual')<.001);assert.ok(num(out,'maximum marginal error vs exact')>.1);const symmetric=evaluate('exact-approximate-inference',[1,0,8,1]);assert.ok(num(symmetric,'maximum marginal error vs exact')<1e-12);assert.ok(num(symmetric,'exact edge spin correlation')>.5);close(num(symmetric,'product of node spin means'),0);for(const eta of [.1,.5,1]){const r=binaryBeliefPropagation(3,[[0,1],[1,2],[2,0]],-1,.7,16,eta);for(const message of r.messages.values()){close(message[0]+message[1],1);assert.ok(message.every(v=>v>0&&v<1));}}
});
test('factor and message helpers reject invalid scopes, values and controls',()=>{
 for(const f of [()=>multiplyFactors([{scope:[0,0],values:[1,1,1,1]}]),()=>multiplyFactors([{scope:[0],values:[1,-1]}]),()=>sumOut({scope:[0],values:[1,2]},1),()=>eliminateBinary([{scope:[0],values:[1,2]}],[0,0]),()=>hmmForwardBackward(2,.5,0),()=>binaryBeliefPropagation(2,[[0,1]],1,0,2,0),()=>isingJoint(2,[[0,2]],1,0)])assert.throws(f);
 for(const kind of graphicalKinds){const values=graphicalLabs[kind].controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[NaN,...values.slice(1)]));assert.throws(()=>evaluate(kind,[]));}
});
for(const kind of graphicalKinds)test(`${kind} has substantive guides and finite graph/plot geometry`,()=>{
 const guide=learningGuides[kind],spec=graphicalLabs[kind];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat().every(v=>v.every(Number.isFinite)));assert.ok(r.graph.nodes.every(v=>Number.isFinite(v.x)&&Number.isFinite(v.y)));assert.ok(r.graph.edges.every(e=>r.graph.nodes.some(n=>n.id===e.from)&&r.graph.nodes.some(n=>n.id===e.to)));assert.ok(!/[가-힣]/.test(r.statement??''));for(const value of Object.values(r.stats))if(typeof value==='number')assert.ok(!Number.isNaN(value));}
});
