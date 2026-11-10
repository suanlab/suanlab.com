import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {klRisk,transportRisk} from '../src/lib/learning-dro';
const near=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
test('5125 finite transport settings preserve exact feasible primal/dual equality and monotone risk',()=>{
 let count=0;for(let a=0;a<=4;a++)for(let b=0;b<=4;b++)for(let c=0;c<=4;c++){let last=-Infinity;for(let code=0;code<=40;code++){
  const r=transportRisk([a,b,c],code);assert.equal(r.primalNumerator,r.dualNumerator);assert.equal(r.exactRiskEquality,true);assert.ok(r.costNumerator<=2*code);assert.deepEqual(r.planUnits.map(row=>row.reduce((a,b)=>a+b,0)),[8,12,20]);assert.ok(r.planUnits.flat().every(v=>Number.isInteger(v)&&v>=0));assert.equal(r.destinationUnits.reduce((a,b)=>a+b,0),40);assert.ok(r.primalNumerator>=last&&r.primalNumerator<=40*Math.max(a,b,c));if(code===0)assert.equal(r.primalNumerator,8*a+12*b+20*c);last=r.primalNumerator;count++;
 }}assert.equal(count,5125);
});
test('625 transport values match an independent9-variable linear-program reference',()=>{
 const fixture=JSON.parse(fs.readFileSync('scripts/fixtures/dro-transport-independent.json','utf8'));assert.equal(fixture.rows.length,625);for(const [a,b,c,code,numerator]of fixture.rows)assert.equal(transportRisk([a,b,c],code).primalNumerator,numerator);
});
test('exact transport guides show average-cost reallocation, slope changes and saturation',()=>{
 for(const [code,risk,q,lambda]of [[0,1.8,[8,12,20],2],[2,2,[8,8,24],2],[6,2.4,[8,0,32],1.5],[10,2.7,[4,0,36],1.5],[14,3,[0,0,40],0]] as const){const r=transportRisk([0,1,3],code);assert.equal(r.primalNumerator/40,risk);assert.deepEqual(r.destinationUnits,q);assert.equal(r.optimalLambda2/2,lambda);}const threshold=transportRisk([0,1,3],14);assert.equal(threshold.costNumerator,28);assert.equal(threshold.optimalLambda2,0);assert.equal(transportRisk([0,1,3],0).riskExact.label,'9/5');
});
test('transport duals upper-bound independently generated feasible couplings',()=>{
 const cost=[[0,1,2],[1,0,1],[2,1,0]],mass=[8,12,20];let state=12345;const random=(n:number)=>{state=(state*1664525+1013904223)>>>0;return state%n;};
 for(let k=0;k<500;k++){const loss=[random(11),random(11),random(11)],plan=mass.map(m=>{const a=random(m+1),b=random(m-a+1);return[a,b,m-a-b];}),used=plan.reduce((s,row,i)=>s+row.reduce((t,v,j)=>t+v*cost[i][j],0),0),code=Math.ceil(used/2),r=transportRisk(loss,code),objective=plan.reduce((s,row)=>s+row.reduce((t,v,j)=>t+v*loss[j],0),0);assert.ok(objective<=r.dualNumerator);}
});
test('constant and tied transport losses keep one optimal witness without requiring plan uniqueness',()=>{
 for(const code of [0,1,10,40]){const a=transportRisk([3,3,3],code);assert.equal(a.primalNumerator,120);assert.equal(a.optimalLambda2,0);}const tie=transportRisk([3,1,3],40);assert.equal(tie.primalNumerator,120);assert.equal(tie.destinationUnits[1],0);
});
test('875 KL settings satisfy simplex, numerical entropy budgets, primal/dual agreement and monotonicity',()=>{
 let count=0;for(let a=0;a<=4;a++)for(let b=0;b<=4;b++)for(let c=0;c<=4;c++){let last=-Infinity;for(const code of [0,1,10,100,500,1000,2000]){
  const r=klRisk([a,b,c],code);assert.ok(r.q.every(v=>v>=0&&Number.isFinite(v)));near(r.q.reduce((s,v)=>s+v,0),1);assert.ok(r.kl<=r.radius+1e-12);assert.ok(r.risk>=r.nominal-1e-12&&r.risk<=Math.max(a,b,c)+1e-12&&r.risk>=last-1e-12);near(r.rawGap,0);last=r.risk;count++;
 }}assert.equal(count,875);
});
test('30 KL optima match a separately constrained SLSQP reference',()=>{
 const fixture=JSON.parse(fs.readFileSync('scripts/fixtures/dro-kl-independent.json','utf8'));assert.equal(fixture.rows.length,30);for(const [a,b,c,code,risk]of fixture.rows)near(klRisk([a,b,c],code).risk,risk,fixture.tolerance);
});
test('KL radius0, constants and maximum-set saturation have explicit dual limit semantics',()=>{
 const zero=klRisk([0,1,3],0);assert.equal(zero.phase,'zero-radius-limit');assert.equal(zero.eta,null);assert.deepEqual(zero.q,[.2,.3,.5]);near(zero.risk,1.8);
 const constant=klRisk([2,2,2],100);assert.equal(constant.phase,'constant-loss');assert.equal(constant.eta,0);assert.equal(constant.kl,0);assert.equal(constant.risk,2);
 const top=klRisk([3,1,3],500);assert.equal(top.phase,'saturation');near(top.q[0],2/7);assert.equal(top.q[1],0);near(top.q[2],5/7);near(top.kl,-Math.log(.7));assert.equal(top.risk,3);assert.equal(top.eta,0);
});
test('near-saturation KL brackets retain feasible numerical tilt witnesses on both sides of the threshold',()=>{
 for(const loss of [[0,1,1],[1,0,1],[1,1,0],[1,0,0],[0,1,0],[0,0,1]]){const top=[.2,.3,.5].reduce((s,p,i)=>s+(loss[i]===1?p:0),0),cut=-Math.log(top)*1000,lo=klRisk(loss,Math.floor(cut)),hi=klRisk(loss,Math.ceil(cut));assert.equal(lo.phase,'finite-tilt');assert.equal(hi.phase,'saturation');assert.ok(lo.kl<=lo.radius+1e-12);assert.ok(lo.risk<1);near(hi.risk,1);near(lo.rawGap,0);}
});
test('KL translation/scaling preserves worst probabilities, while risk and numerical dual transform together',()=>{
 for(const code of [0,1,100,500,2000]){const a=klRisk([0,1,3],code),shift=klRisk([2,3,5],code),scale=klRisk([0,2,6],code);a.q.forEach((p,i)=>{near(p,shift.q[i]);near(p,scale.q[i]);});near(shift.risk,a.risk+2);near(scale.risk,2*a.risk);near(shift.dual,a.dual+2);near(scale.dual,2*a.dual);const t=transportRisk([0,1,3],Math.min(40,code)),u=transportRisk([2,3,5],Math.min(40,code));assert.equal(u.primalNumerator,t.primalNumerator+80);}
});
test('KL dual objective bounds independently sampled feasible distributions',()=>{
 const loss=[0,1,3],r=klRisk(loss,100);assert.ok(r.eta!==null&&r.eta>0);for(let a=0;a<=20;a++)for(let b=0;b<=20-a;b++){const q=[a/20,b/20,(20-a-b)/20],entropy=q.reduce((s,v,i)=>s+(v? v*Math.log(v/r.p[i]):0),0);if(entropy<=r.radius)assert.ok(q.reduce((s,v,i)=>s+v*loss[i],0)<=r.dual+1e-12);}
});
test('all controls reject unsupported values and results do not alias inputs or future calls',()=>{
 for(const fn of [klRisk,transportRisk])for(const loss of [[],[0,1],[0,1,2,3],[-1,0,1],[0,.5,1],[0,1,11],[NaN,1,2],[Infinity,1,2]])assert.throws(()=>fn(loss,0));
 for(const v of [-1,.5,NaN,Infinity,2001])assert.throws(()=>klRisk([0,1,3],v));for(const v of [-1,.5,NaN,Infinity,41])assert.throws(()=>transportRisk([0,1,3],v));
 const loss=[0,1,3],a=klRisk(loss,100);loss[0]=10;a.loss[0]=10;a.p[0]=1;a.q[0]=1;const b=klRisk([0,1,3],100);near(b.risk,2.3403757779057797);assert.equal(b.p[0],.2);
 const t=transportRisk([0,1,3],2);t.planUnits[0][0]=999;t.destinationUnits[0]=999;t.loss[0]=999;const u=transportRisk([0,1,3],2);assert.equal(u.primalNumerator,80);assert.equal(u.planUnits[0].reduce((s,v)=>s+v,0),8);
});

test('KL display adapter decodes radius units and exposes limiting versus numerical phases',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');const zero=evaluatePrivacy('kl-ambiguity-risk-envelope',[0,1,3,0]),small=evaluatePrivacy('kl-ambiguity-risk-envelope',[0,1,3,100]),sat=evaluatePrivacy('kl-ambiguity-risk-envelope',[0,1,3,1000]);assert.equal(zero.stats['1{infinite-temperature dual limit}'],1);assert.equal(zero.stats['selected eta when finite'],null);near(small.stats['numerical worst expected loss'] as number,2.3403757779057797);assert.equal(small.stats['KL ambiguity radius in nats'],.1);assert.equal(sat.stats['1{maximum-loss set reached}'],1);assert.equal(sat.stats['numerical worst expected loss'],3);assert.equal(small.table.rows.length,3);
});
test('transport display adapter certifies feasibility and exact values without a pointwise-radius claim',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');for(const [code,risk]of [[0,1.8],[2,2],[10,2.7],[14,3]]){const out=evaluatePrivacy('finite-wasserstein-risk-certificates',[0,1,3,code]);assert.equal(out.stats['exact finite worst loss'],risk);assert.equal(out.stats['1{selected plan feasible}'],1);assert.equal(out.stats['1{exact feasible primal equals dual}'],1);assert.equal(out.table.rows.length,9);assert.ok((out.stats['transport cost'] as number)<=code/20);}
});
test('DRO adapters expose finite plots within declared bounds at constant, zero, saturated and extreme settings',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');for(const id of ['kl-ambiguity-risk-envelope','finite-wasserstein-risk-certificates'] as const)for(const loss of [[0,0,0],[10,10,10],[0,10,0],[10,0,10]])for(const code of [0,1,id==='kl-ambiguity-risk-envelope'?2000:40]){const r=evaluatePrivacy(id,[...loss,code]),[xmin,xmax,ymin,ymax]=r.bounds!;assert.ok([...r.curves.flat(),...r.dots!].every(([x,y])=>Number.isFinite(x)&&Number.isFinite(y)&&x>=xmin&&x<=xmax&&y>=ymin&&y<=ymax));assert.ok(Object.values(r.stats).every(v=>v===null||typeof v==='number'&&Number.isFinite(v)));}
});
test('DRO adapters reject incomplete, extra and noninteger controller vectors',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');for(const id of ['kl-ambiguity-risk-envelope','finite-wasserstein-risk-certificates'] as const){for(const v of [[],[0,1,3],[0,1,3,0,0],[0,1,3,.5],[0,.5,3,0]])assert.throws(()=>evaluatePrivacy(id,v));}
});
