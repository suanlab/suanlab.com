import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {stateSpaceRealization,controlGramians,kalmanPositionVelocity,lyapunovMetric,controlKinds,evaluateControl} from '../src/lib/learning-control';
import {controlLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0);
const multiply=(A:number[][],B:number[][])=>A.map(row=>B[0].map((_,j)=>sum(row.map((v,k)=>v*B[k][j]))));
const transpose=(A:number[][])=>A[0].map((_,j)=>A.map(row=>row[j]));
const inv=(matrix:number[][])=>{const n=matrix.length,M=matrix.map((r,i)=>[...r,...Array.from({length:n},(_,j)=>Number(i===j))]);for(let k=0;k<n;k++){let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[pivot][k]))pivot=i;assert.ok(Math.abs(M[pivot][k])>1e-14);[M[k],M[pivot]]=[M[pivot],M[k]];const scale=M[k][k];M[k]=M[k].map(v=>v/scale);for(let i=0;i<n;i++)if(i!==k){const f=M[i][k];M[i]=M[i].map((v,j)=>v-f*M[k][j]);}}return M.map(row=>row.slice(n));};
const simpson=(f:(t:number)=>number,T:number,N=2000)=>{let value=f(0)+f(T);for(let i=1;i<N;i++)value+=(i%2?4:2)*f(T*i/N);return value*T/(3*N);};

test('state-space recurrences match independent closed-form matrix-power convolution',()=>{
 for(const s of [-2,0,2])for(const mode of [0,1,2])for(const x0 of [[0,0],[1,-1]]){
  const r=stateSpaceRealization(s,32,mode,.7,x0[0],x0[1],.3),power=(n:number)=>[[.8**n,.2*(.8**n-.5**n)/.3],[0,.5**n]];
  for(const row of r.rows){const P=power(row.t),expected=[P[0][0]*x0[0]+P[0][1]*x0[1],P[1][1]*x0[1]];for(let j=0;j<row.t;j++){const a=power(row.t-1-j);expected[0]+=a[0][1]*r.rows[j].u;expected[1]+=a[1][1]*r.rows[j].u;}row.x.forEach((v,i)=>close(v,expected[i]));close(row.y,expected[0]+.3*row.u);close(row.z[0],row.x[0]+s*row.x[1]);close(row.z[1],row.x[1]);}
  assert.ok(r.maxOutputError<1e-12&&r.maxStateError<1e-12);
 }
});
test('coordinate transforms include B,C and initial state; feedthrough uses the current input',()=>{
 const r=stateSpaceRealization(2,2,0,1,0,0,.5);assert.deepEqual(r.Bt,[2,1]);assert.deepEqual(r.Ct,[1,-2]);close(r.At[0][1],-.4);close(r.rows[0].y,.5);close(r.rows[1].y,0);close(r.rows[2].y,.2);assert.deepEqual(r.rows[1].z,[2,1]);
 const start=stateSpaceRealization(-2,1,1,0,1,1,0);assert.deepEqual(start.rows[0].z,[-1,1]);close(start.rows[0].yt,1);
});
test('finite-horizon Gramians agree with independent numerical energy-response integrals',()=>{
 for(const b of [-1,0,.1,1])for(const c of [-1,0,1])for(const T of [.25,1,3]){const r=controlGramians(b,c,T,.1,.1);for(let i=0;i<2;i++)for(let j=0;j<2;j++){const rates=[1,2],B=[1,b],C=[1,c];close(r.reach.W[i][j],simpson(t=>B[i]*B[j]*Math.exp(-(rates[i]+rates[j])*t),T));close(r.observe.W[i][j],simpson(t=>C[i]*C[j]*Math.exp(-(rates[i]+rates[j])*t),T));}const squared=multiply(r.reach.sqrt,r.reach.sqrt);squared.forEach((row,i)=>row.forEach((v,j)=>close(v,r.reach.W[i][j])));}
});
test('minimum-energy controls reach the target and attain the reported energy by separate quadrature',()=>{
 for(const b of [-1,.1,1])for(const T of [.25,1,3]){const target=[.1,.2],B=[1,b],rates=[1,2],W=B.map((v,i)=>B.map((w,j)=>simpson(t=>v*w*Math.exp(-(rates[i]+rates[j])*t),T))),weights=multiply(inv(W),target.map(v=>[v])).map(r=>r[0]),u=(t:number)=>sum(B.map((v,i)=>v*Math.exp(-rates[i]*(T-t))*weights[i])),r=controlGramians(b,1,T,...target as [number,number]);for(let i=0;i<2;i++)close(simpson(t=>Math.exp(-rates[i]*(T-t))*B[i]*u(t),T),target[i],2e-8);close(simpson(t=>u(t)**2,T),r.energy!,2e-8);}
});
test('singular and weak channels distinguish structural reachability, energy and observability',()=>{
 const missing=controlGramians(0,1,1,0,1);assert.equal(missing.reachable,false);assert.equal(missing.energy,null);assert.equal(missing.controllableRank,1);assert.equal(missing.observableRank,2);const first=controlGramians(0,0,1,1,0);close(first.energy!,2/(1-Math.exp(-2)));assert.equal(first.observableRank,1);close(controlGramians(0,0,1,0,0).energy!,0);
 close(controlGramians(.1,1,1,0,1).energy!,100*controlGramians(1,1,1,0,1).energy!);assert.equal(controlGramians(.1,0,1,0,1).controllableRank,2);
});
function batchCondition(q:number,R:number,dt:number,N:number,drop:number,p0:number,measurements:(number|null)[]){
 const width=2+2*N;let D=Array.from({length:2},()=>Array(width).fill(0) as number[]);D[0][0]=Math.sqrt(p0);D[1][1]=Math.sqrt(p0);const H:number[][]=[],ys:number[]=[];
 for(let t=1;t<=N;t++){D=[D[0].map((v,i)=>v+dt*D[1][i]),D[1].slice()];D[0][2+2*(t-1)]+=Math.sqrt(q)*dt*dt/2;D[1][2+2*(t-1)]+=Math.sqrt(q)*dt;const h=D[0].slice();h[3+2*(t-1)]+=Math.sqrt(R);if(drop===0||t%drop!==0){H.push(h);ys.push(measurements[t]!);}}
 const prior=multiply(D,transpose(D));if(!H.length)return {mean:[0,0],covariance:prior};const cross=multiply(D,transpose(H)),S=multiply(H,transpose(H)),weights=multiply(cross,inv(S)),mean=multiply(weights,ys.map(v=>[v])).map(r=>r[0]),reduction=multiply(weights,transpose(cross));return {mean,covariance:prior.map((row,i)=>row.map((v,j)=>v-reduction[i][j]))};
}
test('Kalman recursion equals independent batch Gaussian conditioning on complete observation histories',()=>{
 for(const q of [0,.2,1])for(const R of [.1,1])for(const dt of [.2,1])for(const drop of [0,1,2])for(const seed of [7,19]){const r=kalmanPositionVelocity(q,R,dt,5,drop,seed,.5),batch=batchCondition(q,R,dt,5,drop,.5,r.rows.map(v=>v.measurement));r.last.mean.forEach((v,i)=>close(v,batch.mean[i],2e-8));r.last.posterior.forEach((row,i)=>row.forEach((v,j)=>close(v,batch.covariance[i][j],2e-8)));}
});
test('Joseph covariances stay symmetric positive definite and each measurement reduces predicted uncertainty',()=>{
 for(const q of [0,1])for(const R of [.05,2])for(const drop of [0,1,4]){const r=kalmanPositionVelocity(q,R,1,24,drop,999,2);for(const row of r.rows){const P=row.posterior;close(P[0][1],P[1][0]);assert.ok(P[0][0]>0&&P[1][1]>0&&P[0][0]*P[1][1]-P[0][1]**2>0);if(row.used){const K=row.gain,S=row.predicted[0][0]+R;for(let i=0;i<2;i++)for(let j=0;j<2;j++)close(row.predicted[i][j]-P[i][j],K[i]*S*K[j]);}else assert.deepEqual(row.posterior,row.predicted);}}
});
test('dropping measurements changes information rather than noise paths, and seeded histories preserve prefixes',()=>{
 const full=kalmanPositionVelocity(.2,.5,.5,16,0,7,1),drop=kalmanPositionVelocity(.2,.5,.5,16,4,7,1),short=kalmanPositionVelocity(.2,.5,.5,8,0,7,1);assert.deepEqual(full.rows.map(v=>v.truth),drop.rows.map(v=>v.truth));assert.deepEqual(full.rows.map(v=>v.rawMeasurement),drop.rows.map(v=>v.rawMeasurement));assert.deepEqual(short.rows,full.rows.slice(0,9));assert.equal(drop.observationCount,12);
});
test('one-step update and all-missing prediction have independently known covariance values',()=>{
 const r=kalmanPositionVelocity(0,1,1,1,0,7,1);close(r.last.gain[0],2/3);close(r.last.gain[1],1/3);close(r.last.posterior[0][0],2/3);close(r.last.posterior[0][1],1/3);close(r.last.posterior[1][1],2/3);
 const missing=kalmanPositionVelocity(0,1,1,4,1,7,1);assert.equal(missing.observationCount,0);assert.deepEqual(missing.last.posterior,[[17,4],[4,1]]);assert.equal(missing.last.innovation,null);
});
test('joint Gaussian contour has chi-square-two threshold rather than one-dimensional two-sigma coverage',()=>{
 const r=kalmanPositionVelocity(.2,.5,.5,12,0,7,1),W=inv(r.last.posterior),threshold=-2*Math.log(.05);close(1-Math.exp(-threshold/2),.95);assert.ok(Math.abs(1-Math.exp(-2)-.95)>.08);for(const x of r.ellipse)close(x[0]*(W[0][0]*x[0]+W[0][1]*x[1])+x[1]*(W[1][0]*x[0]+W[1][1]*x[1]),threshold);const e=r.error;close(r.mahalanobis,e[0]*(W[0][0]*e[0]+W[0][1]*e[1])+e[1]*(W[1][0]*e[0]+W[1][1]*e[1]));
});
test('Lyapunov matrices certify negative derivatives for all tested states and exact positive principal minors',()=>{
 for(const a of [.25,1,2])for(const b of [.25,1,2])for(const k of [-4,0,4]){const r=lyapunovMetric(a,b,k,3,1,1),P=r.P,A=r.A,identity=multiply(transpose(A),P),other=multiply(P,A);for(let i=0;i<2;i++)for(let j=0;j<2;j++)close(identity[i][j]+other[i][j],-Number(i===j));const det=P[0][0]*P[1][1]-P[0][1]**2;close(det,1/(4*a*b)+k*k/(4*a*b*(a+b)**2));assert.ok(det>0&&P[0][0]>0&&r.minEigen>0);for(const row of r.rows){close(row.Vdot,-row.euclidean);assert.ok(row.V<=r.V0+1e-9);}for(const x of r.contour)close(x[0]*(P[0][0]*x[0]+P[0][1]*x[1])+x[1]*(P[1][0]*x[0]+P[1][1]*x[1]),r.V0);}
});
test('analytic triangular trajectories match independent high-resolution RK4 integration',()=>{
 for(const [a,b,k,T,x1,x2] of [[1,1,4,3,1,1],[.5,1.5,3,4,1,1],[1,1+1e-12,-4,5,-1,1]]){const r=lyapunovMetric(a,b,k,T,x1,x2);let x=[x1,x2];const h=T/10000,F=(v:number[])=>[-a*v[0]+k*v[1],-b*v[1]];for(let t=0;t<10000;t++){const k1=F(x),k2=F(x.map((v,i)=>v+h*k1[i]/2)),k3=F(x.map((v,i)=>v+h*k2[i]/2)),k4=F(x.map((v,i)=>v+h*k3[i]));x=x.map((v,i)=>v+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6);}r.rows.at(-1)!.x.forEach((v,i)=>close(v,x[i]));}
});
test('repeated-pole trajectories and their nearby-rate limit retain the correct transfer term',()=>{
 const equal=lyapunovMetric(1,1,4,5,1,1),near=lyapunovMetric(1,1+1e-12,4,5,1,1);for(const [i,row] of equal.rows.entries()){close(row.x[0],Math.exp(-row.t)*(1+4*row.t));close(row.x[1],Math.exp(-row.t));row.x.forEach((v,j)=>close(v,near.rows[i].x[j]));}
});
test('nonnormal transient growth coexists with Lyapunov decrease and zero states stay exactly zero',()=>{
 const r=lyapunovMetric(1,1,4,3,1,1);close(r.initialEuclideanDerivative,4);close(r.initialVdot,-2);close(r.V0,7);assert.ok(r.rows.some(v=>v.euclidean>2));assert.ok(r.rows.at(-1)!.V<r.V0);close(lyapunovMetric(1,2,0,3,1,1).initialEuclideanDerivative,-6);
 const zero=lyapunovMetric(1,1,4,3,0,0);assert.ok(zero.rows.every(v=>v.V===0&&v.euclidean===0));assert.ok(zero.contour.flat().every(v=>v===0));
});
for(const id of controlKinds)test(`control guides, presets, domain boundaries and finite plotted results: ${id}`,()=>{
 const spec=controlLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);const initial=spec.controls.map(c=>c.initial),vectors=[initial,...spec.presets.map(v=>v.values)];for(const [i,c] of spec.controls.entries())for(const value of [c.min,c.max]){const p=initial.slice();p[i]=value;vectors.push(p);}
 for(const p of vectors){const r=evaluateControl(id,p);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const value of Object.values(r.stats).flat())assert.ok(value===null||Number.isFinite(value));assert.ok(!/[가-힣]/.test(r.statement!));if(r.phase)assert.ok([...r.phase.curves.flat(2),...r.phase.dots.flat()].every(Number.isFinite));}
 assert.throws(()=>evaluateControl(id,[]));for(const [i,c] of spec.controls.entries()){const p=initial.slice();p[i]=NaN;assert.throws(()=>evaluateControl(id,p));if(c.step===1){const bad=initial.slice();bad[i]=c.min+.5;assert.throws(()=>evaluateControl(id,bad));}}
});
for(const id of controlKinds)for(const lang of ['ko','en'] as const)test(`control theory, worked solution and rendered equations: ${id}.${lang}`,async()=>{
 const raw=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),content=await getLearningContent(id);assert.ok((raw.match(/^## /gm)||[]).length>=6);assert.ok(raw.includes('$$'));assert.ok(raw.includes('https://'));assert.ok(lang==='en'?/exercise/i.test(raw)&&/solution|answer/i.test(raw):/문제/.test(raw)&&/풀이|해설/.test(raw));if(lang==='en'){assert.ok(raw.split(/\s+/).length>400);assert.ok(!/[가-힣]/.test(raw));}else assert.ok((raw.match(/[가-힣]/g)||[]).length>700);assert.ok(content[lang].includes('katex-display'));assert.ok(!content[lang].includes('katex-error'));
});
