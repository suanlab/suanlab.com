import test from 'node:test';
import assert from 'node:assert/strict';
import { binaryEntropy, finiteEntropy, finiteCrossEntropy, finiteKL, renyiEntropy, renyiDivergence, isPrefixFree, bernoulliChannelInformation, evaluateInformation as evaluate, informationKinds } from '../src/lib/learning-information';
import { informationLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const H=(p:number[])=>p.reduce((s,v)=>s+(v>0?-v*Math.log(v)/Math.log(2):0),0);
const MI=(joint:number[][])=>{const px=joint.map(row=>row.reduce((s,v)=>s+v,0)),py=joint[0].map((_,j)=>joint.reduce((s,row)=>s+row[j],0));return joint.reduce((s,row,i)=>s+row.reduce((sum,v,j)=>sum+(v>0?v*Math.log2(v/(px[i]*py[j])):0),0),0);};
const num=(r:ReturnType<typeof evaluate>,key:string)=>{assert.equal(typeof r.stats[key],'number');return r.stats[key] as number;};
test('binary entropy matches direct finite sums with deterministic endpoints and a one-bit maximum',()=>{
 for(let i=0;i<=100;i++){const p=i/100;close(binaryEntropy(p),H([1-p,p]));assert.ok(binaryEntropy(p)>=0&&binaryEntropy(p)<=1+1e-14);close(binaryEntropy(p),binaryEntropy(1-p));}
 assert.equal(binaryEntropy(0),0);assert.equal(binaryEntropy(1),0);assert.equal(binaryEntropy(.5),1);assert.ok(binaryEntropy(1e-300)>0);
});
test('zero-probability surprisal is mathematical infinity with zero weighted contribution',()=>{
 const r=evaluate('self-information-entropy',[0,1]);assert.equal(r.stats['selected surprisal (bits)'],Infinity);assert.equal(r.stats['entropy H(X) (bits)'],0);assert.deepEqual(r.table.rows[1],[1,0,'∞',0]);const a=evaluate('self-information-entropy',[.25,1]),b=evaluate('self-information-entropy',[.25,0]);close(num(a,'selected surprisal (bits)'),2);assert.equal(a.stats['entropy H(X) (bits)'],b.stats['entropy H(X) (bits)']);
});
test('conditional entropy and mutual information agree with independent joint-table enumeration',()=>{
 for(const p of [0,.2,.5,.9,1])for(const e of [0,.1,.5,.9,1]){const r=evaluate('conditional-entropy-mutual-information',[p,e]),joint=[[0,0],[0,0]];for(const row of r.table.rows)joint[Number(row[0])][Number(row[1])]=Number(row[2]);close(joint.flat().reduce((s,v)=>s+v,0),1);const px=joint.map(row=>row.reduce((s,v)=>s+v,0)),py=[0,1].map(y=>joint[0][y]+joint[1][y]),mi=MI(joint);close(num(r,'I(X;Y) (bits)'),mi);close(num(r,'H(X,Y) (bits)'),H(joint.flat()));close(num(r,'H(X|Y) (bits)'),H(joint.flat())-H(py));close(num(r,'H(Y|X) (bits)'),H(joint.flat())-H(px));for(const row of r.table.rows)if(Number(row[3])===0)assert.equal(row[4],'—');else close(Number(row[4]),Number(row[2])/Number(row[3]));}
});
test('deterministic inversion retains information while a fair-noise channel erases it',()=>{
 for(const p of [0,.2,.5,.8,1]){close(bernoulliChannelInformation(p,1),binaryEntropy(p));close(bernoulliChannelInformation(p,.5),0);}
 close(num(evaluate('conditional-entropy-mutual-information',[.5,1]),'H(X|Y) (bits)'),0);
});
test('cross-entropy separates entropy and directed KL including both zero-support conventions',()=>{
 for(const p of [0,.2,.5,.8,1])for(const q of [.1,.5,.9]){const P=[1-p,p],Q=[1-q,q];close(finiteCrossEntropy(P,Q),H(P)+finiteKL(P,Q));assert.ok(finiteKL(P,Q)>=0);}
 assert.notEqual(finiteKL([.2,.8],[.5,.5]),finiteKL([.5,.5],[.2,.8]));assert.equal(finiteKL([1,0],[1,0]),0);assert.equal(finiteCrossEntropy([1,0],[1,0]),0);assert.equal(finiteKL([.2,.8],[1,0]),Infinity);assert.equal(evaluate('cross-entropy-kl',[.8,0]).stats['cross entropy H(P,Q) (bits)'],Infinity);
});
test('cross-entropy plot window neither clips readouts nor draws nonfinite endpoints',()=>{
 const r=evaluate('cross-entropy-kl',[.8,1e-300]);assert.ok(num(r,'cross entropy H(P,Q) (bits)')>700);assert.deepEqual(r.dots,[]);assert.ok(r.curves.flat().every(p=>p.every(Number.isFinite)));assert.equal(evaluate('cross-entropy-kl',[.8,0]).stats['1{Q covers P support}'],0);assert.equal(evaluate('cross-entropy-kl',[0,0]).stats['1{Q covers P support}'],1);
});
test('Jensen gaps are nonnegative and minus-log curvature matches finite differences',()=>{
 for(const weight of [0,.2,.5,1])for(const u of [.1,.5,2,4])for(const v of [.1,1,4]){const r=evaluate('jensen-information-inequalities',[weight,u,v]);assert.ok(num(r,'Jensen gap')>=0);if(weight===0||weight===1||u===v)close(num(r,'Jensen gap'),0);}
 const r=evaluate('jensen-information-inequalities',[.5,.5,2]);close(num(r,'Jensen gap'),Math.log2(1.25));for(const t of [.2,.5,1,3]){const h=1e-4,f=(x:number)=>-Math.log2(x),curvature=(f(t+h)-2*f(t)+f(t-h))/(h*h);close(curvature,1/(t*t*Math.LN2),6e-6);}
});
test('Jensen KL argument handles target support smaller than comparison support',()=>{
 close(finiteKL([1,0],[.3,.7]),-Math.log2(.3));const P=[.4,.6,0],Q=[.1,.2,.7],meanRatio=P.reduce((s,p,i)=>s+(p>0?Q[i]:0),0);assert.ok(finiteKL(P,Q)>=-Math.log2(meanRatio));
});
test('data processing information agrees with all eight joint masses and marginal channel products',()=>{
 for(const p of [0,.2,.5,1])for(const e1 of [0,.1,.5])for(const e2 of [0,.2,.5,1]){const r=evaluate('data-processing-information',[p,e1,e2]),xy=[[0,0],[0,0]],xz=[[0,0],[0,0]];let total=0;for(const row of r.table.rows){const [x,y,z,v]=row.map(Number);xy[x][y]+=v;xz[x][z]+=v;total+=v;}close(total,1);close(num(r,'I(X;Y) (bits)'),MI(xy));close(num(r,'I(X;Z) (bits)'),MI(xz));assert.ok(MI(xz)<=MI(xy)+1e-12);close(num(r,'effective flip probability'),e1*(1-e2)+(1-e1)*e2);if(e2===1)close(num(r,'information lost (bits)'),0);}
});
test('direct access to the source is a counterexample when the Markov assumption is removed',()=>{
 const independent=[[.25,.25],[.25,.25]],copied=[[.5,0],[0,.5]];close(MI(independent),0);close(MI(copied),1);
});
test('prefix-code checks identify ambiguity and two valid codebooks independently round-trip every three-symbol message',()=>{
 const books=[['0','10','110','111'],['00','01','10','11']];for(const book of books){assert.ok(isPrefixFree(book));close(book.reduce((s,w)=>s+2**(-w.length),0),1);for(let message=0;message<64;message++){const symbols=[message%4,Math.floor(message/4)%4,Math.floor(message/16)],encoded=symbols.map(i=>book[i]).join('');let remaining=encoded;const decoded:number[]=[];while(remaining){const i=book.findIndex(w=>remaining.startsWith(w));assert.ok(i>=0);decoded.push(i);remaining=remaining.slice(book[i].length);}assert.deepEqual(decoded,symbols);}}
 assert.equal(isPrefixFree(['0','01','10','11']),false);assert.equal('0'+'10','01'+'0');assert.equal(isPrefixFree(['0','0']),false);assert.equal(isPrefixFree(['']),true);assert.equal(isPrefixFree(['','1']),false);
});
test('two-part MDL counts literal encoded data and headers, and excludes the invalid candidate',()=>{
 for(const pattern of [0,1,2])for(const repeat of [1,4,16])for(const selected of [0,1]){const r=evaluate('coding-compression-mdl',[pattern,selected,repeat]);let literal='';for(const row of r.table.rows)literal+=String(row[3]).repeat(Number(row[2]));const header=selected===0?'10':'0';assert.equal(num(r,'selected valid total bits'),header.length+literal.length);assert.ok(num(r,'selected raw mean length')>=num(r,'empirical entropy (bits/symbol)')-1e-12);}
 const defaultR=evaluate('coding-compression-mdl',[0,0,1]);assert.equal(defaultR.stats['two-part bits for codebook 0'],16);assert.equal(defaultR.stats['two-part bits for codebook 1'],17);assert.equal(evaluate('coding-compression-mdl',[1,1,1]).stats['winning valid codebook'],1);const invalid=evaluate('coding-compression-mdl',[0,2,1]);assert.equal(invalid.stats['selected valid total bits'],null);assert.equal(invalid.stats['selected Kraft sum'],1.25);
});
test('binary rate-distortion bound holds across all small binary channels, not only the symmetric family',()=>{
 for(let i=0;i<=20;i++)for(let j=0;j<=20;j++){const u=i/20,v=j/20,joint=[[(1-u)/2,u/2],[(1-v)/2,v/2]],distortion=(u+1-v)/2,R=distortion>=.5?0:1-H([distortion,1-distortion]);assert.ok(MI(joint)>=R-1e-12);}
 for(const D of [0,.1,.25,.5]){const r=evaluate('rate-distortion-binary',[D,D]);close(num(r,'test-channel I(X;Y) (bits)'),num(r,'rate-distortion R(D) (bits/symbol)'));}
 assert.equal(evaluate('rate-distortion-binary',[.1,.25]).stats['1{test channel feasible}'],0);assert.ok(num(evaluate('rate-distortion-binary',[.75,.75]),'excess over R at actual distortion')>0);
});
test('Fano values and MAP errors match direct M-by-M joint enumeration for all label counts',()=>{
 for(let M=2;M<=8;M++)for(const e of [0,.1,.3,(M-1)/M,1]){const r=evaluate('fano-error-bounds',[M,e]),joint=Array.from({length:M},(_,x)=>Array.from({length:M},(_,y)=>(x===y?1-e:e/(M-1))/M)),hy=Array.from({length:M},(_,y)=>joint.reduce((s,row)=>s+row[y],0)),conditional=H(joint.flat())-H(hy),success=hy.reduce((s,_,y)=>s+Math.max(...joint.map(row=>row[y])),0);close(num(r,'H(X|Y) (bits)'),conditional);close(num(r,'MAP decoder error'),1-success);close(num(r,'I(X;Y) (bits)'),MI(joint));assert.ok(num(r,'coarse error lower bound')<=1-success+1e-12);assert.ok(conditional<=1+e*Math.log2(M)+1e-12);}
 const inverted=evaluate('fano-error-bounds',[2,1]);assert.equal(inverted.stats['chosen decoder error'],1);assert.equal(inverted.stats['MAP decoder error'],0);assert.equal(inverted.stats['I(X;Y) (bits)'],1);
});
test('Rényi special orders agree with support, collision enumeration and Shannon limits',()=>{
 const P=[.2,.8],Q=[.5,.5],collision=P.reduce((s,p,i)=>s+P.reduce((t,q,j)=>t+(i===j?p*q:0),0),0);close(renyiEntropy(P,0),1);close(renyiEntropy(P,1),H(P));close(renyiEntropy(P,2),-Math.log2(collision));close(renyiDivergence(P,Q,2),Math.log2(P.reduce((s,p,i)=>s+p*p/Q[i],0)));
 for(const delta of [-1e-8,-1e-12,1e-12,1e-8]){close(renyiEntropy(P,1+delta),H(P),1e-7);close(renyiDivergence(P,Q,1+delta),finiteKL(P,Q),1e-7);}
 for(const alpha of [0,.5,1,2,8]){close(renyiEntropy([1,0],alpha),0);close(renyiDivergence([1,0],[.4,.6],alpha),-Math.log2(.4));close(renyiEntropy([.5,.5],alpha),1);}
});
test('Rényi entropy is order-monotone and tiny positive support is distinct from zero',()=>{
 for(const p of [.01,.2,.5,.99]){const P=[1-p,p],orders=[0,.5,1,2,4,8],entropies=orders.map(a=>renyiEntropy(P,a)),divergences=orders.map(a=>renyiDivergence(P,[.4,.6],a));for(let i=1;i<orders.length;i++){assert.ok(entropies[i]<=entropies[i-1]+1e-12);assert.ok(divergences[i]>=divergences[i-1]-1e-12);}assert.ok(entropies.at(-1)!>=-Math.log2(Math.max(p,1-p))-1e-12);}
 assert.equal(renyiEntropy([1e-320,1],0),1);assert.equal(renyiEntropy([0,1],0),0);for(const alpha of [.0001,.5,1,1+1e-12,8])assert.ok(Number.isFinite(renyiEntropy([1e-320,1],alpha)));
});
test('softmax normalization is invariant to common logit shifts and entropy converts between units',()=>{
 const a=evaluate('information-units-normalization',[0,0,1,4]),b=evaluate('information-units-normalization',[4,4,1,4]);for(const key of ['softmax entropy (bits)','softmax entropy (nats)','softmax probability sum'])close(num(a,key),num(b,key));close(num(b,'log normalizer')-num(a,'log normalizer'),4);close(num(a,'softmax entropy (bits)'),1);close(num(a,'softmax entropy (nats)'),Math.log(2));
});
test('uniform density integrates to one and bin entropy differs from differential entropy by log width',()=>{
 for(const scale of [.1,.25,1,4])for(const K of [1,4,16]){const r=evaluate('information-units-normalization',[2,0,scale,K]),density=num(r,'uniform density height'),width=num(r,'bin width Δ');close(density*scale,1);close(density*width,1/K);close(num(r,'differential entropy (nats)'),-scale*density*Math.log(density));close(num(r,'h minus log bin width'),Math.log(K));close(num(r,'quantized entropy (nats)'),Math.log(K));}
 assert.ok(num(evaluate('information-units-normalization',[2,0,.25,4]),'differential entropy (nats)')<0);
});
test('information helpers and controls reject invalid mass functions, orders and domains',()=>{
 for(const f of [()=>binaryEntropy(NaN),()=>binaryEntropy(1.1),()=>finiteEntropy([]),()=>finiteEntropy([.2,.2]),()=>finiteEntropy([-.1,1.1]),()=>finiteCrossEntropy([1],[.5,.5]),()=>finiteKL([1],[NaN]),()=>renyiEntropy([.5,.5],-1),()=>renyiDivergence([.5,.5],[0,1],2),()=>renyiDivergence([.5,.5],[.5,.5],Infinity)])assert.throws(f);
 for(const kind of informationKinds){const values=informationLabs[kind].controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[NaN,...values.slice(1)]));assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[Infinity,...values.slice(1)]));}
});
for(const kind of informationKinds)test(`${kind} has four concept steps and three guided examples with finite plots`,()=>{
 const guide=learningGuides[kind],spec=informationLabs[kind];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat().every(v=>v.every(Number.isFinite)));assert.ok(!/[가-힣]/.test(r.statement??''));for(const value of Object.values(r.stats))if(typeof value==='number')assert.ok(!Number.isNaN(value));}
});

test('structural equality cases display exact zeros without clipping genuinely small information',()=>{
 assert.equal(bernoulliChannelInformation(1,.3),0);assert.equal(bernoulliChannelInformation(0,.9),0);assert.ok(bernoulliChannelInformation(1e-300,0)>0);
 assert.equal(evaluate('jensen-information-inequalities',[.2,.1,.1]).stats['Jensen gap'],0);
 assert.equal(evaluate('data-processing-information',[.2,.1,1]).stats['information lost (bits)'],0);
 assert.equal(evaluate('fano-error-bounds',[3,2/3]).stats['I(X;Y) (bits)'],0);
});
