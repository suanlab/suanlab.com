import test from 'node:test';
import assert from 'node:assert/strict';
import { binaryChannel, binaryEntropyBits, evaluateCoding } from '../src/lib/learning-coding';
const close=(x:number,y:number)=>assert.ok(Math.abs(x-y)<1e-12,`${x} != ${y}`);
test('binary channel information agrees with joint-table logarithms and numerical capacity maximization',()=>{
 for(const erasure of [false,true]) for(const noise of [0,.1,.5,.8,1]) {
  let maximum=0;
  for(let i=0;i<=100;i++) {
   const p=i/100,r=binaryChannel(p,noise,erasure),px=[1-p,p];let direct=0,total=0;
   r.joint.forEach((row,x)=>row.forEach((v,y)=>{total+=v;if(v>0)direct+=v*Math.log2(v/(px[x]*r.output[y]));}));
   close(total,1);close(r.mutualInformation,direct);assert.ok(direct<=r.capacity+1e-12);
   maximum=Math.max(maximum,direct);
  }
  close(maximum,binaryChannel(.5,noise,erasure).capacity);
 }
});
test('flips and erasures have different capacities and deterministic inversion preserves a bit',()=>{
 close(binaryChannel(.5,.1,false).capacity,.5310044064107188);
 close(binaryChannel(.5,.1,true).capacity,.9);
 close(binaryChannel(.5,1,false).mutualInformation,1);
 close(binaryChannel(.5,1,true).mutualInformation,0);
 for(const p of [0,.01,.3,.5,1])close(binaryChannel(p,.5,false).mutualInformation,0);
 close(binaryEntropyBits(0),0);close(binaryEntropyBits(1),0);
});
test('channel information preserves weak contrast and rare inputs without subtracting entropies',()=>{
 for(const delta of [2**-20,2**-30,2**-40]){
  const r=binaryChannel(.5,.5+delta,false),leading=(2*delta)**2/(2*Math.LN2);
  assert.ok(r.capacity>0);assert.ok(Math.abs(r.capacity/leading-1)<1e-10);
  assert.equal(r.mutualInformation,r.capacity);
 }
 const rare=binaryChannel(1e-300,.1,false);
 assert.ok(rare.mutualInformation>0);assert.ok(Math.abs(rare.mutualInformation/1e-300-.8*Math.log2(9))<1e-12);
 assert.equal(binaryChannel(.5,.5,false).capacity,0);
 const under=evaluateCoding('binary-channel-capacity',[0,1e-300,.5+2**-50]);
 assert.equal(under.warning,'numeric-range');assert.equal(under.stats['I(X;Y) (bits/use)'],null);assert.deepEqual(under.dots,[]);
 const erased=evaluateCoding('binary-channel-capacity',[1,1e-320,1-2**-52]);
 assert.equal(erased.warning,'numeric-range');assert.equal(erased.stats['I(X;Y) (bits/use)'],null);assert.deepEqual(erased.dots,[]);
 for(const values of [[0,0,.1],[0,.5,.5],[1,.5,1]]){
  const zero=evaluateCoding('binary-channel-capacity',values);
  assert.equal(zero.stats['I(X;Y) (bits/use)'],0);assert.equal(zero.warning,undefined);
 }
});
test('channel controls reject invalid mode, length, nonfinite values and probabilities',()=>{
 for(const values of [[],[2,.5,.1],[0,NaN,.1],[0,.5,Infinity],[0,-.1,.2],[1,.2,1.1]])assert.throws(()=>evaluateCoding('binary-channel-capacity',values));
 for(const mode of [0,1])for(const noise of [0,.5,1])assert.ok(evaluateCoding('binary-channel-capacity',[mode,.3,noise]).curves.flat(2).every(Number.isFinite));
});

import { repetitionError,hammingEncode,hammingDecode,encodeRuns,decodeRuns,finitePrograms,finiteDecode,finiteSearch,twoPartMDL,fingerprint,fingerprintRows,codingKinds } from '../src/lib/learning-coding';
const pop=(x:number)=>x.toString(2).replace(/0/g,'').length;
test('majority repetition error matches every independent error pattern including boundaries',()=>{
 for(const n of [1,3,5,7,9])for(const noise of [0,.1,.5,.8,1]){let error=0,mass=0;for(let mask=0;mask<2**n;mask++){const k=pop(mask),w=noise**k*(1-noise)**(n-k);mass+=w;if(k>n/2)error+=w;}close(mass,1);close(repetitionError(n,noise),error);}
 close(repetitionError(3,.1),.028);close(repetitionError(9,.5),.5);assert.ok(repetitionError(9,.8)>repetitionError(1,.8));
});
test('unrepresentable repetition error is withheld instead of advertised as zero',()=>{
 const r=evaluateCoding('repetition-error-correction',[4,1e-100,0,511]);
 assert.equal(r.warning,'numeric-range');assert.equal(r.stats['majority block error probability'],null);
 assert.equal(r.stats['selected received-word probability'],null);
 const exact=evaluateCoding('repetition-error-correction',[4,0,0,511]);
 assert.equal(exact.warning,undefined);assert.equal(exact.stats['majority block error probability'],0);
 assert.equal(exact.stats['selected received-word probability'],0);
});
test('Hamming code is linear, has distance three and corrects all single errors for every payload',()=>{
 const words=Array.from({length:16},(_,i)=>hammingEncode(i));assert.equal(new Set(words).size,16);let minimum=7;
 for(let a=0;a<16;a++){assert.equal(hammingDecode(words[a]).syndrome,0);for(let b=0;b<16;b++){assert.equal(hammingEncode(a^b),words[a]^words[b]);if(a!==b)minimum=Math.min(minimum,pop(words[a]^words[b]));}for(let j=0;j<7;j++){const r=hammingDecode(words[a]^(1<<j));assert.equal(r.syndrome,j+1);assert.equal(r.data,a);assert.equal(r.corrected,words[a]);}}
 assert.equal(minimum,3);const bad=hammingDecode(hammingEncode(9)^3);assert.equal(bad.data,8);assert.equal(hammingDecode(bad.corrected).syndrome,0);
});
test('run protocol round trips all eight-bit words and many sixteen-bit words with exact headers',()=>{
 for(const n of [8,16])for(let x=0;x<2**n;x+=n===8?1:127){const word=x.toString(2).padStart(n,'0'),r=encodeRuns(word);assert.equal(decodeRuns(r.raw,n),word);assert.equal(decodeRuns(r.rle,n),word);assert.equal(decodeRuns(r.encoded,n),word);assert.equal(r.rle.length,2+Math.log2(n)*r.runs.length);assert.ok(r.encoded.length<=n+1);}
 const r=encodeRuns('0'.repeat(16));assert.equal(r.encoded.length,6);assert.equal(encodeRuns('01'.repeat(8)).rle.length,66);assert.equal(encodeRuns('01'.repeat(8)).encoded.length,17);
 for(const [code,n] of [['0',8],['11000',16],['1100000000',8],['1',8],['2',8]] as [string,number][])assert.throws(()=>decodeRuns(code,n));
});
test('finite decoder is prefix-free and all 278 programs decode according to their explicit grammar',()=>{
 const programs=finitePrograms();assert.equal(programs.length,278);close(programs.reduce((s,p)=>s+2**(-p.code.length),0),15/16);
 for(const p of programs){assert.equal(finiteDecode(p.code),p.output);for(const q of programs)if(p!==q)assert.ok(!q.code.startsWith(p.code));const expected=p.family===0?parseInt(p.code.slice(1),2):p.family===1?Number(p.code.slice(2))?255:0:p.family===2?parseInt(p.code.slice(3).repeat(4),2):parseInt(p.code.slice(4).repeat(2),2);assert.equal(p.output,expected);}
 for(const [x,L] of [[0,3],[255,3],[85,5],[170,5],[51,8],[173,9]])assert.equal(Math.min(...programs.filter(p=>p.output===x).map(p=>p.code.length)),L);
});
test('bounded coverage counts distinct outputs instead of counting duplicate descriptions',()=>{
 for(let L=0;L<=9;L++){const r=evaluateCoding('kolmogorov-uncomputability',[L,173]),expected=L<3?0:L<5?2:L<8?4:L<9?16:256;assert.equal(r.stats['distinct eight-bit outputs'],expected);assert.ok(expected<=Math.min(256,2**L));assert.equal(r.stats['1{selected byte covered}'],Number(L>=9));}
 assert.equal(evaluateCoding('kolmogorov-uncomputability',[8,0]).stats['programs within budget'],22);
});
test('two-part MDL equals an independently formed ordered-data probability plus prefix cost',()=>{
 for(const successes of [0,4,8,12,16])for(const repeats of [1,8]){const r=twoPartMDL(successes,repeats);r.models.forEach(m=>{let LL=0;for(let i=0;i<r.N;i++)LL-=Math.log2(i<r.K?m.p:1-m.p);close(m.data,LL);close(m.total,m.header.length+LL);});assert.ok(r.models.every(m=>m.total>=r.models[r.winner].total));assert.ok(r.models[2].total>r.models[0].total);}
 assert.equal(twoPartMDL(8,1).winner,0);assert.equal(twoPartMDL(4,1).winner,1);assert.equal(twoPartMDL(12,8).winner,3);
});
test('search budgets give monotone discovered bounds with unknown before a first match',()=>{
 assert.equal(finiteSearch(173,1,false).best,null);
 assert.equal(finiteSearch(0,1,true).best!.code.length,9);assert.equal(finiteSearch(0,257,true).best!.code.length,3);
 for(const target of [0,85,51,173])for(const order of [false,true]){const r=finiteSearch(target,278,order);let prev=Infinity;for(const item of r.history)if(item.best!==null){assert.ok(item.best<=prev);assert.ok(item.best>=r.oracle);prev=item.best;}assert.equal(r.best!.code.length,r.oracle);assert.equal(finiteDecode(r.best!.code),target);}
});
test('each nonzero difference collides on exactly half the uniformly enumerated GF2 public rows',()=>{
 for(let delta=1;delta<256;delta++){let collisions=0;for(let row=0;row<256;row++)if(fingerprint(delta,[row])===0)collisions++;assert.equal(collisions,128);}
 for(const delta of [1,37,255]){let pairs=0;for(let r1=0;r1<256;r1++)for(let r2=0;r2<256;r2++)if(fingerprint(delta,[r1,r2])===0)pairs++;assert.equal(pairs,16384);}
 assert.equal(fingerprint(0,[1,2,4]),fingerprint(128,[1,2,4]));assert.notEqual(fingerprint(0,Array.from({length:8},(_,i)=>1<<i)),fingerprint(128,Array.from({length:8},(_,i)=>1<<i)));assert.deepEqual(fingerprintRows(4,1),fingerprintRows(4,1));
});
test('all extension controls reject bad lengths, nonfinite inputs and out-of-domain integers',()=>{
 const samples:[typeof codingKinds[number],number[]][]=[['repetition-error-correction',[1,.1,0,0]],['hamming-linear-codes',[9,3]],['lossless-compression-reconstruction',[1,0]],['finite-description-length',[85,3]],['kolmogorov-uncomputability',[5,173]],['two-part-mdl-selection',[4,1]],['budgeted-description-search',[0,257,1]],['communication-fingerprints',[0,128,3,0,1]]];
 for(const [kind,values] of samples){assert.throws(()=>evaluateCoding(kind,[]));assert.throws(()=>evaluateCoding(kind,[NaN,...values.slice(1)]));assert.throws(()=>evaluateCoding(kind,[-1,...values.slice(1)]));const r=evaluateCoding(kind,values);assert.ok(r.curves.flat(2).every(Number.isFinite));}
 for(const f of [()=>hammingEncode(16),()=>hammingDecode(-1),()=>repetitionError(2,.1),()=>finiteDecode('1111'),()=>finiteSearch(0,279,true),()=>fingerprintRows(3,0)])assert.throws(f);
});

import {codingLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
for(const kind of codingKinds)test(`${kind}: three authored presets have finite geometry and valid discovery/readout states`,()=>{
 const spec=codingLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);
 for(const experiment of learningGuides[kind].experiments)assert.ok(spec.presets[experiment.preset]);
 for(const preset of spec.presets){const r=evaluateCoding(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.dots?.flat().every(Number.isFinite)??true);for(const value of Object.values(r.stats))if(typeof value==='number')assert.ok(Number.isFinite(value));if(r.bounds){assert.ok(r.bounds[1]>r.bounds[0]);assert.ok(r.bounds[3]>r.bounds[2]);}assert.ok(!/[가-힣]/.test(r.statement??''));}
});

test('displayed parity-check columns use the same top-to-bottom ordering as the chapter matrix',()=>{
 const H=[[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]];
 const r=evaluateCoding('hamming-linear-codes',[9,4]);r.table.rows.forEach((row,j)=>assert.equal(row[1],H.map(line=>line[j]).join('')));
 for(let payload=0;payload<16;payload++)for(const row of H)assert.equal(row.reduce((sum,h,j)=>sum+h*((hammingEncode(payload)>>j)&1),0)%2,0);
});

test('equality fingerprint curves agree with the zero false-positive risk for identical inputs',()=>{
 const same=evaluateCoding('communication-fingerprints',[128,128,3,1,1]);assert.equal(same.stats['ideal random false-positive probability'],0);assert.ok(same.curves[0].every(([,y])=>y===0));
 const distinct=evaluateCoding('communication-fingerprints',[0,128,3,1,1]);distinct.curves[0].forEach(([k,y])=>close(y,2**(-k)));
});

test('a single discovered candidate is shown as a visible point, while an unknown target has no invented point',()=>{
 assert.deepEqual(evaluateCoding('budgeted-description-search',[0,1,1]).dots,[[1,9]]);
 assert.deepEqual(evaluateCoding('budgeted-description-search',[173,1,0]).dots,[]);
});
