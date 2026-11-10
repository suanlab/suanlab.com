/** Independent direct rational state enumeration and labelled set partitions.
 * Arithmetic reads only owned input; response fractions are never parsed.
 * No producer arithmetic, common-factor squared weights or fast Walsh transform.
 */
import {Rational as Q,Z,O,sum} from './learning-graph-gauge-validation-math';
import {readNongaussianFieldInput,type NongaussianFieldInput} from './learning-nongaussian-field-input';
import type {NongaussianFieldResult} from './learning-nongaussian-field-protocol';
import {snapshotQubitWire} from './learning-qubit-wire';
const product=(values:Q[])=>values.reduce((a,b)=>a.mul(b),O),integer=(n:number)=>new Q(BigInt(n));
function power(value:Q,exponent:number){let out=O;for(let i=0;i<Math.abs(exponent);i++)out=out.mul(value);return exponent<0?O.div(out):out;}
function gcd(a:bigint,b:bigint){while(b){const rest=a%b;a=b;b=rest;}return a;}
function partitions(slots:number[]):number[][][]{
 if(!slots.length)return[[]];const[first,...rest]=slots,result:number[][][]=[];
 for(const p of partitions(rest)){result.push([[first],...p]);for(let i=0;i<p.length;i++)result.push(p.map((block,j)=>i===j?[first,...block]:block));}return result;
}
function reconstruct(value:NongaussianFieldInput){
 const input=readNongaussianFieldInput(value);if(!input)throw Error('Invalid spin-field input');const n=input.sites,N=2**n,T=Q.read(input.temperature),q=Q.read(input.pairFactor),r=Q.read(input.fourBodyFactor),sources=input.sourceFactors.map(Q.read),edges=Array.from({length:n-1},(_,i)=>[i,i+1]);if(input.boundary==='periodic')edges.push([n-1,0]);
 const states=Array.from({length:N},(_,index)=>Array.from({length:n},(_,i)=>Math.floor(index/2**i)%2?-1:1)),edgeValues=states.map(s=>edges.reduce((v,[i,j])=>v+s[i]*s[j],0)),fourValues=states.map(s=>s[0]*s[1]*s[2]*s[3]);
 const rationalWeights=states.map((s,index)=>power(q,edgeValues[index]).mul(power(r,fourValues[index])).mul(product(sources.map((p,i)=>power(p,s[i])))));
 // Align already-constructed rational weights by their LCM. This avoids repeated
 // rational additions in the direct moment dot products at the largest input.
 const lcm=rationalWeights.reduce((d,w)=>d/gcd(d,w.d)*w.d,BigInt(1)),weights=rationalWeights.map(w=>w.n*(lcm/w.d)),total=weights.reduce((a,b)=>a+b,BigInt(0));
 const subsets=Array.from({length:N},(_,mask)=>Array.from({length:n},(_,i)=>i).filter(i=>Math.floor(mask/2**i)%2===1)),sign=(s:number[],indices:number[])=>indices.reduce((v,i)=>v*s[i],1);
 const numerators=subsets.map(indices=>weights.reduce((v,w,index)=>v+BigInt(sign(states[index],indices))*w,BigInt(0))),moments=numerators.map(v=>new Q(v,total));
 const moment=(indices:number[])=>moments[indices.reduce((mask,i)=>mask^(1<<i),0)],mean=states[0].map((_,i)=>moment([i])),raw=mean.map((_,i)=>mean.map((__,j)=>moment([i,j]))),C=raw.map((row,i)=>row.map((v,j)=>v.sub(mean[i].mul(mean[j]))));
 const cumulant=(indices:number[])=>sum(partitions(indices.map((_,i)=>i)).map(p=>{let coefficient=(-1)**(p.length-1);for(let i=2;i<p.length;i++)coefficient*=i;return integer(coefficient).mul(product(p.map(block=>moment(block.map(i=>indices[i])))));}));
 const probabilities=weights.map(w=>new Q(w,total)),indices=[...input.probes],[i,j,k,l]=indices,pairs=[[i,j,k,l],[i,k,j,l],[i,l,j,k]],wick=pairs.map(([a,b,c,d])=>C[a][b].mul(C[c][d])),k3=cumulant(indices.slice(0,3)),k4=cumulant(indices),centeredFourth=sum(states.map((s,index)=>probabilities[index].mul(product(indices.map(a=>integer(s[a]).sub(mean[a]))))));
 if(centeredFourth.sub(sum(wick)).text()!==k4.text())throw Error('Independent cumulant and direct centering disagree');
 for(let state=0;state<N;state++){const inverse=numerators.reduce((v,num,mask)=>v+BigInt(sign(states[state],subsets[mask]))*num,BigInt(0));if(inverse!==BigInt(N)*weights[state])throw Error('Direct character inversion failed');}
 const avg=(values:number[])=>sum(values.map((v,index)=>probabilities[index].mul(integer(v)))),edgeMean=avg(edgeValues),fourMean=avg(fourValues);
 const canonicalD=product([...sources,...edges.map(()=>q),r].map(v=>new Q(v.n*v.d))).n,canonicalWeights=rationalWeights.map(w=>w.mul(new Q(canonicalD))),canonicalTotal=sum(canonicalWeights);
 let odd=Z;for(let mask=0;mask<N;mask++)if(subsets[mask].length%2){const v=moments[mask].n<BigInt(0)?moments[mask].neg():moments[mask];if(v.n*odd.d>odd.n*v.d)odd=v;}
 const wire=(a:Q[][])=>a.map(row=>row.map(v=>v.text()));
 return{input,edges,stateCount:N,normalization:{partition:new Q(total,lcm).text(),integerWeightSum:canonicalTotal.text(),commonDenominator:String(canonicalD)},states:states.map((spins,index)=>({index,spins,integerWeight:canonicalWeights[index].text(),probability:probabilities[index].text()})),parityMoments:moments.map(v=>v.text()),mean:mean.map(v=>v.text()),rawTwo:wire(raw),connectedTwo:wire(C),susceptibility:wire(C.map(row=>row.map(v=>v.div(T)))),threePoint:{indices:indices.slice(0,3),raw:moment(indices.slice(0,3)).text(),centered:k3.text(),connectedCumulant:k3.text()},fourPoint:{indices,raw:moment(indices).text(),centered:centeredFourth.text(),meanProduct:product(indices.map(a=>mean[a])).text(),wickPairProducts:wick.map(v=>v.text()),connectedCumulant:k4.text(),incorrectRawPairSubtraction:moment(indices).sub(sum(pairs.map(([a,b,c,d])=>raw[a][b].mul(raw[c][d])))).text()},sourceDerivatives:{logZGradient:mean.map(v=>v.div(T).text()),logZHessian:wire(C.map(row=>row.map(v=>v.div(power(T,2))))),logZThirdSelected:k3.div(power(T,3)).text(),logZFourthSelected:k4.div(power(T,4)).text(),meanSecondSelected:k3.div(power(T,2)).text(),meanThirdSelected:k4.div(power(T,3)).text()},conjugate:{edgeMean:edgeMean.text(),edgeVariance:avg(edgeValues.map(v=>v*v)).sub(edgeMean.mul(edgeMean)).text(),fourBodyMean:fourMean.text(),fourBodyVariance:O.sub(fourMean.mul(fourMean)).text(),edgeFourBodyCovariance:avg(edgeValues.map((v,index)=>v*fourValues[index])).sub(edgeMean.mul(fourMean)).text()},magnetization:Array.from({length:n+1},(_,down)=>{const m=n-2*down;return{magnetization:m,probability:sum(probabilities.filter((_,index)=>states[index].reduce((a,b)=>a+b,0)===m)).text()};}),certificates:{probabilitySum:sum(probabilities).text(),zeroMaskMoment:moments[0].text(),walshInversionResidual:'0',positiveStateWeights:rationalWeights.every(w=>w.n>BigInt(0)),spinDiagonalResidual:C.map((row,a)=>row[a].sub(O.sub(mean[a].mul(mean[a]))).text()),spinFlipSymmetric:sources.every(p=>p.n===p.d),oddMomentMaximumAbs:odd.text()},contract:'Finite classical spins, 4..8 sites, open or periodic edges. Positive rational Boltzmann factors q=exp(J/T), r=exp(g/T), p_i=exp(h_i/T); four-body interaction acts on sites0,1,2,3 only. T alone at fixed factors rescales physical energies; it does not heat a fixed Hamiltonian. Counting measure over every spin configuration; no sampling, continuum or thermodynamic limit.'};
}
function same(a:unknown,b:unknown):boolean{
 if(b===null||typeof b!=='object')return a===b;if(!a||typeof a!=='object'||Array.isArray(a)!==Array.isArray(b))return false;
 if(Array.isArray(b))return Array.isArray(a)&&a.length===b.length&&b.every((v,i)=>same(a[i],v));const x=a as Record<string,unknown>,y=b as Record<string,unknown>,keys=Object.keys(y);return Object.keys(x).length===keys.length&&keys.every(k=>Object.hasOwn(x,k)&&same(x[k],y[k]));
}
const cache=new Map<string,unknown>();
export function validateNongaussianFieldResult(source:NongaussianFieldInput,value:unknown):NongaussianFieldResult|null{
 try{const input=readNongaussianFieldInput(source);if(!input)return null;const owned=snapshotQubitWire(value),key=JSON.stringify(input);let expected=cache.get(key);if(!expected){expected=snapshotQubitWire(reconstruct(input));if(cache.size===16)cache.delete(cache.keys().next().value!);cache.set(key,expected);}return same(owned,expected)?owned as NongaussianFieldResult:null;}catch{return null;}
}
