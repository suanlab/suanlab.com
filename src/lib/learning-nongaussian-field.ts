/** Exact finite spin moments by integer weights and a Walsh transform; Worker-only. */
import {readNongaussianFieldInput} from './learning-nongaussian-field-input';
import {f,zero,one,add,sub,mul,div,neg,sum,type F} from './learning-quantum-hilbert-core';
const read=(s:string)=>{const[a,b='1']=s.split('/');return f(BigInt(a),BigInt(b));},text=(x:F)=>x.d===BigInt(1)?String(x.n):`${x.n}/${x.d}`;
const bits=(mask:number)=>{let n=0;while(mask){n+=mask&1;mask>>>=1;}return n;};
function walsh(values:bigint[]){const a=values.slice();for(let width=1;width<a.length;width*=2)for(let start=0;start<a.length;start+=2*width)for(let j=0;j<width;j++){const x=a[start+j],y=a[start+j+width];a[start+j]=x+y;a[start+j+width]=x-y;}return a;}
export function calculateNongaussianField(value:unknown){
 const input=readNongaussianFieldInput(value);if(!input)throw RangeError('Invalid finite interacting spin field');const n=input.sites,count=2**n,T=read(input.temperature),q=read(input.pairFactor),r=read(input.fourBodyFactor),sources=input.sourceFactors.map(read),edges=Array.from({length:n-1},(_,i)=>[i,i+1]);if(input.boundary==='periodic')edges.push([n-1,0]);
 const factors=[...sources.map((factor,i)=>({factor,mask:1<<i})),...edges.map(([i,j])=>({factor:q,mask:(1<<i)^(1<<j)})),{factor:r,mask:15}],denominator=factors.reduce((v,{factor:a})=>v*a.n*a.d,BigInt(1));
 // For a/b raised to ±1, multiply by ab and use a² or b². All states share this denominator.
 const weights=Array.from({length:count},(_,state)=>factors.reduce((w,{factor:a,mask})=>w*(bits(state&mask)%2?a.d*a.d:a.n*a.n),BigInt(1))),spectrum=walsh(weights),total=spectrum[0],moments=spectrum.map(v=>f(v,total));
 const moment=(indices:number[])=>moments[indices.reduce((mask,i)=>mask^(1<<i),0)],mean=Array.from({length:n},(_,i)=>moment([i])),raw=mean.map((_,i)=>mean.map((__,j)=>moment([i,j]))),C=raw.map((row,i)=>row.map((v,j)=>sub(v,mul(mean[i],mean[j]))));
 const centered=(indices:number[])=>{let result=zero;for(let mask=0;mask<2**indices.length;mask++){let term=moment(indices.filter((_,i)=>(mask&(1<<i))!==0));for(let i=0;i<indices.length;i++)if(!(mask&(1<<i)))term=mul(term,neg(mean[indices[i]]));result=add(result,term);}return result;};
 const indices=[...input.probes],[i,j,k,l]=indices,pairings=[[i,j,k,l],[i,k,j,l],[i,l,j,k]],wick=pairings.map(([a,b,c,d])=>mul(C[a][b],C[c][d])),center4=centered(indices),cumulant4=sub(center4,sum(wick)),threeIndices=indices.slice(0,3),cumulant3=centered(threeIndices);
 const edgeMean=sum(edges.map(edge=>moment(edge))),edgeSecond=sum(edges.flatMap(a=>edges.map(b=>moment([...a,...b])))),fourBody=moment([0,1,2,3]),cross=sum(edges.map(edge=>moment([...edge,0,1,2,3])));
 const histogram=Array.from({length:n+1},(_,down)=>({magnetization:n-2*down,weight:weights.reduce((v,w,state)=>bits(state)===down?v+w:v,BigInt(0))}));
 const inverse=walsh(spectrum);if(weights.some((w,state)=>w<=BigInt(0)||inverse[state]!==BigInt(count)*w))throw Error('Invalid exact spin probability transform');
 const odd=moments.filter((_,mask)=>bits(mask)%2);let largest=zero;for(const v of odd){const a=v.n<BigInt(0)?neg(v):v;if(a.n*largest.d>largest.n*a.d)largest=a;}
 const diagonal=C.map((row,a)=>sub(row[a],sub(one,mul(mean[a],mean[a]))));if(diagonal.some(v=>v.n!==BigInt(0)))throw Error('Spin diagonal identity failed');
 return{input,edges,stateCount:count,normalization:{partition:text(f(total,denominator)),integerWeightSum:String(total),commonDenominator:String(denominator)},
 states:weights.map((w,state)=>({index:state,spins:Array.from({length:n},(_,a)=>(state&(1<<a))?-1:1),integerWeight:String(w),probability:text(f(w,total))})),parityMoments:moments.map(text),mean:mean.map(text),rawTwo:raw.map(row=>row.map(text)),connectedTwo:C.map(row=>row.map(text)),susceptibility:C.map(row=>row.map(v=>text(div(v,T)))),
 threePoint:{indices:threeIndices,raw:text(moment(threeIndices)),centered:text(cumulant3),connectedCumulant:text(cumulant3)},
 fourPoint:{indices,raw:text(moment(indices)),centered:text(center4),meanProduct:text(indices.map(a=>mean[a]).reduce(mul,one)),wickPairProducts:wick.map(text),connectedCumulant:text(cumulant4),incorrectRawPairSubtraction:text(sub(moment(indices),sum(pairings.map(([a,b,c,d])=>mul(raw[a][b],raw[c][d])))))},
 sourceDerivatives:{logZGradient:mean.map(v=>text(div(v,T))),logZHessian:C.map(row=>row.map(v=>text(div(v,mul(T,T))))),logZThirdSelected:text(div(cumulant3,mul(mul(T,T),T))),logZFourthSelected:text(div(cumulant4,mul(mul(T,T),mul(T,T)))),meanSecondSelected:text(div(cumulant3,mul(T,T))),meanThirdSelected:text(div(cumulant4,mul(mul(T,T),T)))},
 conjugate:{edgeMean:text(edgeMean),edgeVariance:text(sub(edgeSecond,mul(edgeMean,edgeMean))),fourBodyMean:text(fourBody),fourBodyVariance:text(sub(one,mul(fourBody,fourBody))),edgeFourBodyCovariance:text(sub(cross,mul(edgeMean,fourBody)))},
 magnetization:histogram.map(bin=>({magnetization:bin.magnetization,probability:text(f(bin.weight,total))})),
 certificates:{probabilitySum:text(sum(weights.map(w=>f(w,total)))),zeroMaskMoment:text(moments[0]),walshInversionResidual:'0',positiveStateWeights:true,spinDiagonalResidual:diagonal.map(text),spinFlipSymmetric:sources.every(p=>p.n===p.d),oddMomentMaximumAbs:text(largest)},
 contract:'Finite classical spins, 4..8 sites, open or periodic edges. Positive rational Boltzmann factors q=exp(J/T), r=exp(g/T), p_i=exp(h_i/T); four-body interaction acts on sites0,1,2,3 only. T alone at fixed factors rescales physical energies; it does not heat a fixed Hamiltonian. Counting measure over every spin configuration; no sampling, continuum or thermodynamic limit.'};
}
export type NongaussianFieldResult=ReturnType<typeof calculateNongaussianField>;
