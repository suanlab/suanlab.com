/** Exact discrete projection identity. Hidden terms are signed vectors, not laws. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F} from './learning-finite-markov-exact';
import {snapshotMarkovCoarseParameters,type MarkovCoarseParameters} from './learning-markov-coarse-input';
export {isMarkovCoarseParameters,snapshotMarkovCoarseParameters,type MarkovCoarseParameters} from './learning-markov-coarse-input';
const zero=f(BigInt(0)),one=f(BigInt(1)),add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),sum=(v:F[])=>v.reduce(add,zero),eq=(a:F,b:F)=>a.n*b.d===b.n*a.d,abs=(a:F)=>a.n<BigInt(0)?neg(a):a;
type M=F[][];const identity=(n:number):M=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?one:zero));
const mm=(a:M,b:M):M=>a.map(row=>b[0].map((_,j)=>sum(row.map((v,k)=>mul(v,b[k][j])))));
const mv=(a:M,v:F[])=>a.map(row=>sum(row.map((x,j)=>mul(x,v[j]))));
const vadd=(a:F[],b:F[])=>a.map((v,i)=>add(v,b[i])),vsub=(a:F[],b:F[])=>a.map((v,i)=>sub(v,b[i]));
const msub=(a:M,b:M)=>a.map((row,i)=>row.map((v,j)=>sub(v,b[i][j])));
const obs=(v:F)=>exact(v.n,v.d),vec=(v:F[])=>v.map(obs),mat=(m:M)=>m.map(vec),zvec=(n:number)=>Array.from({length:n},()=>zero);
const law=(v:F[])=>eq(sum(v),one)&&v.every(x=>x.n>=BigInt(0)),distance=(a:F[],b:F[])=>div(sum(vsub(a,b).map(abs)),f(BigInt(2)));
export function markovCoarseModel(input:MarkovCoarseParameters){
 const parameters=snapshotMarkovCoarseParameters(input),p=parameters,D=BigInt(p.transitionDenominator),q=(n:number)=>f(BigInt(n),D),a=q(p.weight01),b=q(p.weight02),c=q(p.weight10),d=q(p.weight12),e=q(p.weight20),g=q(p.weight21),alpha=f(BigInt(p.liftNumerator),BigInt(p.liftDenominator));
 // T[target][source] is column-stochastic. Initial weights define a probability law once.
 const T=[[sub(sub(one,a),b),c,e],[a,sub(sub(one,c),d),g],[b,d,sub(sub(one,e),g)]],C=[[one,one,zero],[zero,zero,one]],R=[[alpha,zero],[sub(one,alpha),zero],[zero,one]],H=msub(identity(3),mm(R,C)),A=mm(mm(C,T),R),B=mm(C,T),E=mm(H,T),forcingMap=mm(mm(H,T),R);
 const initial=p.initialWeights.map(n=>f(BigInt(n))),total=sum(initial),p0=initial.map(v=>div(v,total)),m0=mv(C,p0),h0=mv(H,p0),strong=eq(b,d),lambda=sum(E.map((row,i)=>row[i]));
 const kernels:M[]=[],powers:M[]=[identity(3)];for(let lag=0;lag<p.steps;lag++){kernels.push(mm(mm(B,powers[lag]),forcingMap));powers.push(mm(E,powers[lag]));}
 let micro=p0,naive=m0,truncated=m0,basis0=[one,zero,zero],basis1=[zero,one,zero];const macros:F[][]=[],approx:F[][]=[];
 const history=[];
 for(let k=0;k<=p.steps;k++){
  const macro=mv(C,micro);macros.push(macro);approx.push(truncated);
  const instantaneous=mv(A,macro),initialTerm=mv(mm(B,powers[k]),h0);let memory=zvec(2),approxMemory=zvec(2),omitted=zvec(2);
  for(let j=0;j<k;j++){const lag=k-1-j,term=mv(kernels[lag],macros[j]);memory=vadd(memory,term);if(lag<p.memoryDepth)approxMemory=vadd(approxMemory,mv(kernels[lag],approx[j]));else omitted=vadd(omitted,term);}
  const nextMicro=mv(T,micro),nextCoarse=mv(C,nextMicro),reconstructed=vadd(vadd(instantaneous,initialTerm),memory),nextTruncated=vadd(vadd(mv(A,truncated),initialTerm),approxMemory);
  const witness0=mv(C,basis0),witness1=mv(C,basis1),hidden=mv(H,micro);
  history.push({step:k,micro:vec(micro),macro:vec(macro),hidden:vec(hidden),naiveMarkov:vec(naive),initialAwareTruncated:vec(truncated),naiveErrorTV:obs(distance(macro,naive)),truncatedErrorHalfL1:obs(distance(macro,truncated)),truncatedIsProbability:law(truncated),instantaneousTerm:vec(instantaneous),initialHiddenTerm:vec(initialTerm),fullMemoryTerm:vec(memory),omittedMemoryOnExactHistory:vec(omitted),exactNextMacro:vec(nextCoarse),reconstructedNextMacro:vec(reconstructed),reconstructionResidual:vec(vsub(reconstructed,nextCoarse)),microMassResidual:obs(sub(sum(micro),one)),macroMassResidual:obs(sub(sum(macro),one)),witnessFromState0:vec(witness0),witnessFromState1:vec(witness1),witnessDistanceTV:obs(distance(witness0,witness1))});
  micro=nextMicro;naive=mv(A,naive);truncated=nextTruncated;basis0=mv(T,basis0);basis1=mv(T,basis1);
 }
 return{parameters,transition:mat(T),aggregation:mat(C),lifting:mat(R),hiddenProjection:mat(H),instantaneousMacroTransition:mat(A),hiddenPropagation:mat(E),resolvedToHidden:mat(forcingMap),hiddenEigenvalue:obs(lambda),lumpabilityExitDefect:obs(sub(b,d)),classification:{strongLumpableAllInitialLaws:strong,initialHiddenExactlyZero:h0.every(v=>v.n===BigInt(0)),allRetainedLagsForDisplayedHistory:p.memoryDepth>=Math.max(0,p.steps-1),geometricHiddenModeStrictlyDecays:abs(lambda).n<abs(lambda).d,hiddenModeAlternates:lambda.n<BigInt(0),kernelsAreProbabilities:false,physicalNoiseSpecified:false,continuousTimeModel:false},memoryKernels:kernels.map((m,lag)=>({lag,matrix:mat(m)})),history,current:history.at(-1)!,approximationContract:'Three-state discrete column-stochastic T[target][source], blocks A={0,1},B={2}, C R=I with alpha supplied, H=I-RC. All probabilities, matrix powers, signed kernels, forcing, errors and residuals are exact rational arithmetic; Number fields are display projections. Full memory identity m(k+1)=CTR m(k)+CT(HT)^k h0+sum_j CT(HT)^(k-1-j)HTR m(j). Strong closure for every initial law iff the exit probabilities0->2 and1->2 agree. An instantaneous lifting-dependent A=CTR is a stochastic approximation, not automatically the true coarse process. Truncation retains lag0..memoryDepth-1 and the entire initial forcing; it is not projected or clipped back onto probabilities. Its half-L1 error is labeled TV only when both arguments are probability laws. Omitted terms evaluated on the exact history are local defects, not a bound on global truncated rollout error. History spans steps0..steps and includes a separately labeled diagnostic of the next macro update even at its final record. No physical clock, random sampling, added noise, worker, reader or whole-topic completion claim.'};
}
