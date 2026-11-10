/** Two-parameter constructed bipartite states with fixed maximally mixed marginals. */
import {f,zero,one,two,add,sub,mul,div,le,eq,q,observe,type F}from'./learning-quantum-hilbert-core';
import {point,plus,minus,scale,intersect,log2Range,binaryEntropyRange}from'./learning-quantum-information-entropies';
function probability(x:F){if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.d<=BigInt(0)||x.n<BigInt(0)||x.n>x.d||x.n.toString().length>32||x.d.toString().length>32)throw new RangeError('Probability in[0,1] with at most32 rational digits required');}
export function correlatedBellEntropy(correlationWeight:F,coherence:F){
 const c=correlationWeight,t=coherence;probability(c);probability(t);
 const four=f(BigInt(4)),a=div(add(one,c),four),b=div(sub(one,c),four),off=div(mul(c,t),two);
 const eigenvalues=[add(a,off),sub(a,off),b,b];
 const terms=eigenvalues.filter(x=>x.n>BigInt(0)).map(x=>scale(log2Range(point(x)),sub(zero,x)));
 const joint=eq(c,zero)?point(two):eq(c,one)&&eq(t,zero)?point(one):eq(c,one)&&eq(t,one)?point(zero):intersect(terms.reduce(plus,point(zero)),zero,two);
 const dephased=plus(point(one),binaryEntropyRange(point(div(sub(one,c),two))));
 const complex=(x:F)=>({real:q(x),imaginary:q(zero)});
 const density=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>complex(i===j?(i===0||i===3?a:b):(i===0&&j===3)||(i===3&&j===0)?off:zero)));
 const local=[[complex(div(one,two)),complex(zero)],[complex(zero),complex(div(one,two))]];
 const pptEigenvalues=[a,a,add(b,off),sub(b,off)],negative=sub(off,b),negativity=le(negative,zero)?zero:negative;
 return {correlationWeight:q(c),coherence:q(t),density,localDensityA:local,localDensityB:local,jointSpectrum:eigenvalues.map(q),
  localEntropyA:q(one),localEntropyB:q(one),jointEntropy:observe(joint),conditionalEntropy:observe(minus(joint,point(one))),mutualInformation:observe(minus(point(two),joint)),
  dephasedJointEntropy:observe(dephased),dephasingEntropyGain:observe(eq(c,zero)||eq(t,zero)?point(zero):intersect(minus(dephased,joint),zero,two)),rawDephasingEntropyGain:observe(minus(dephased,joint)),
  partialTransposeEigenvalues:pptEigenvalues.map(q),negativity:q(negativity),entangled:!eq(negativity,zero),
  contract:'Constructed rho=c[rho_Bell_dephased(t)]+(1-c)I4/4, where rho_Bell_dephased(t) has00/11 populations1/2 and coherence t/2. Exact rational spectra and PPT classification;256-bit directed entropy logs. Partial-transpose negativity detects all entanglement for this2x2 family. No arbitrary-state entropy/eigensolver, Worker/public input reader/UI.'};
}
