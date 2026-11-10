/** General mixed-result schema and exact input attribution,without eigensolving. */
import type {mixedEntanglementModel} from './learning-mixed-entanglement';
import type {MixedEntanglementParameters} from './learning-mixed-entanglement-input';
import type {PartitionObservation as Q} from './learning-partition-exact-core';
import {createTensorWireChecks,shape,literal,choice,bool,integer,array,list,nullable,fields,type Check} from './learning-tensor-wire-schema';
import {mixedTensorContract} from './learning-tensor-contracts';
import {f,zero,one,two,int,add,sub,mul,div,neg,sum,eq,le,lo,matches,c,cm,identity,product,matrixScale,matrixSub,trace,projector,tensor,partial,pt,invariants,matrixNorm2,matrixMatches,mixedMatrix,range,point,radd,rsub,rmul,contains,overlap,polynomial,type F,type Matrix} from './learning-tensor-wire-math';

export function validateMixedTensorResult(value:unknown,p:MixedEntanglementParameters):boolean{
 const {q,exact,z,nonnegativeInterval,prob,matrix}=createTensorWireChecks(),params:Check=x=>shape(x,Object.fromEntries(Object.entries(p).map(([k,v])=>[k,literal(v)])));
 const minors=(d:number):Check=>array(2**d-1,v=>shape(v,{indices:list(1,d,w=>integer(w,0,d-1)),determinant:exact}));
 const spectral=(d:number):Check=>list(1,d,v=>shape(v,{eigenvalue:q,multiplicity:w=>integer(w,1,d),sign:choice(-1,0,1),isolatingLower:exact,isolatingUpper:exact,exactRationalRootKnown:bool}));
 const density=(d:number,bloch=false):Check=>x=>shape(x,{density:matrix(d,exact),rank:v=>integer(v,1,d),purity:prob,principalMinors:minors(d),characteristicPolynomial:array(d+1,exact),spectrum:spectral(d),entropyNats:q,entropyBits:q,...(bloch?{blochVector:array(3,exact)}:{})});
 const probabilityResiduals:Check=x=>shape(x,{mass:z,marginalA:array(2,z),marginalB:array(2,z),correlation:z});
 if(!shape(value,{parameters:params,dimensionA:literal(2),dimensionB:literal(2),jointDimension:literal(4),flatteningConvention:literal('A tensor B; flatIndex=2*aIndex+bIndex'),precisionBits:literal(512),arithmeticWorkBits:literal(768),rootIsolationBits:literal(1024),rawHermitianNumerator:matrix(4,exact),normalizationTrace:exact,joint:density(4),reducedA:density(2,true),reducedB:density(2,true),...fields('mutualInformationNats mutualInformationBits conditionalEntropyAGivenBNats conditionalEntropyBGivenANats',q),pauliConvention:literal('X,Y=[[0,-i],[i,0]],Z; tensor order A,B'),correlationTensor:array(3,array(3,exact)),
 partialTranspose:v=>shape(v,{on:literal('B'),matrix:matrix(4,exact),principalMinors:minors(4),characteristicPolynomial:array(5,exact),spectrum:spectral(4),positiveSemidefinite:bool,negativeMultiplicity:n=>integer(n,0,1),negativity:nonnegativeInterval,logNegativityBits:q}),
 localTests:v=>shape(v,{outcomeOrder:array(2,w=>w===1||w===-1),settings:array(4,w=>shape(w,{name:choice('a0','a1','b0','b1'),positiveProjector:matrix(2,exact),observable:matrix(2,exact),blochDirection:array(3,exact)})),pairs:array(2,array(2,w=>shape(w,{settings:array(2,choice('a0','a1','b0','b1')),jointProbabilities:array(2,array(2,prob)),marginalA:array(2,prob),marginalB:array(2,prob),correlation:exact,connected:exact,residuals:probabilityResiduals}))),conditionals:array(4,w=>shape(w,{setting:choice('a0','a1','b0','b1'),conditionedSubsystem:choice('a','b'),rows:array(2,y=>shape(y,{event:n=>integer(n,0,1),probability:prob,unnormalizedDensity:matrix(2,exact),conditionalDensity:nullable(matrix(2,exact)),conditionalPurity:nullable(prob)})),nonselectiveDensity:matrix(2,exact),noSignallingResidual:matrix(2,z)}))}),
 chsh:v=>shape(v,{expression:literal('E00+E01+E10-E11'),selectedSigned:exact,selectedAbsolute:exact,selectedViolates:bool,localHiddenVariableBound:exact,gramMatrix:matrix(3,exact),gramCharacteristicPolynomial:array(4,exact),gramSpectrum:spectral(3),topTwoEigenvalueSum:q,optimizedMaximum:nonnegativeInterval,tsirelsonBound:nonnegativeInterval,optimizedComparisonToLocalBound:choice(-1,0,1),optimizedViolates:bool,optimizedAtBoundary:bool,optimalAxesComputed:literal(false),exactBoundaryCertificate:w=>shape(w,{threshold:exact,shiftedMatrix:matrix(3,exact),principalMinors:minors(3),positiveSemidefinite:bool,determinant:exact})}),
 exactResiduals:v=>shape(v,{trace:w=>shape(w,{real:z,imaginary:z}),...fields('partialTransposeTrace partialTransposeSquaredNorm reducedTraceA reducedTraceB',z),pauliReconstruction:matrix(4,z)}),
 classification:v=>shape(v,{...fields('globallyPure separableTwoQubit entangledTwoQubit localAIsPure localBIsPure selectedChshViolation optimizedChshViolation',bool),noChshViolationProvesLocalHiddenVariableModel:literal(false),pptSufficiencyDimension:literal('2x2 only in this model'),finiteSampleExperimentSupplied:literal(false),physicalTimeSpecified:literal(false)}),approximationContract:literal(mixedTensorContract)}))return false;
 const x=value as ReturnType<typeof mixedEntanglementModel>,H=mixedMatrix(p),N=trace(H).re,rho=matrixScale(H,div(one,N)),A=partial(rho,2,2,'a'),B=partial(rho,2,2,'b'),PT=pt(rho,2),iPT=invariants(PT),I=identity(2),pauli:Matrix[]=[[[c(zero),c(one)],[c(one),c(zero)]],[[c(zero),c(zero,neg(one))],[c(zero,one),c(zero)]],[[c(one),c(zero)],[c(zero),c(neg(one))]]];
 const expect=(state:Matrix,O:Matrix)=>state.reduce((total,row,i)=>row.reduce((s,v,j)=>add(s,cm(v,O[j][i]).re),total),zero),qvec=(v:Q[],expected:F[])=>v.every((q,i)=>matches(q,expected[i]));
 const minorMatch=(v:typeof x.joint.principalMinors,inv:ReturnType<typeof invariants>)=>v.every((m,i)=>m.indices.join(',')===inv.minors[i].indices.join(',')&&matches(m.determinant,inv.minors[i].value.re));
 function spectrumMatches(rows:typeof x.joint.spectrum,m:Matrix,positive:boolean){
  const inv=invariants(m),d=m.length;let mass=point(zero),second=point(zero),count=0;
  for(const[i,v]of rows.entries()){
   const l=lo(v.isolatingLower),h=lo(v.isolatingUpper),box={lower:l,upper:h},sign=l.n===BigInt(0)&&h.n===BigInt(0)?0:h.n<BigInt(0)?-1:l.n>BigInt(0)?1:null;
   if(!le(l,h)||!le(sub(h,l),f(BigInt(1),BigInt(1)<<BigInt(1024)))||i>0&&le(l,lo(rows[i-1].isolatingUpper))||v.exactRationalRootKnown!==eq(l,h)||sign===null||v.sign!==sign||positive&&sign<0||!overlap(box,range(v.eigenvalue))||!contains(polynomial(inv.characteristic,box),zero))return false;
   count+=v.multiplicity;mass=radd(mass,rmul(point(int(v.multiplicity)),range(v.eigenvalue)));second=radd(second,rmul(point(int(v.multiplicity)),rmul(range(v.eigenvalue),range(v.eigenvalue))));
  }
  return count===d&&contains(mass,trace(m).re)&&contains(second,matrixNorm2(m));
 }
 function densityMatches(v:typeof x.joint,m:Matrix){const inv=invariants(m);return matrixMatches(v.density,m)&&v.rank===inv.positivePrincipalRank&&matches(v.purity,matrixNorm2(m))&&minorMatch(v.principalMinors,inv)&&qvec(v.characteristicPolynomial,inv.characteristic)&&spectrumMatches(v.spectrum,m,true)&&(v.rank!==1||matches(v.entropyNats,zero)&&matches(v.entropyBits,zero));}
 if(!matrixMatches(x.rawHermitianNumerator,H)||!matches(x.normalizationTrace,N)||!densityMatches(x.joint,rho)||!densityMatches(x.reducedA,A)||!densityMatches(x.reducedB,B))return false;
 const r=pauli.map(P=>expect(A,P)),s=pauli.map(P=>expect(B,P)),T=pauli.map(P=>pauli.map(Q=>expect(rho,tensor(P,Q))));
 if(!qvec(x.reducedA.blochVector,r)||!qvec(x.reducedB.blochVector,s)||!x.correlationTensor.every((row,i)=>qvec(row,T[i])))return false;
 const G=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>c(sum(T.map(row=>mul(row[i],row[j])))))),iG=invariants(G),threshold=sub(trace(G).re,one),K=matrixSub(G,matrixScale(identity(3),threshold)),iK=invariants(K),comparison=iK.psd?(iK.minors.at(-1)!.value.re.n===BigInt(0)?0:-1):1,cert=x.chsh.exactBoundaryCertificate;
 if(!matrixMatches(x.chsh.gramMatrix,G)||!qvec(x.chsh.gramCharacteristicPolynomial,iG.characteristic)||!spectrumMatches(x.chsh.gramSpectrum,G,true)||!matches(cert.threshold,threshold)||!matrixMatches(cert.shiftedMatrix,K)||!minorMatch(cert.principalMinors,iK)||cert.positiveSemidefinite!==iK.psd||!matches(cert.determinant,iK.minors.at(-1)!.value.re)||x.chsh.optimizedComparisonToLocalBound!==comparison||x.chsh.optimizedViolates!==(comparison===1)||x.chsh.optimizedAtBoundary!==(comparison===0))return false;
 const expanded=x.chsh.gramSpectrum.flatMap(v=>Array.from({length:v.multiplicity},()=>range(v.eigenvalue))),M=expanded.slice(-2).reduce(radd,point(zero));
 if(!overlap(range(x.chsh.topTwoEigenvalueSum),M)||!overlap(rmul(range(x.chsh.optimizedMaximum),range(x.chsh.optimizedMaximum)),rmul(point(int(4)),M))||!contains(rmul(range(x.chsh.tsirelsonBound),range(x.chsh.tsirelsonBound)),int(8))||!matches(x.chsh.localHiddenVariableBound,two))return false;
 const ppt=iPT.psd,negative=x.partialTranspose.spectrum.filter(v=>v.sign===-1),negativeMultiplicity=negative.reduce((n,v)=>n+v.multiplicity,0),negativity=negative.reduce((n,v)=>rsub(n,rmul(point(int(v.multiplicity)),range(v.eigenvalue))),point(zero));
 if(!matrixMatches(x.partialTranspose.matrix,PT)||!minorMatch(x.partialTranspose.principalMinors,iPT)||!qvec(x.partialTranspose.characteristicPolynomial,iPT.characteristic)||!spectrumMatches(x.partialTranspose.spectrum,PT,false)||x.partialTranspose.positiveSemidefinite!==ppt||x.partialTranspose.negativeMultiplicity!==negativeMultiplicity||negativeMultiplicity!==(ppt?0:1)||!overlap(range(x.partialTranspose.negativity),negativity)||ppt&&(!matches(x.partialTranspose.negativity,zero)||!matches(x.partialTranspose.logNegativityBits,zero)))return false;
 const ln2={lower:f(BigInt('6931471805599453094'),BigInt(10)**BigInt(19)),upper:f(BigInt('6931471805599453095'),BigInt(10)**BigInt(19))};
 for(const v of [x.joint,x.reducedA,x.reducedB])if(!overlap(range(v.entropyNats),rmul(range(v.entropyBits),ln2)))return false;
 const mutual=rsub(radd(range(x.reducedA.entropyNats),range(x.reducedB.entropyNats)),range(x.joint.entropyNats));
 if(!overlap(range(x.mutualInformationNats),mutual)||!overlap(range(x.mutualInformationNats),rmul(range(x.mutualInformationBits),ln2))||!overlap(range(x.conditionalEntropyAGivenBNats),rsub(range(x.joint.entropyNats),range(x.reducedB.entropyNats)))||!overlap(range(x.conditionalEntropyBGivenANats),rsub(range(x.joint.entropyNats),range(x.reducedA.entropyNats))))return false;
 const names=['a0','a1','b0','b1'],settings=names.map(name=>{const spinor=[0,1].map(i=>c(int(p[`${name}Real${i}`]),int(p[`${name}Imag${i}`]))),P=projector(spinor);return{name,P,observable:matrixSub(matrixScale(P,two),I),events:[P,matrixSub(I,P)]};});
 if(x.localTests.outcomeOrder.join(',')!=='1,-1'||!x.localTests.settings.every((v,i)=>v.name===names[i]&&matrixMatches(v.positiveProjector,settings[i].P)&&matrixMatches(v.observable,settings[i].observable)&&qvec(v.blochDirection,pauli.map(P=>expect(settings[i].P,P)))))return false;
 const correlations:F[][]=[];
 for(let i=0;i<2;i++){correlations.push([]);for(let j=0;j<2;j++){
  const a=settings[i],b=settings[j+2],v=x.localTests.pairs[i][j],E=expect(rho,tensor(a.observable,b.observable));correlations[i].push(E);
  if(v.settings.join(',')!==[a.name,b.name].join(',')||!v.jointProbabilities.every((row,u)=>row.every((p,w)=>matches(p,expect(rho,tensor(a.events[u],b.events[w])))))||!qvec(v.marginalA,a.events.map(P=>expect(A,P)))||!qvec(v.marginalB,b.events.map(P=>expect(B,P)))||!matches(v.correlation,E)||!matches(v.connected,sub(E,mul(expect(A,a.observable),expect(B,b.observable)))))return false;
 }}
 const selected=sub(add(add(correlations[0][0],correlations[0][1]),correlations[1][0]),correlations[1][1]),absolute=selected.n<BigInt(0)?neg(selected):selected,violates=sub(absolute,two).n>BigInt(0);
 if(!matches(x.chsh.selectedSigned,selected)||!matches(x.chsh.selectedAbsolute,absolute)||x.chsh.selectedViolates!==violates)return false;
 for(const[i,v]of x.localTests.conditionals.entries()){
  const onA=i<2,prior=onA?B:A,setting=settings[i];if(v.setting!==setting.name||v.conditionedSubsystem!==(onA?'b':'a')||!matrixMatches(v.nonselectiveDensity,prior))return false;
  for(const[j,row]of v.rows.entries()){
   const K=onA?tensor(setting.events[j],I):tensor(I,setting.events[j]),global=product(product(K,rho),K),local=partial(global,2,2,onA?'b':'a'),probability=trace(global).re,none=probability.n===BigInt(0);
   if(row.event!==j||!matches(row.probability,probability)||!matrixMatches(row.unnormalizedDensity,local)||(row.conditionalDensity===null)!==none||(row.conditionalPurity===null)!==none)return false;
   if(!none){const conditional=matrixScale(local,div(one,probability));if(!matrixMatches(row.conditionalDensity!,conditional)||!matches(row.conditionalPurity!,matrixNorm2(conditional)))return false;}
  }
 }
 const cls=x.classification;return cls.globallyPure===(x.joint.rank===1)&&cls.separableTwoQubit===ppt&&cls.entangledTwoQubit===!ppt&&cls.localAIsPure===(x.reducedA.rank===1)&&cls.localBIsPure===(x.reducedB.rank===1)&&cls.selectedChshViolation===violates&&cls.optimizedChshViolation===(comparison===1);
}
