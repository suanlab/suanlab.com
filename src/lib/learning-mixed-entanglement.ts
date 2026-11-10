/** Arbitrary bounded two-qubit mixed states: PPT, local statistics and CHSH. */
import {snapshotMixedEntanglementParameters,type MixedEntanglementParameters} from './learning-mixed-entanglement-input';
import {mixedInputMatrix,mixedHermitianInvariants} from './learning-mixed-entanglement-algebra';
import {isolateTensorSpectrum} from './learning-tensor-roots';
import {createQuantumEvolutionContext,type EvolutionInterval} from './learning-quantum-evolution-interval';
import {createTranscendentalContext} from './learning-interval-transcendentals';
import {f,c,zero,one,two,add,sub,mul,div,neg,sum,ca,cm,q,cq,type F} from './learning-quantum-hilbert-core';
import {identity,projector,product,trace,matrixAdd,matrixSub,matrixScale,matrixObservation,matrixSquaredNorm,type Matrix} from './learning-observable-matrix';
export {isMixedEntanglementParameters,snapshotMixedEntanglementParameters,type MixedEntanglementParameters} from './learning-mixed-entanglement-input';
const tensor=(a:Matrix,b:Matrix):Matrix=>a.flatMap(row=>b.map(other=>row.flatMap(v=>other.map(w=>cm(v,w)))));
const partial=(rho:Matrix,keep:'a'|'b'):Matrix=>Array.from({length:2},(_,i)=>Array.from({length:2},(_,j)=>[0,1].map(k=>keep==='a'?rho[i*2+k][j*2+k]:rho[k*2+i][k*2+j]).reduce(ca,c(zero))));
const pauli:Matrix[]=[[[c(zero),c(one)],[c(one),c(zero)]],[[c(zero),c(zero,neg(one))],[c(zero,one),c(zero)]],[[c(one),c(zero)],[c(zero),c(neg(one))]]];
export function mixedEntanglementModel(value:MixedEntanglementParameters) {
 const parameters=snapshotMixedEntanglementParameters(value),H=mixedInputMatrix(parameters),normalizer=trace(H).re,rho=matrixScale(H,div(one,normalizer)),A=partial(rho,'a'),B=partial(rho,'b'),I=identity(2),I4=identity(4);
 const pt=rho.map((row,u)=>row.map((_,v)=>rho[Math.floor(u/2)*2+v%2][Math.floor(v/2)*2+u%2]));
 const inv=mixedHermitianInvariants(rho),ptInv=mixedHermitianInvariants(pt),ppt=ptInv.positiveSemidefinite;
 const expect=(state:Matrix,observable:Matrix)=>{const v=trace(product(state,observable));if(v.im.n!==BigInt(0))throw new RangeError('Real expectation required');return v.re;};
 const blochA=pauli.map(P=>expect(A,P)),blochB=pauli.map(P=>expect(B,P)),T=pauli.map(P=>pauli.map(Q=>expect(rho,tensor(P,Q))));
 const gram:Matrix=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>c(sum(T.map(row=>mul(row[i],row[j])))))),gramInv=mixedHermitianInvariants(gram);
 const ctx=createQuantumEvolutionContext(),logs=createTranscendentalContext(768),integer=(n:number)=>f(BigInt(n));
 function logPoint(v:F){const l=logs.logRatio({coefficient:v.n,exponent:0},{coefficient:v.d,exponent:0});return{lower:f(l.lower,logs.scale),upper:f(l.upper,logs.scale)};}
 function logarithm(v:EvolutionInterval){const r=ctx.rationalRange(v);if(r.lower.n<=BigInt(0))throw new RangeError('Positive logarithm argument required');return ctx.range({lower:logPoint(r.lower).lower,upper:logPoint(r.upper).upper});}
 function root(v:EvolutionInterval){const r=ctx.rationalRange(v);if(r.lower.n<BigInt(0))throw new RangeError('Nonnegative square root required');return{lower:ctx.sqrt(r.lower).lower,upper:ctx.sqrt(r.upper).upper};}
 const ln2=ctx.range(logPoint(two));
 function spectrum(matrix:Matrix,positive:boolean) {
  const invariants=mixedHermitianInvariants(matrix),isolated=isolateTensorSpectrum(invariants.characteristicPolynomial);
  const roots=isolated.roots.map(r=>{
   const sign=r.lower.n===BigInt(0)&&r.upper.n===BigInt(0)?0:r.upper.n<BigInt(0)?-1:r.lower.n>BigInt(0)?1:null;
   if(sign===null||positive&&sign<0)throw new RangeError('Unresolved or invalid spectral sign');
   return{...r,sign,interval:ctx.range(r)};
  });
  return {invariants,roots,rows:roots.map(r=>({eigenvalue:r.exact?q(r.lower):ctx.observation(r.interval),multiplicity:r.multiplicity,sign:r.sign,isolatingLower:q(r.lower),isolatingUpper:q(r.upper),exactRationalRootKnown:r.exact})),
   expanded:roots.flatMap(r=>Array.from({length:r.multiplicity},()=>r.interval))};
 }
 function density(state:Matrix){const s=spectrum(state,true),rank=s.invariants.positivePrincipalRank;
  const entropy=rank===1?ctx.zero:s.roots.filter(r=>r.sign===1).reduce((total,r)=>ctx.sub(total,ctx.multiply(ctx.from(integer(r.multiplicity)),ctx.multiply(r.interval,logarithm(r.interval)))),ctx.zero);
  return {spectral:s,entropy,observation:{density:matrixObservation(state),rank,purity:q(matrixSquaredNorm(state)),principalMinors:s.invariants.principalMinors.map(m=>({indices:m.indices,determinant:q(m.determinant)})),characteristicPolynomial:s.invariants.characteristicPolynomial.map(q),spectrum:s.rows,entropyNats:ctx.observation(entropy),entropyBits:ctx.observation(ctx.divide(entropy,ln2))}};
 }
 const joint=density(rho),localA=density(A),localB=density(B),ptSpectrum=spectrum(pt,false),gramSpectrum=spectrum(gram,true);
 const negativity=ppt?ctx.zero:ptSpectrum.roots.filter(r=>r.sign===-1).reduce((total,r)=>ctx.sub(total,ctx.multiply(r.interval,ctx.from(integer(r.multiplicity)))),ctx.zero),logNegativity=ppt?ctx.zero:ctx.divide(logarithm(ctx.add(ctx.one,ctx.multiply(ctx.from(two),negativity))),ln2);
 if(ppt!==ptSpectrum.roots.every(r=>r.sign>=0))throw new RangeError('PPT principal-minor and spectral certificates disagree');
 // M=sum of the two largest eigenvalues of T^T T. Compare exactly through
 // M>1 iff lambda_min(G)<trace(G)-1 iff G-(trace(G)-1)I is not PSD.
 // PSD plus zero determinant certifies equality, including repeated roots.
 const threshold=sub(trace(gram).re,one),shifted=matrixSub(gram,matrixScale(identity(3),threshold)),boundary=mixedHermitianInvariants(shifted),optimalComparison=boundary.positiveSemidefinite?(boundary.determinant.n===BigInt(0)?0:-1):1;
 const topTwo=gramSpectrum.expanded.slice(-2),M=topTwo.reduce(ctx.add,ctx.zero),maximum=ctx.multiply(ctx.from(two),root(M)),tsirelson=ctx.multiply(ctx.from(two),ctx.sqrt(two));
 function setting(name:string){const spinor=[0,1].map(i=>c(integer(parameters[`${name}Real${i}`]),integer(parameters[`${name}Imag${i}`]))),P=projector(spinor),events=[P,matrixSub(I,P)],observable=matrixSub(matrixScale(P,two),I),direction=pauli.map(Q=>expect(P,Q));return{name,P,events,observable,direction};}
 const settingsA=[setting('a0'),setting('a1')],settingsB=[setting('b0'),setting('b1')];
 const settings=settingsA.map(a=>settingsB.map(b=>{
  const jointProbabilities=a.events.map(P=>b.events.map(Q=>expect(rho,tensor(P,Q)))),marginalA=a.events.map(P=>expect(A,P)),marginalB=b.events.map(Q=>expect(B,Q)),correlation=expect(rho,tensor(a.observable,b.observable));
  if(jointProbabilities.flat().some(v=>v.n<BigInt(0)))throw new RangeError('Nonnegative joint probabilities required');
  const weighted=sum(jointProbabilities.flatMap((row,i)=>row.map((v,j)=>i===j?v:neg(v))));
  return{settings:[a.name,b.name],jointProbabilities:jointProbabilities.map(row=>row.map(q)),marginalA:marginalA.map(q),marginalB:marginalB.map(q),correlation:q(correlation),connected:q(sub(correlation,mul(expect(A,a.observable),expect(B,b.observable)))),exactCorrelation:correlation,
   residuals:{mass:q(sub(sum(jointProbabilities.flat()),one)),marginalA:jointProbabilities.map((row,i)=>q(sub(sum(row),marginalA[i]))),marginalB:[0,1].map(j=>q(sub(sum(jointProbabilities.map(row=>row[j])),marginalB[j]))),correlation:q(sub(weighted,correlation))}};
 }));
 const selected= sub(add(add(settings[0][0].exactCorrelation,settings[0][1].exactCorrelation),settings[1][0].exactCorrelation),settings[1][1].exactCorrelation),absoluteSelected=selected.n<BigInt(0)?neg(selected):selected,selectedViolates=sub(absoluteSelected,two).n>BigInt(0);
 function condition(s:ReturnType<typeof setting>,on:'a'|'b'){
  const prior=on==='a'?B:A,keep=on==='a'?'b':'a';
  const rows=s.events.map((P,event)=>{
   const K=on==='a'?tensor(P,I):tensor(I,P),global=product(product(K,rho),K),reduced=partial(global,keep),probability=trace(global).re,conditional=probability.n===BigInt(0)?null:matrixScale(reduced,div(one,probability));
   return {event,probability:q(probability),unnormalizedDensity:matrixObservation(reduced),conditionalDensity:conditional?matrixObservation(conditional):null,conditionalPurity:conditional?q(matrixSquaredNorm(conditional)):null,exactReduced:reduced};
  });
  const nonselective=matrixAdd(rows[0].exactReduced,rows[1].exactReduced);
  return{setting:s.name,conditionedSubsystem:keep,rows:rows.map(row=>({event:row.event,probability:row.probability,unnormalizedDensity:row.unnormalizedDensity,conditionalDensity:row.conditionalDensity,conditionalPurity:row.conditionalPurity})),nonselectiveDensity:matrixObservation(nonselective),noSignallingResidual:matrixObservation(matrixSub(nonselective,prior))};
 }
 // Reconstruct rho from its full Pauli expansion to expose Y-sign/index errors.
 let reconstructed=I4;
 for(let i=0;i<3;i++){
  reconstructed=matrixAdd(reconstructed,matrixScale(tensor(pauli[i],I),blochA[i]));
  reconstructed=matrixAdd(reconstructed,matrixScale(tensor(I,pauli[i]),blochB[i]));
  for(let j=0;j<3;j++)reconstructed=matrixAdd(reconstructed,matrixScale(tensor(pauli[i],pauli[j]),T[i][j]));
 }
 reconstructed=matrixScale(reconstructed,f(BigInt(1),BigInt(4)));
 const mutual=ctx.sub(ctx.add(localA.entropy,localB.entropy),joint.entropy);
 return{parameters,dimensionA:2,dimensionB:2,jointDimension:4,flatteningConvention:'A tensor B; flatIndex=2*aIndex+bIndex',precisionBits:512,arithmeticWorkBits:768,rootIsolationBits:1024,
  rawHermitianNumerator:matrixObservation(H),normalizationTrace:q(normalizer),joint:joint.observation,reducedA:{...localA.observation,blochVector:blochA.map(q)},reducedB:{...localB.observation,blochVector:blochB.map(q)},
  mutualInformationNats:ctx.observation(mutual),mutualInformationBits:ctx.observation(ctx.divide(mutual,ln2)),conditionalEntropyAGivenBNats:ctx.observation(ctx.sub(joint.entropy,localB.entropy)),conditionalEntropyBGivenANats:ctx.observation(ctx.sub(joint.entropy,localA.entropy)),
  pauliConvention:'X,Y=[[0,-i],[i,0]],Z; tensor order A,B',correlationTensor:T.map(row=>row.map(q)),partialTranspose:{on:'B',matrix:matrixObservation(pt),principalMinors:ptInv.principalMinors.map(m=>({indices:m.indices,determinant:q(m.determinant)})),characteristicPolynomial:ptInv.characteristicPolynomial.map(q),spectrum:ptSpectrum.rows,positiveSemidefinite:ppt,negativeMultiplicity:ptSpectrum.roots.filter(r=>r.sign===-1).reduce((n,r)=>n+r.multiplicity,0),negativity:ctx.observation(negativity),logNegativityBits:ctx.observation(logNegativity)},
  localTests:{outcomeOrder:[1,-1],settings:[...settingsA,...settingsB].map(s=>({name:s.name,positiveProjector:matrixObservation(s.P),observable:matrixObservation(s.observable),blochDirection:s.direction.map(q)})),pairs:settings.map(row=>row.map(s=>({settings:s.settings,jointProbabilities:s.jointProbabilities,marginalA:s.marginalA,marginalB:s.marginalB,correlation:s.correlation,connected:s.connected,residuals:s.residuals}))),conditionals:[...settingsA.map(s=>condition(s,'a')),...settingsB.map(s=>condition(s,'b'))]},
  chsh:{expression:'E00+E01+E10-E11',selectedSigned:q(selected),selectedAbsolute:q(absoluteSelected),selectedViolates,localHiddenVariableBound:q(two),gramMatrix:matrixObservation(gram),gramCharacteristicPolynomial:gramInv.characteristicPolynomial.map(q),gramSpectrum:gramSpectrum.rows,topTwoEigenvalueSum:ctx.observation(M),optimizedMaximum:ctx.observation(maximum),tsirelsonBound:ctx.observation(tsirelson),optimizedComparisonToLocalBound:optimalComparison,optimizedViolates:optimalComparison===1,optimizedAtBoundary:optimalComparison===0,optimalAxesComputed:false,
   exactBoundaryCertificate:{threshold:q(threshold),shiftedMatrix:matrixObservation(shifted),principalMinors:boundary.principalMinors.map(m=>({indices:m.indices,determinant:q(m.determinant)})),positiveSemidefinite:boundary.positiveSemidefinite,determinant:q(boundary.determinant)}},
  exactResiduals:{trace:cq({re:sub(trace(rho).re,one),im:trace(rho).im}),partialTransposeTrace:q(sub(trace(pt).re,one)),partialTransposeSquaredNorm:q(sub(matrixSquaredNorm(pt),matrixSquaredNorm(rho))),pauliReconstruction:matrixObservation(matrixSub(reconstructed,rho)),reducedTraceA:q(sub(trace(A).re,one)),reducedTraceB:q(sub(trace(B).re,one))},
  classification:{globallyPure:inv.positivePrincipalRank===1,separableTwoQubit:ppt,entangledTwoQubit:!ppt,localAIsPure:localA.observation.rank===1,localBIsPure:localB.observation.rank===1,selectedChshViolation:selectedViolates,optimizedChshViolation:optimalComparison===1,noChshViolationProvesLocalHiddenVariableModel:false,pptSufficiencyDimension:'2x2 only in this model',finiteSampleExperimentSupplied:false,physicalTimeSpecified:false},
  approximationContract:'32 owned bounded integers of magnitude<=1e6:general Hermitian4x4 numerator(10 real upper-triangle and6 imaginary strict-upper entries),positive trace and all15 nonnegative principal minors;four nonzero complex2-spinor local tests. Normalize by trace without clipping or PSD repair. Exact rational partial traces,Pauli tensor,selected joint probabilities,conditional/null states,no-signalling and PPT certificates. PPT iff separable is used only for2x2. Exact square-free/Sturm spectra at1024bits;768-bit work/outward512-bit entropy and negativity. Optimized CHSH is over local traceless±1 qubit projective observables;exact shifted-Gram principal minors distinguish M<1,=1,>1 without numerical tolerance. Optimal axes are not computed. Failure to violate CHSH does not establish a general local hidden-variable model;no finite-sample data,filtering or postselection is modeled.'};
}
