/** General Hermitian Rayleigh energy: exact moments and certified spectral subspaces. */
import {snapshotRayleighParameters, type RayleighParameters} from './learning-rayleigh-input';
import {variationalCharacteristic} from './learning-quantum-variational-math';
import {isolateTensorSpectrum} from './learning-tensor-roots';
import {createQuantumEvolutionContext} from './learning-quantum-evolution-interval';
import {f,c,zero,one,two,add,sub,mul,div,neg,le,eq,cs,scaleC,dot,norm2,q,cq,normalized,type C,type R} from './learning-quantum-hilbert-core';
import {identity,apply,matrixSub,matrixScale,matrixObservation,type Matrix} from './learning-observable-matrix';
export {isRayleighParameters,snapshotRayleighParameters,rayleighKeys,type RayleighParameters} from './learning-rayleigh-input';
export function rayleighModel(value: RayleighParameters) {
  const parameters = snapshotRayleighParameters(value), p = parameters, d = p.dimension, int = (n: number) => f(BigInt(n));
  const H: Matrix = Array.from({length:d}, (_, i) => Array.from({length:d}, (_, j) => {
    const a=Math.min(i,j), b=Math.max(i,j);
    return c(div(int(p[`hReal${a}${b}`]),int(p.hDenominator)),i===j?zero:div(int(p[`hImag${a}${b}`]*(i<j?1:-1)),int(p.hDenominator)));
  }));
  const vector = (prefix:string) => Array.from({length:d},(_,i)=>c(int(p[`${prefix}Real${i}`]),int(p[`${prefix}Imag${i}`]))), z=vector('state'), eta=vector('direction'), I=identity(d);
  function moments(state:C[]) {
    const N=norm2(state);if(N.n===BigInt(0))return null;
    const Hz=apply(H,state), mean=scaleC(dot(state,Hz),div(one,N)), second=div(norm2(Hz),N), E=mean.re, residual=Hz.map((v,i)=>cs(v,scaleC(state[i],E))), variance=div(norm2(residual),N);
    if(mean.im.n!==BigInt(0)||variance.n<BigInt(0)||!eq(variance,sub(second,mul(E,E))))throw new RangeError('Hermitian moment certificates disagree');
    return {N,E,second,residual,variance,orthogonality:dot(state,residual)};
  }
  const m=moments(z)!, centered=matrixSub(H,matrixScale(I,m.E)), first=div(mul(two,dot(eta,m.residual).re),m.N), normDerivative=mul(two,dot(z,eta).re), second= sub(div(mul(two,dot(eta,apply(centered,eta)).re),m.N),div(mul(mul(two,first),normDerivative),m.N));
  const t=f(BigInt(p.pathNumerator),BigInt(p.pathDenominator)), path=z.map((v,i)=>({re:add(v.re,mul(t,eta[i].re)),im:add(v.im,mul(t,eta[i].im))})), pathMoments=moments(path), characteristic=variationalCharacteristic(H), roots=isolateTensorSpectrum(characteristic), ctx=createQuantumEvolutionContext(), toInterval=(r:R)=>ctx.range(r), matrix=H.map(row=>row.map(ctx.complex)), iz=z.map(ctx.complex), n=ctx.from(m.N), isEigenstate=m.variance.n===BigInt(0);
  const eigenIndex=isEigenstate?roots.roots.findIndex(r=>le(r.lower,m.E)&&le(m.E,r.upper)):-1;
  if(isEigenstate&&eigenIndex<0)throw new RangeError('Exact eigenstate must lie in the certified spectrum');
  const spectra=roots.roots.map((root,index)=>{
    const lambda=toInterval(root);
    let P=ctx.identity(d);
    for(const [j,other]of roots.roots.entries())if(j!==index){
      const mu=toInterval(other),gap=ctx.sub(lambda,mu),shifted=matrix.map((row,i)=>row.map((v,k)=>i===k?ctx.complexSubtract(v,{real:mu,imaginary:ctx.zero}):v));
      P=ctx.product(P,shifted).map(row=>row.map(v=>ctx.complexScale(v,ctx.divide(ctx.one,gap))));
    }
    const image=ctx.apply(P,iz), rawWeight=ctx.divide(ctx.dot(iz,image).real,n), weight=isEigenstate?(index===eigenIndex?ctx.one:ctx.zero):rawWeight;
    return {root,lambda,weight,projector:P,image};
  });
  const ground=spectra[0],gap=spectra.length===1?null:ctx.sub(spectra[1].lambda,ground.lambda),excess=isEigenstate&&eigenIndex===0?ctx.zero:ctx.sub(ctx.from(m.E),ground.lambda),leak=isEigenstate?(eigenIndex===0?ctx.zero:ctx.one):ctx.sub(ctx.one,ground.weight),bound=gap===null?null:ctx.divide(excess,gap),boundRange=bound===null?null:ctx.rationalRange(bound),leakageUpperBound=boundRange===null?null:le(boundRange.upper,one)?boundRange.upper:one;
  const sumProjectors=spectra.reduce((sum,v)=>sum.map((row,i)=>row.map((a,j)=>ctx.complexAdd(a,v.projector[i][j]))),ctx.identity(d).map(row=>row.map(()=>ctx.complex(c(zero)))));
  const sumWeight=spectra.reduce((s,v)=>ctx.add(s,v.weight),ctx.zero),sumEnergy=spectra.reduce((s,v)=>ctx.add(s,ctx.multiply(v.lambda,v.weight)),ctx.zero),sumSecond=spectra.reduce((s,v)=>ctx.add(s,ctx.multiply(ctx.square(v.lambda),v.weight)),ctx.zero);
  return {
    parameters,dimension:d,precisionBits:512,arithmeticWorkBits:768,rootIsolationBits:1024,units:'User-supplied common energy unit; state and direction have a shared amplitude unit',
    hamiltonian:matrixObservation(H),state:z.map(cq),normalizedState:z.map(v=>normalized(v,m.N)),direction:eta.map(cq),rawNormSquared:q(m.N),energy:q(m.E),secondMoment:q(m.second),variance:q(m.variance),residualNumerators:m.residual.map(cq),normalizedResidual:m.residual.map(v=>normalized(v,m.N)),residualNorm:ctx.observation(ctx.sqrt(m.variance)),centeredHamiltonian:matrixObservation(centered),
    variation:{realParameter:true,firstDerivative:q(first),secondDerivative:q(second),normFirstDerivative:q(normDerivative),stationaryInAllComplexDirections:isEigenstate},
    path:{parameter:q(t),rawState:path.map(cq),defined:pathMoments!==null,moments:pathMoments===null?null:{rawNormSquared:q(pathMoments.N),energy:q(pathMoments.E),secondMoment:q(pathMoments.second),variance:q(pathMoments.variance)}},
    characteristicPolynomial:characteristic.map(q),squareFreeFactors:roots.squareFreeFactors.map(v=>({multiplicity:v.multiplicity,ascending:v.ascending.map(q)})),
    spectrum:spectra.map(v=>({energy:ctx.observation(v.lambda),lower:q(v.root.lower),upper:q(v.root.upper),multiplicity:v.root.multiplicity,exact:v.root.exact,projector:ctx.observeMatrix(v.projector),projectedRawState:v.image.map(ctx.observeComplex),probability:ctx.probability(v.weight)})),
    ground:{multiplicity:ground.root.multiplicity,energy:ctx.observation(ground.lambda),overlap:ctx.probability(ground.weight),leakage:ctx.probability(leak),excessEnergy:ctx.observation(excess),gap:gap===null?null:ctx.observation(gap),energyOverGap:bound===null?null:ctx.observation(bound),certifiedLeakageUpperBound:leakageUpperBound===null?null:q(leakageUpperBound)},
    exactResiduals:{orthogonality:cq(m.orthogonality),varianceIdentity:q(sub(m.variance,sub(m.second,mul(m.E,m.E)))),scaleDirectionDerivative:q(div(mul(two,dot(z,m.residual).re),m.N)),phaseDirectionDerivative:q(div(mul(two,dot(z.map(v=>c(neg(v.im),v.re)),m.residual).re),m.N))},
    directedResiduals:{probabilityMass:ctx.observation(ctx.sub(sumWeight,ctx.one)),energyAccount:ctx.observation(ctx.sub(sumEnergy,ctx.from(m.E))),secondMomentAccount:ctx.observation(ctx.sub(sumSecond,ctx.from(m.second))),projectorCompleteness:ctx.observeMatrix(ctx.matrixSubtract(sumProjectors,ctx.identity(d))),spectralSubspaces:spectra.map(v=>({hermiticity:ctx.observeMatrix(ctx.matrixSubtract(v.projector,ctx.adjoint(v.projector))),idempotence:ctx.observeMatrix(ctx.matrixSubtract(ctx.product(v.projector,v.projector),v.projector)),eigenvalueEquation:ctx.observeMatrix(ctx.matrixSubtract(ctx.product(matrix,v.projector),v.projector.map(row=>row.map(z=>ctx.complexScale(z,v.lambda))))),traceRank:ctx.observation(ctx.sub(v.projector.reduce((s,row,i)=>ctx.add(s,row[i].real),ctx.zero),ctx.from(int(v.root.multiplicity))))}))},
    classification:{exactEigenstate:isEigenstate,groundEigenstate:isEigenstate&&eigenIndex===0,excitedEigenstate:isEigenstate&&eigenIndex>0,groundDegenerate:ground.root.multiplicity>1,constantHamiltonian:spectra.length===1,nonzeroDirection:norm2(eta).n!==BigInt(0),pathZeroVector:pathMoments===null,spectrumEigenvectorsComputed:false,physicalTimeSpecified:false},
    approximationContract:'General Hermitian d=1..4 with36 owned bounded integer fields and positive common energy/path denominators. Exact complex-rational normalization,moments,residual and directional derivatives. Exact characteristic and square-free/Sturm roots at1024bits; spectral subspace projectors and weights use outward768-bit work/512-bit observations. Distinct eigenvalues are never merged by tolerance. Probabilities intersect proven[0,1] bounds,without state renormalization. Degeneracies identify whole subspaces,not a preferred eigenvector. Zero path state is undefined/null; a constant Hamiltonian has no positive excitation gap. Zero variance alone never certifies the ground state. No continuous Hamiltonian,ansatz optimizer,time evolution,measurement sampling or physical convergence theorem is supplied.',
  };
}
