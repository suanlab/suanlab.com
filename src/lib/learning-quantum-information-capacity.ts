/** Dephasing-family capacities by theorem, plus an explicitly selected input's information. */
import { binaryQubitDiscrimination } from './learning-quantum-information-discrimination';
import { f, zero, one, two, add, sub, mul, div, eq, le, q, observe, sqrtRange, type F } from './learning-quantum-hilbert-core';
import { point, plus, minus, scale, intersect, binaryEntropyRange, qubitEntropyRange, dot } from './learning-quantum-information-entropies';
type Bloch=readonly[F,F,F];
export function qubitDephasingCapacity(probability:F,input:Bloch){
 const p=probability;
 if(!p||typeof p.n!=='bigint'||typeof p.d!=='bigint'||p.d<=BigInt(0)||p.n<BigInt(0)||p.n>p.d||p.n.toString().length>32||p.d.toString().length>32)throw new RangeError('A rational phase-flip probability in[0,1] with at most32 digits is required');
 const validated=binaryQubitDiscrimination(input,input,div(one,two));
 const complement=sub(one,p),attenuation=sub(one,mul(two,p));
 const output:Bloch=[mul(attenuation,input[0]),mul(attenuation,input[1]),input[2]];
 const h=binaryEntropyRange(point(p)),quantum=intersect(minus(point(one),h)),assisted=minus(point(two),h);
 const inputEntropy=qubitEntropyRange(input),outputEntropy=qubitEntropyRange(output);
 const envRadiusSquared=add(mul(attenuation,attenuation),mul(f(BigInt(4)),mul(mul(p,complement),mul(input[2],input[2]))));
 const envRadius=sqrtRange(envRadiusSquared);
 const envSmallEigenvalue=intersect(scale(minus(point(one),envRadius),div(one,two)),zero,div(one,two));
 const environmentEntropy=binaryEntropyRange(envSmallEigenvalue);
 const rawCoherent=minus(outputEntropy,environmentEntropy),pureInput=eq(dot(input,input),one),completeDephasing=eq(mul(two,p),one);
 const coherent=pureInput||completeDephasing?point(zero):rawCoherent;
 const mutual=plus(inputEntropy,coherent),half=(x:F)=>div(x,two);
 const complex=(re:ReturnType<typeof q>,im=q(zero))=>({real:re,imaginary:im});
 const outputDensity=[
  [complex(q(half(add(one,output[2])))),complex(q(half(output[0])),q(sub(zero,half(output[1]))))],
  [complex(q(half(output[0])),q(half(output[1]))),complex(q(half(sub(one,output[2]))))],
 ];
 const envCoherence=scale(sqrtRange(mul(p,complement)),input[2]);
 const envDensity=[
  [complex(q(complement)),complex(observe(envCoherence))],
  [complex(observe(envCoherence)),complex(q(p))],
 ];
 const choi=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>complex(q(i===j&&(i===0||i===3)?half(one):i!==j&&(i===0||i===3)&&(j===0||j===3)?half(attenuation):zero))));
 return {phaseFlipProbability:q(p),inputBloch:input.map(q),inputDensity:validated.density0,outputBloch:output.map(q),outputDensity,
  environmentDensity:envDensity,environmentSmallerEigenvalue:observe(envSmallEigenvalue),normalizedChoi:choi,
  inputEntropy:observe(inputEntropy),outputEntropy:observe(outputEntropy),environmentEntropy:observe(environmentEntropy),
  selectedInputCoherentInformation:observe(coherent),rawSelectedInputCoherentInformation:observe(rawCoherent),selectedInputReferenceMutualInformation:observe(mutual),
  capacitiesByFamilyTheorem:{classicalBitsPerUse:q(one),entanglementAssistedClassicalBitsPerUse:observe(assisted),quantumQubitsPerUse:observe(quantum)},
  quantumCapacityStrictlyPositive:!completeDephasing,phaseErrorEntropy:observe(h),
  zBasisMessageInformation:q(one),xBasisMessageInformation:observe(quantum),xBasisEqualPriorGuessSuccess:q(le(p,complement)?complement:p),
  assumptions:['Known fixed memoryless qubit phase-flip channel (1-p)rho+p ZrhoZ','C and Q are unassisted asymptotic vanishing-error rates; CE allows preshared entanglement','Degradability and Pauli symmetry justify the single-letter quantum optimum at I/2 for this family','The channel mutual-information theorem and symmetry justify CE for this family; no finite-block encoder is supplied'],
  contract:'Exact rational physicality and channel/Choi data; directed512-bit spectral roots and256-bit entropy logarithms in bits. Capacity values use theorems specific to the dephasing family, not a grid search or the selected input. Exact pure-input/complete-dephasing coherent-information zeros are separated from raw intervals. No generic capacity solver, private/two-way-assisted capacity, finite-block guarantee, Worker/public input reader/UI.'};
}
