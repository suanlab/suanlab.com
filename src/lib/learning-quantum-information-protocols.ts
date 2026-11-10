/** Exact ideal teleportation, dense coding and CNOT-copy counterexamples.
 * Raw vectors are normalized only as densities; amplitudes keep their scale. */
import { zero, one } from './learning-quantum-hilbert-core';
import { runQubitCircuit, type CircuitGate } from './learning-qubit-circuit-exact';
import { quadratic, circuitNormSquared, type CircuitScalar as C } from './learning-qubit-scalar';
import { Z,O,complex,times,conjugate,divide,qDiv,qSign,qEqual,mWire,circuitScalarWire,densityFromVector,partialTrace,tensor,product,trace,mEqual,type M,type Q } from './learning-qubit-channel-algebra';
export function requireProtocolVector(input:readonly C[]) {
  if(!Array.isArray(input)||input.length!==2)throw new RangeError('Two exact complex amplitudes required');
  const pair=(p:Q)=>{
    if(!Array.isArray(p)||p.length!==2)throw new RangeError('Two rational coefficients required');
    for(const x of Array.from(p))if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.d<=BigInt(0)||x.n.toString().replace('-','').length>32||x.d.toString().length>32)throw new RangeError('Coefficients require at most32 digits and positive denominators');
  };
  for(const z of Array.from(input)){if(!z)throw new RangeError('Dense amplitude array required');pair(z.re);pair(z.im);}
  if(qSign(circuitNormSquared(input))<=0)throw new RangeError('Nonzero state required');
}
const wire=(state:readonly C[])=>state.map(circuitScalarWire);
const density=(state:readonly C[]):M=>{
  const norm=circuitNormSquared(state);
  return state.map(a=>state.map(b=>divide(times(a,conjugate(b)),norm)));
};
const qWire=(x:Q)=>circuitScalarWire(complex(x)).real;
const basis=(sites:number,index:number)=>Array.from({length:2**sites},(_,j)=>j===index?O:Z);
export function teleportQubit(input:readonly C[]) {
  requireProtocolVector(input);
  const initial=Array.from({length:8},(_,j)=>j===0?input[0]:j===4?input[1]:Z);
  const gates:CircuitGate[]=[{gate:'H',target:1},{gate:'CX',control:1,target:2},{gate:'CX',control:0,target:1},{gate:'H',target:0}];
  const circuit=runQubitCircuit(3,gates,initial),norm=circuitNormSquared(input);
  const branches=Array.from({length:4},(_,label)=>{
    const a=Math.floor(label/2),b=label%2,raw=circuit.state.slice(label*2,label*2+2);
    const corrections:CircuitGate[]=[...(b?[{gate:'X' as const,target:0}]:[]),...(a?[{gate:'Z' as const,target:0}]:[])];
    const corrected=runQubitCircuit(1,corrections,raw).state;
    return {a,b,probability:qWire(qDiv(circuitNormSquared(raw),norm)),rawBob:wire(raw),correctedBob:wire(corrected),
      bobDensity:mWire(densityFromVector(raw)),correctedDensity:mWire(densityFromVector(corrected)),correctionGates:corrections};
  });
  return {siteOrder:['Alice input','Alice Bell half','Bob Bell half'],input:wire(input),inputNormSquared:qWire(norm),
    prefixes:circuit.prefixes.map(wire),gates,branches,inputDensity:mWire(densityFromVector(Array.from(input))),
    bobBeforeMessage:mWire(partialTrace(density(circuit.state),4,2,'second')),
    resources:{sharedEbits:1,classicalBitsSent:2,quantumInputCopies:1},
    contract:'Exact ideal three-qubit circuit. Branch vectors are unnormalized and retain input scale. Every branch has probability1/4; corrections execute X^b then Z^a. No finite-noise fidelity, implementation security, Worker, public input reader or UI.'};
}
export function denseCoding(a:number,b:number,order:'x-then-z'|'z-then-x'='x-then-z') {
  if(!Number.isInteger(a)||!Number.isInteger(b)||![0,1].includes(a)||![0,1].includes(b)||!['x-then-z','z-then-x'].includes(order))throw new RangeError('Two message bits and a valid Pauli order required');
  const preparation:CircuitGate[]=[{gate:'H',target:0},{gate:'CX',control:0,target:1}];
  // Lists are chronological; reversing X/Z changes a global sign only when both bits are1.
  const encoding:CircuitGate[]=order==='x-then-z'?[...(b?[{gate:'X' as const,target:0}]:[]),...(a?[{gate:'Z' as const,target:0}]:[])]
    :[...(a?[{gate:'Z' as const,target:0}]:[]),...(b?[{gate:'X' as const,target:0}]:[])];
  const decoding:CircuitGate[]=[{gate:'CX',control:0,target:1},{gate:'H',target:0}];
  const result=runQubitCircuit(2,[...preparation,...encoding,...decoding],basis(2,0));
  const encoded=result.prefixes[preparation.length+encoding.length];
  return {message:[a,b],order,encoding,decodedState:wire(result.state),encodedState:wire(encoded),
    bobBeforeTransmission:mWire(partialTrace(density(encoded),2,2,'second')),
    decodedProbabilities:result.state.map(z=>qWire(circuitNormSquared([z]))),
    decodedDensity:mWire(density(result.state)),decodedIndex:result.state.findIndex(z=>qSign(circuitNormSquared([z]))>0),prefixes:result.prefixes.map(wire),
    resources:{sharedEbits:1,qubitsSent:1,classicalMessageBits:2},contract:'Ideal shared Bell pair, one transmitted qubit, joint receiver decoding. Raw global phase is retained. No unassisted one-qubit two-bit capacity claim.'};
}
export function cnotCopyCounterexample(input:readonly C[]) {
  requireProtocolVector(input);
  const initial=[input[0],Z,input[1],Z],result=runQubitCircuit(2,[{gate:'CX',control:0,target:1}],initial);
  const joint=density(result.state),target=densityFromVector(Array.from(input)),a=partialTrace(joint,2,2,'first'),b=partialTrace(joint,2,2,'second');
  return {joint:mWire(joint),first:mWire(a),second:mWire(b),desiredProduct:mWire(tensor(target,target)),
    isExactProductClone:mEqual(joint,tensor(target,target)),singleCopySquaredFidelity:circuitScalarWire(trace(product(target,a))),
    contract:'This tests the CNOT candidate with a blank |0> and one specified pure input. Failure is a counterexample, not a proof of universal no-cloning.'};
}
export function noCloningOverlap(input:readonly C[],other:readonly C[]) {
  requireProtocolVector(input);requireProtocolVector(other);
  const overlap=trace(product(densityFromVector(Array.from(input)),densityFromVector(Array.from(other))));
  return {squaredOverlap:circuitScalarWire(overlap),compatibleWithClosedUnitaryCloning:qEqual(overlap.re,quadratic(zero))||qEqual(overlap.re,quadratic(one)),
    contract:'A fixed blank and a closed unitary cloning both pure inputs would require |<psi|phi>|=|<psi|phi>|^2. This is a necessary condition; no cloner is constructed.'};
}
