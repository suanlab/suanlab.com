/** Exact observables for the finite pure-state circuit teaching model. Worker/CLI only. */
import { runQubitCircuit,inverseCircuit,type CircuitGate } from './learning-qubit-circuit-exact';
import { circuitNormSquared } from './learning-qubit-scalar';
import { f,zero,one,quadratic,complex,Z,O,plus,times,conjugate,equal,minus,divide,
  qSub,qMul,qDiv,qSign,mWire,mEqual,mSub,identity,zeros,adjoint,product,trace,tensor,
  circuitScalarWire,type Q,type C,type M } from './learning-qubit-channel-algebra';

const X:M=[[Z,O],[O,Z]],Y:M=[[Z,complex(quadratic(zero),quadratic(f(BigInt(-1))))],[complex(quadratic(zero),quadratic(one)),Z]],ZZ:M=[[O,Z],[Z,complex(quadratic(f(BigInt(-1))))]];
const axes=[X,Y,ZZ];
const wireReal=(x:Q)=>circuitScalarWire(complex(x));
const abs2=(z:C)=>times(z,conjugate(z)).re;
const density=(state:C[],norm:Q):M=>state.map(a=>state.map(b=>divide(times(a,conjugate(b)),norm)));
function reduced(d:M,sites:number,site:number):M {
  const mask=2**(sites-site-1),out=zeros(2);
  for(let i=0;i<d.length;i++)for(let j=0;j<d.length;j++){
    if((i&~mask)===(j&~mask)){
      const a=(i&mask)?1:0,b=(j&mask)?1:0;out[a][b]=plus(out[a][b],d[i][j]);
    }
  }return out;
}
function pauliOperator(sites:number,first:number,axis:number,second=-1,otherAxis=0):M {
  let out:M=[[O]];
  for(let k=0;k<sites;k++)out=tensor(out,k===first?axes[axis]:k===second?axes[otherAxis]:identity(2));
  return out;
}
function expectation(rho:M,A:M):C {
  let value=Z;
  for(let i=0;i<rho.length;i++)for(let j=0;j<rho.length;j++)value=plus(value,times(rho[i][j],A[j][i]));
  return value;
}

export function qubitCircuitReport(sites:number,gates:readonly CircuitGate[],input:readonly C[]){
  const calculation=runQubitCircuit(sites,gates,input),norm=circuitNormSquared(input);
  if(qSign(norm)<=0)throw new RangeError('Positive input norm required');
  const dimension=2**sites;
  const localOperators=Array.from({length:sites},(_,site)=>axes.map((_,axis)=>pauliOperator(sites,site,axis)));
  const pairs=Array.from({length:sites},(_,first)=>Array.from({length:sites-first-1},(_,offset)=>{
    const second=first+offset+1;
    return {first,second,operators:axes.map((_,a)=>axes.map((_,b)=>pauliOperator(sites,first,a,second,b)))};
  })).flat();
  const prefixes=calculation.prefixes.map((state,step)=>{
    const rho=density(state,norm);
    const bloch=localOperators.map(operators=>operators.map(A=>expectation(rho,A)));
    const local=Array.from({length:sites},(_,site)=>{
      const d=reduced(rho,sites,site);
      return {site,density:mWire(d),purity:circuitScalarWire(trace(product(d,d))),bloch:bloch[site].map(circuitScalarWire)};
    });
    return {step,rawAmplitudes:state.map(circuitScalarWire),normSquared:wireReal(circuitNormSquared(state)),
      normResidual:wireReal(qSub(circuitNormSquared(state),norm)),density:mWire(rho),
      probabilities:state.map(z=>wireReal(qDiv(abs2(z),norm))),
      purity:circuitScalarWire(trace(product(rho,rho))),singleQubits:local,
      pauliPairs:pairs.map(pair=>{
        const correlations=pair.operators.map(row=>row.map(A=>expectation(rho,A)));
        return {first:pair.first,second:pair.second,expectations:correlations.map(row=>row.map(circuitScalarWire)),
          connected:correlations.map((row,a)=>row.map((v,b)=>circuitScalarWire(minus(v,times(bloch[pair.first][a],bloch[pair.second][b])))))};
      })};
  });
  // Independent columns expose the implemented full unitary, not just one state.
  const columns=Array.from({length:dimension},(_,basis)=>runQubitCircuit(sites,gates,Array.from({length:dimension},(_,i)=>i===basis?O:Z)).state);
  const U=Array.from({length:dimension},(_,i)=>columns.map(column=>column[i]));
  const residual=mSub(product(adjoint(U),U),identity(dimension));
  const restored=runQubitCircuit(sites,inverseCircuit(gates),calculation.state).state;
  const overlap=input.reduce((s,z,i)=>plus(s,times(conjugate(z),calculation.state[i])),Z);
  const fidelity=qDiv(abs2(overlap),qMul(norm,norm));
  const inverseOverlap=input.reduce((s,z,i)=>plus(s,times(conjugate(z),restored[i])),Z);
  return {schemaVersion:1,basisOrder:'site-zero-most-significant',gateOrder:'chronological',amplitudeConvention:'raw-with-exact-squared-norm',fidelityConvention:'squared-pure-state-overlap',
    sites,gates:gates.map(op=>({...op})),pauliOrder:['X','Y','Z'],inputNormSquared:wireReal(norm),prefixes,
    unitary:mWire(U),unitarityResidual:mWire(residual),unitaryVerified:mEqual(residual,zeros(dimension)),
    overlapWithInput:circuitScalarWire(divide(overlap,norm)),fidelityWithInput:wireReal(fidelity),
    sameAmplitudesAsInput:calculation.state.every((z,i)=>equal(z,input[i])),
    samePhysicalStateAsInput:mEqual(density(calculation.state,norm),density(input.slice(),norm)),
    inverseRawAmplitudes:restored.map(circuitScalarWire),inverseResidual:restored.map((z,i)=>circuitScalarWire(minus(z,input[i]))),
    inverseFidelity:wireReal(qDiv(abs2(inverseOverlap),qMul(norm,norm))),inverseRecovered:restored.every((z,i)=>equal(z,input[i]))};
}
