/** Explicit two-outcome LOCC dilution family, not a general protocol synthesizer. */
import { f, zero, one, add, mul, eq, type F } from './learning-quantum-hilbert-core';
import { requireProtocolVector } from './learning-quantum-information-protocols';
import { complex,quadratic,Z,O,identity,tensor,product,adjoint,mAdd,mWire,circuitScalarWire,qDiv,divide,conjugate,times,type M } from './learning-qubit-channel-algebra';
import { circuitNormSquared } from './learning-qubit-scalar';
export function bellPairDilution(a:F,b:F){
 requireProtocolVector([complex(quadratic(a)),complex(quadratic(b))]);
 if(a.n<BigInt(0)||b.n<BigInt(0)||!eq(add(mul(a,a),mul(b,b)),one))throw new RangeError('Nonnegative rational amplitudes with a²+b²=1 required');
 const ca=complex(quadratic(a)),cb=complex(quadratic(b)),M0=[[ca,Z],[Z,cb]],M1=[[cb,Z],[Z,ca]],X=[[Z,O],[O,Z]];
 const h=complex(quadratic(zero,f(BigInt(1),BigInt(2)))),bell=[h,Z,Z,h];
 const target=[ca,Z,Z,cb];
 const rho=(v:typeof target):M=>v.map(x=>v.map(y=>divide(times(x,conjugate(y)),circuitNormSquared(v))));
 const branches=[M0,M1].map((M,index)=>{
  const raw=product(tensor(M,identity(2)),bell.map(z=>[z])).map(row=>row[0]);
  const corrected=index===0?raw:product(tensor(X,X),raw.map(z=>[z])).map(row=>row[0]);
  return {outcome:index,probability:circuitScalarWire(complex(qDiv(circuitNormSquared(raw),circuitNormSquared(bell)))).real,
   raw:raw.map(circuitScalarWire),corrected:corrected.map(circuitScalarWire),correctedDensity:mWire(rho(corrected)),bothPartiesApplyX:index===1};
 });
 return {target:target.map(circuitScalarWire),targetDensity:mWire(rho(target)),aliceOperators:[mWire(M0),mWire(M1)],
  measurementCompleteness:mWire(mAdd(product(adjoint(M0),M0),product(adjoint(M1),M1))),branches,
  contract:'One Bell pair to a|00>+b|11>, a,b nonnegative rational and a²+b²=1. Alice performs a two-outcome local measurement and announces its outcome; both parties apply X on outcome1. Each outcome has probability1/2. This explicit family does not synthesize general majorization/Vidal-optimal protocols; no Worker/public input reader/UI.'};
}
