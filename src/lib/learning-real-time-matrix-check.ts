/** Independent semantic checks of packed finite-chain matrix observations.
 * This is a partial check, never a whole-response validation receipt.
 */
import {snapshotQubitWire} from './learning-qubit-wire';
import {realTimePhaseData} from './learning-real-time-phase-protocol';
import {verifyRealTimeExactCore} from './learning-real-time-exact-core-check';
import {CheckerRational as Q} from './learning-real-time-validation-rational';

type Fraction={n:bigint;d:bigint};
const Z=BigInt(0);
function record(value:unknown,keys:string[]) {
  const result=realTimePhaseData(value,keys);
  if(!result)throw Error('Packed matrix fields');return result;
}
function array(value:unknown,length:number):unknown[] {
  if(!Array.isArray(value)||value.length!==length)throw Error('Packed matrix shape');return value;
}
function coefficient(value:unknown):bigint {
  if(typeof value!=='string'||value.length>160||! /^(?:0|-?[1-9][0-9]*)$/.test(value))throw Error('Canonical matrix coefficient');
  return BigInt(value);
}
/** Validate exact enclosure and resolution using cross products. No rounding,
 * producer packing, normalization or floating point comparison is reused.
 */
function checkPacked(value:unknown,dimension:number,complex:boolean,entry:(i:number,j:number)=>Fraction[]) {
  const data=record(value,['encoding','dimension','coefficientBits','scalePowerOfTwo','upperRows']);
  if(data.encoding!==(complex?'symmetric-complex-binary-intervals':'symmetric-real-binary-intervals')||
    !Object.is(data.dimension,dimension)||!Object.is(data.coefficientBits,512)||
    typeof data.scalePowerOfTwo!=='number'||!Number.isSafeInteger(data.scalePowerOfTwo)||
    Object.is(data.scalePowerOfTwo,-0)||Math.abs(data.scalePowerOfTwo)>32768)throw Error('Packed matrix metadata');
  const shift=512-data.scalePowerOfTwo;
  let hasNonzero=false,hasLargeEntry=false,count=0;
  function range(value:unknown,exact:Fraction) {
    const r=record(value,['lower','upper']),lower=coefficient(r.lower),upper=coefficient(r.upper);
    if(exact.d<=Z||lower>upper||upper-lower>BigInt(3))throw Error('Matrix interval resolution');
    if(exact.n===Z) {
      if(lower!==Z||upper!==Z)throw Error('Matrix exact zero');
    }else {
      hasNonzero=true;
      // Coefficients represent exact*2^(512-scale). Their enclosing box is
      // allowed to differ from the producer's particular outward rounding.
      const n=shift>=0?exact.n<<BigInt(shift):exact.n;
      const d=shift>=0?exact.d:exact.d<<BigInt(-shift);
      if(lower*d>n||upper*d<n)throw Error('Matrix observation misses exact entry');
      const magnitude=n<Z?-n:n,unit=d<<BigInt(512);
      if(magnitude>=unit)throw Error('Matrix shared scale too small');
      if(magnitude*BigInt(4)>unit)hasLargeEntry=true;
    }
    count++;
  }
  array(data.upperRows,dimension).forEach((row,i)=>array(row,dimension-i).forEach((cell,k)=>{
    const expected=entry(i,i+k);
    if(complex){const c=record(cell,['real','imaginary']);range(c.real,expected[0]);range(c.imaginary,expected[1]);}
    else range(cell,expected[0]);
  }));
  if(hasNonzero?!hasLargeEntry:data.scalePowerOfTwo!==0)throw Error('Matrix shared scale too large');
  return count;
}

/** Accept only {inverse,formalConnectedResponse,regulatedConnectedResponse}.
 * Reverify the exact subrecord before trusting any derived matrix entry.
 */
export function verifyRealTimeMatrixObservations(source:unknown,exact:unknown,value:unknown) {
  try {
    const core=verifyRealTimeExactCore(source,exact);
    if(!core||core.input.mode!=='chain')return null;
    const owned=snapshotQubitWire(value),data=record(owned,['inverse','formalConnectedResponse','regulatedConnectedResponse']);
    const formal=record(data.formalConnectedResponse,['interpretation','inverseMultiplier','imaginary']);
    if(formal.interpretation!=='complex-generating-response-not-probability-covariance'||
      formal.inverseMultiplier!==core.input.hbar||formal.imaginary!==true)throw Error('Formal response meaning');
    const d=core.input.intervals-1,h=Q.input(core.input.hbar);
    let realComponents=0,complexComponents=0;
    if(core.inverseEntry)realComponents=checkPacked(data.inverse,d,false,(i,j)=>[core.inverseEntry!(i,j)]);
    else if(data.inverse!==null)throw Error('Caustic inverse must be absent');
    if(core.regulatedInverseEntry)complexComponents=checkPacked(data.regulatedConnectedResponse,d,true,(i,j)=>{
      const x=core.regulatedInverseEntry!(i,j);
      return[x.real,x.imaginary].map(q=>({n:q.n*h.n,d:q.d*h.d}));
    });
    else if(data.regulatedConnectedResponse!==null)throw Error('Unrequested regulated response');
    return Object.freeze({scope:'finite-matrix-observations-only' as const,fullResponseValidated:false as const,
      realComponents,complexComponents,observations:owned});
  }catch{return null;}
}
