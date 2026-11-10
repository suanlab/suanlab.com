/** Independent Abel-wave subrecord verification. This is not a whole result
 * receipt; unregulated waves,matrices,packets andfolds have separate checks.
 */
import {snapshotQubitWire} from './learning-qubit-wire';
import {verifyRealTimeExactCore} from './learning-real-time-exact-core-check';
import {CheckerRational as Q,checkerCanonicalReader} from './learning-real-time-validation-rational';
import {createRealTimeCheckerInterval,checkerRealTimeRotation,checkerFloor,type CheckerInterval as I} from './learning-real-time-checker-interval';
import {createRealTimeObservationCheck,checkerObservationRecord as record} from './learning-real-time-observation-check';
const Z=BigInt(0),O=BigInt(1),T=BigInt(2);
type CI={real:I;imaginary:I};
let context:ReturnType<typeof createRealTimeCheckerInterval>|undefined;

/** Raw values can reach the producer's absolute 8192-bit grid floor.
 * Tight relative output resolution is retained except for this explicitly
 * bounded floor:32*(1+prefactor)*2^-8192 plus8 output quanta.
 */
function rawObservation(value:unknown,expected:CI,amplitude:I,ctx:ReturnType<typeof createRealTimeCheckerInterval>) {
  const data=record(value,['scalePowerOfTwo','real','imaginary']),e=data.scalePowerOfTwo;
  if(typeof e!=='number'||!Number.isSafeInteger(e)||Object.is(e,-0)||Math.abs(e)>32768)throw Error('Raw exponent');
  const read=checkerCanonicalReader(),out=O<<BigInt(512);
  const ceil=(n:bigint,d:bigint)=>-checkerFloor(-n,d);
  const observationBudget=e>=512?(BigInt(8)*ctx.S)<<BigInt(e-512):ceil(BigInt(8)*ctx.S,O<<BigInt(512-e));
  const floorBudget=ceil(BigInt(32)*(amplitude.upper+ctx.S),O<<BigInt(8192));
  let containsOrigin=true,normalized=false;
  for(const key of ['real','imaginary']as const) {
    const cell=record(data[key],['lower','upper']),lo=read(cell.lower,32768),hi=read(cell.upper,32768);
    for(const q of [lo,hi]) {
      if(q.d>out||(q.d&(q.d-O))!==Z)throw Error('Raw dyadic precision');
      const magnitude=q.n<Z?-q.n:q.n;
      if(magnitude>=T*q.d)throw Error('Raw mantissa range');
      if(BigInt(4)*magnitude>q.d)normalized=true;
    }
    if(lo.n*hi.d>hi.n*lo.d)throw Error('Raw range ordering');
    containsOrigin=containsOrigin&&lo.n<=Z&&hi.n>=Z;
    const shifted=(q:Q)=>e>=0?{n:q.n<<BigInt(e),d:q.d}:{n:q.n,d:q.d<<BigInt(-e)};
    const l=shifted(lo),h=shifted(hi);
    if(l.n*ctx.S>expected[key].lower*l.d||h.n*ctx.S<expected[key].upper*h.d)throw Error('Raw misses independent enclosure');
    if((h.n*l.d-l.n*h.d)*ctx.S>(observationBudget+floorBudget)*h.d*l.d)throw Error('Raw grid resolution');
  }
  if(!normalized)throw Error('Raw zero orunnormalized value');
  return containsOrigin;
}

export function verifyRealTimeRegulatedWave(source:unknown,exact:unknown,value:unknown) {
  try {
    const core=verifyRealTimeExactCore(source,exact);if(!core||core.input.mode!=='chain')return null;
    const owned=snapshotQubitWire(value),input=core.input;
    if(Q.input(input.damping).n===Z) {
      if(owned!==null)return null;
      return Object.freeze({scope:'abel-chain-wave-only' as const,fullResponseValidated:false as const,observation:owned});
    }
    const ctx=context??=createRealTimeCheckerInterval(9216),check=createRealTimeObservationCheck(ctx),read=checkerCanonicalReader();
    const data=record(owned,['kind','scaledAmplitude','unitPhase','scaledValue','value','negativeExponential','logMagnitude','rawValueRequiresScaledRepresentation','branch']);
    if(data.kind!=='abel-regulated-kernel'||data.branch!=='product-of-positive-real-pivot-square-roots')throw Error('Abel interpretation');
    const regulator=core.wire.regulator as Record<string,unknown>;
    const complex=(real:I,imaginary=ctx.zero):CI=>({real,imaginary});
    const multiply=(a:CI,b:CI):CI=>complex(ctx.sub(ctx.multiply(a.real,b.real),ctx.multiply(a.imaginary,b.imaginary)),
      ctx.add(ctx.multiply(a.real,b.imaginary),ctx.multiply(a.imaginary,b.real)));
    const scale=(a:CI,b:I):CI=>complex(ctx.multiply(a.real,b),ctx.multiply(a.imaginary,b));
    const qc=(value:unknown)=>{const q=value as{real:unknown;imaginary:unknown};return{r:read(q.real),i:read(q.imaginary)};};
    const norm=(z:ReturnType<typeof qc>)=>z.r.mul(z.r).add(z.i.mul(z.i));
    const s=Q.input(input.mass).mul(Q.of(BigInt(input.intervals))).div(Q.input(input.time)),h=Q.input(input.hbar);
    const det=qc(regulator.determinant),detNorm=norm(det),detMagnitude=ctx.root(ctx.from(detNorm),2);
    const power=Q.of(s.n**BigInt(input.intervals),s.d**BigInt(input.intervals));
    const squared=ctx.divide(ctx.from(power.div(h.mul(Q.of(T)))),ctx.multiply(ctx.pi(),detMagnitude));
    const amplitude=ctx.root(squared,2);
    const half=ctx.root(ctx.from({n:O,d:T}),2),nh=ctx.neg(half),minus=ctx.neg(ctx.one);
    const octants=[complex(ctx.one),complex(half,nh),complex(ctx.zero,minus),complex(nh,nh),complex(minus),complex(nh,half),complex(ctx.zero,ctx.one),complex(half,half)];
    let unit=octants[input.intervals%8];
    const minors=(regulator.continuants as unknown[]).map(qc);
    for(let j=1;j<minors.length;j++) {
      const a=minors[j],b=minors[j-1],denominator=norm(b);
      const real=a.r.mul(b.r).add(a.i.mul(b.i)).div(denominator),imaginary=a.i.mul(b.r).sub(a.r.mul(b.i)).div(denominator);
      if(real.n<=Z)throw Error('Abel pivot lacks accretive branch');
      // Construct the Cartesian principal square root first,then normalize
      // its conjugate. No principal square root of the total determinant.
      const rho=ctx.root(ctx.from(real.mul(real).add(imaginary.mul(imaginary))),2);
      const cartesianReal=ctx.root(ctx.divideInteger(ctx.add(rho,ctx.from(real)),2),2),rootRho=ctx.root(rho,2);
      const pivotPhase=complex(ctx.divide(cartesianReal,rootRho),ctx.neg(ctx.divide(ctx.from(imaginary),ctx.scale(ctx.multiply(cartesianReal,rootRho),2))));
      unit=multiply(unit,pivotPhase);
    }
    const exponent=qc(regulator.exponent);if(exponent.r.n>Z)throw Error('Abel exponent amplifies');
    const rotation=checkerRealTimeRotation(exponent.i,1536),lift=O<<BigInt(ctx.bits-rotation.bits);
    unit=multiply(unit,complex(ctx.scale(rotation.cos,lift),ctx.scale(rotation.sin,lift)));
    const decay=ctx.from(exponent.r.neg()),scaled=scale(unit,amplitude),raw=scale(scaled,ctx.expNegative(decay));
    const logAmplitude=ctx.divideInteger(ctx.sub(ctx.sub(ctx.sub(ctx.scale(ctx.log(ctx.from(s)),input.intervals),ctx.log(ctx.from(h.mul(Q.of(T))))),ctx.divideInteger(ctx.log(ctx.from(detNorm)),2)),ctx.log(ctx.pi())),2);
    check.scaledPositive(data.scaledAmplitude,amplitude);check.complex(data.unitPhase,unit);check.scaledComplex(data.scaledValue,scaled);
    check.range(data.negativeExponential,decay);check.range(data.logMagnitude,ctx.sub(logAmplitude,decay));
    const needsScale=rawObservation(data.value,raw,amplitude,ctx);
    if(data.rawValueRequiresScaledRepresentation!==needsScale)throw Error('Raw display annotation');
    return Object.freeze({scope:'abel-chain-wave-only' as const,fullResponseValidated:false as const,observation:owned});
  }catch{return null;}
}
