/** Independently verify the full unregulated chain-wave subrecord.
 * Abel,packet andAiry observations still need their own checks.
 */
import {snapshotQubitWire} from './learning-qubit-wire';
import {verifyRealTimeExactCore} from './learning-real-time-exact-core-check';
import {CheckerRational as Q,checkerCanonicalReader} from './learning-real-time-validation-rational';
import {createRealTimeCheckerInterval,checkerRealTimeRotation,type CheckerInterval as I} from './learning-real-time-checker-interval';
import {createRealTimeObservationCheck,checkerObservationRecord as record} from './learning-real-time-observation-check';
type CI={real:I;imaginary:I};
let context:ReturnType<typeof createRealTimeCheckerInterval>|undefined;
export function verifyRealTimeUnregulatedWave(source:unknown,exact:unknown,value:unknown) {
  try {
    const core=verifyRealTimeExactCore(source,exact);if(!core||core.input.mode!=='chain')return null;
    const ctx=context??=createRealTimeCheckerInterval(9216),check=createRealTimeObservationCheck(ctx),read=checkerCanonicalReader();
    const owned=snapshotQubitWire(value),Z=BigInt(0),O=BigInt(1);
    const complex=(real:I,imaginary=ctx.zero):CI=>({real,imaginary});
    function octant(k:number):CI {
      const h=ctx.root(ctx.from({n:O,d:BigInt(2)}),2),nh=ctx.neg(h),minus=ctx.neg(ctx.one);
      return[complex(ctx.one),complex(h,nh),complex(ctx.zero,minus),complex(nh,nh),complex(minus),complex(nh,h),complex(ctx.zero,ctx.one),complex(h,h)][((k%8)+8)%8];
    }
    if(core.inertia.zero) {
      const data=record(owned,['kind','parity','maslovIndex','unitPhase','pointwiseValue']),index=core.wire.causticIndex;
      if(typeof index!=='number'||data.kind!=='parity-delta'||!Object.is(data.maslovIndex,index)||!Object.is(data.parity,index%2?-1:1)||data.pointwiseValue!==null)throw Error('Distributional caustic interpretation');
      check.complex(data.unitPhase,octant(2*index));
    }else {
      const data=record(owned,['kind','amplitude','unitPhase','value','logMagnitude']);
      if(data.kind!=='finite-quadratic-kernel')throw Error('Finite wave kind');
      const input=core.input,s=Q.input(input.mass).mul(Q.of(BigInt(input.intervals))).div(Q.input(input.time)),h=Q.input(input.hbar);
      const det=read(core.wire.determinant),absolute=det.n<Z?det.neg():det;
      const power=Q.of(s.n**BigInt(input.intervals),s.d**BigInt(input.intervals));
      const withoutPi=power.div(h.mul(Q.of(BigInt(2))).mul(absolute));
      const amplitude=ctx.root(ctx.divide(ctx.from(withoutPi),ctx.pi()),2);
      const angle=read(core.wire.stationaryAction).div(h),rotation=checkerRealTimeRotation(angle,1536),lift=O<<BigInt(ctx.bits-rotation.bits);
      const phase=complex(ctx.scale(rotation.cos,lift),ctx.scale(rotation.sin,lift)),branch=octant(1+2*core.inertia.negative);
      const unit=complex(ctx.sub(ctx.multiply(phase.real,branch.real),ctx.multiply(phase.imaginary,branch.imaginary)),
        ctx.add(ctx.multiply(phase.real,branch.imaginary),ctx.multiply(phase.imaginary,branch.real)));
      const log=ctx.divideInteger(ctx.sub(ctx.sub(ctx.sub(ctx.scale(ctx.log(ctx.from(s)),input.intervals),ctx.log(ctx.from(h.mul(Q.of(BigInt(2)))))),ctx.log(ctx.from(absolute))),ctx.log(ctx.pi())),2);
      check.scaledPositive(data.amplitude,amplitude);check.complex(data.unitPhase,unit);
      check.scaledComplex(data.value,complex(ctx.multiply(amplitude,unit.real),ctx.multiply(amplitude,unit.imaginary)));
      check.range(data.logMagnitude,log);
    }
    return Object.freeze({scope:'unregulated-chain-wave-only' as const,fullResponseValidated:false as const,observation:owned});
  }catch{return null;}
}
