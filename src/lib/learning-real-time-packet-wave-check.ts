/** Independently verify every Gaussian packet wave observation. This partial
 * scope does not issue a whole three-mode protocol receipt.
 */
import {snapshotQubitWire} from './learning-qubit-wire';
import {verifyRealTimeExactCore} from './learning-real-time-exact-core-check';
import {CheckerRational as Q,checkerCanonicalReader} from './learning-real-time-validation-rational';
import {createRealTimeCheckerInterval,checkerRealTimeRotation,checkerFloor,type CheckerInterval as I} from './learning-real-time-checker-interval';
import {createRealTimeObservationCheck,checkerObservationRecord as record} from './learning-real-time-observation-check';
type CI={real:I;imaginary:I};
const Z=BigInt(0),O=BigInt(1),T=BigInt(2);
let context:ReturnType<typeof createRealTimeCheckerInterval>|undefined;
// Only33fixed standardized density factors are retained,never candidate
// results orphysical-input-dependent references. Symmetric samples reuse them.
let standardizedDensities:readonly I[]|undefined;
export function verifyRealTimePacketWave(source:unknown,exact:unknown,value:unknown) {
  try {
    const core=verifyRealTimeExactCore(source,exact);if(!core||core.input.mode!=='packet')return null;
    // Uniform physical bounds give variance>2^-1552 and phase sensitivity
    // <2^1516;3072 work bits retain ample margin over512 output bits.
    const ctx=context??=createRealTimeCheckerInterval(3072),check=createRealTimeObservationCheck(ctx),read=checkerCanonicalReader();
    const owned=snapshotQubitWire(value),data=record(owned,['normalization','logNormalization','metaplecticPhase','centerPhase','centerValue','standardDeviation','samples','positionVariance','sampleCoordinates','workingBits','observationBits']);
    if(data.positionVariance!=='final-symmetrized-covariance-qq'||data.sampleCoordinates!=='mean-plus-standard-deviations'||
      !Object.is(data.workingBits,8192)||!Object.is(data.observationBits,512)||!Array.isArray(data.samples)||data.samples.length!==65)throw Error('Packet observation metadata');
    const input=core.input,covariance=(core.wire.finalCovariance as unknown[][]).map(row=>row.map(q=>read(q))),variance=covariance[0][0],sigma=ctx.root(ctx.from(variance),2);
    const h=Q.input(input.hbar),initialVariance=read(core.wire.variance),center=(core.wire.center as unknown[]).map(q=>read(q));
    const complex=(real:I,imaginary=ctx.zero):CI=>({real,imaginary});
    const product=(a:CI,b:CI):CI=>complex(ctx.sub(ctx.multiply(a.real,b.real),ctx.multiply(a.imaginary,b.imaginary)),
      ctx.add(ctx.multiply(a.real,b.imaginary),ctx.multiply(a.imaginary,b.real)));
    const scale=(a:CI,b:I):CI=>complex(ctx.multiply(a.real,b),ctx.multiply(a.imaginary,b));
    const half=ctx.root(ctx.from({n:O,d:T}),2),nh=ctx.neg(half),minus=ctx.neg(ctx.one);
    const octants=[complex(ctx.one),complex(half,nh),complex(ctx.zero,minus),complex(nh,nh),complex(minus),complex(nh,half),complex(ctx.zero,ctx.one),complex(half,half)];
    const octant=(k:number)=>octants[((k%8)+8)%8];
    function rotation(angle:I):CI {
      const result=checkerRealTimeRotation({n:angle.lower+angle.upper,d:T*ctx.S},1536),lift=O<<BigInt(ctx.bits-result.bits);
      const radius=-checkerFloor(-(angle.upper-angle.lower),T);
      const widen=(x:I):I=>({lower:x.lower*lift-radius< -ctx.S?-ctx.S:x.lower*lift-radius,upper:x.upper*lift+radius>ctx.S?ctx.S:x.upper*lift+radius});
      return complex(widen(result.cos),widen(result.sin));
    }
    const A=core.map[0][0],B=core.map[0][1];let metaplectic:CI;
    if(B.n===Z) {
      const k=core.wire.causticIndex;if(typeof k!=='number')throw Error('Packet parity index');metaplectic=octant(2*k);
    }else {
      // alpha=A+B*Z0, with Im(alpha) having the sign of B. Construct its
      // principal inverse-root phase with the stable Cartesian branch.
      // Relative to the producer's positive-real Gaussian coefficient:
      // alpha=i*B*cg; the B<0 case contributes one extra quarter turn.
      const x=A.add(B.mul(Q.input(input.chirp))),y=B.mul(h.div(initialVariance.mul(Q.of(T))));
      const rho=ctx.root(ctx.from(x.mul(x).add(y.mul(y))),2),rootRho=ctx.root(rho,2);
      let real:I,imaginary:I;
      if(x.n>=Z) {
        const a=ctx.root(ctx.divideInteger(ctx.add(rho,ctx.from(x)),2),2);
        real=ctx.divide(a,rootRho);imaginary=ctx.neg(ctx.divide(ctx.from(y),ctx.scale(ctx.multiply(a,rootRho),2)));
      }else {
        const b=ctx.root(ctx.divideInteger(ctx.sub(rho,ctx.from(x)),2),2),ay=y.n<Z?y.neg():y;
        real=ctx.divide(ctx.from(ay),ctx.scale(ctx.multiply(b,rootRho),2));
        imaginary=ctx.scale(ctx.divide(b,rootRho),y.n<Z?1:-1);
      }
      const winding=core.inertia.negative+(B.n<Z?1:0);metaplectic=product(octant(2*winding),complex(real,imaginary));
    }
    const centerAction=read(core.wire.centralAction),centerAngle=ctx.from(centerAction.div(h)),centerPhase=product(metaplectic,rotation(centerAngle));
    const twoPiVariance=ctx.multiply(ctx.scale(ctx.pi(),2),ctx.from(variance));
    // Derive wave amplitude from the physical density, rather than squaring
    // the producer's amplitude/envelope to recover the density.
    const centralDensity=ctx.divide(ctx.one,ctx.root(twoPiVariance,2)),normalization=ctx.root(centralDensity,2);
    const logNormalization=ctx.neg(ctx.divideInteger(ctx.add(ctx.add(ctx.log(ctx.integer(2)),ctx.log(ctx.pi())),ctx.log(ctx.from(variance))),4));
    check.scaledPositive(data.normalization,normalization);check.range(data.logNormalization,logNormalization);
    check.complex(data.metaplecticPhase,metaplectic);check.complex(data.centerPhase,centerPhase);
    check.scaledComplex(data.centerValue,scale(centerPhase,normalization));check.scaledPositive(data.standardDeviation,sigma);
    const factors=standardizedDensities??=Object.freeze(Array.from({length:33},(_,k)=>
      Object.freeze(ctx.expNegative(ctx.from({n:BigInt(k*k),d:BigInt(128)})))));
    for(let index=0;index<65;index++) {
      const row=record(data.samples[index],['standardDeviations','position','unitPhase','value','density']),offset=Q.of(BigInt(index-32),BigInt(8));
      if(!Object.is(row.standardDeviations,(index-32)/8))throw Error('Packet sample ordering');
      const dx=ctx.multiply(ctx.from(offset),sigma),position=ctx.add(ctx.from(center[0]),dx);
      // dx=offset*sqrt(V),so (Sigma_qp/V)*dx^2=Sigma_qp*offset^2
      // exactly. Cancel V anddivide byhbar in rational coefficients first.
      const linear=ctx.multiply(ctx.from(center[1].mul(offset).div(h)),sigma);
      const quadratic=ctx.from(covariance[0][1].mul(offset.mul(offset)).div(h.mul(Q.of(T))));
      const totalAngle=ctx.add(centerAngle,ctx.add(linear,quadratic)),unit=product(metaplectic,rotation(totalAngle));
      const density=ctx.multiply(centralDensity,factors[Math.abs(index-32)]);
      check.range(row.position,position);check.complex(row.unitPhase,unit);
      check.scaledComplex(row.value,scale(unit,ctx.root(density,2)));check.scaledPositive(row.density,density);
    }
    return Object.freeze({scope:'packet-wave-only' as const,fullResponseValidated:false as const,samplesVerified:65,observation:owned});
  }catch{return null;}
}
