/** Worker-side directed argument reduction for large exact rational phases.
 * Machin's identity supplies a certified pi interval using alternating atan
 * series. Reduce at magnitude-adaptive precision before evaluating Taylor
 * series on [-4,4]; no floating point phase or trigonometry is used.
 */
export interface RealTimeRational { n: bigint; d: bigint; }
export interface PhaseInterval { lower: bigint; upper: bigint; }
export interface CertifiedPhaseRotation {
  sin: PhaseInterval; cos: PhaseInterval; precisionBits: number;
  reductionBits: number; taylorTerms: number; turns: string;
}
const Z = BigInt(0), O = BigInt(1), T = BigInt(2);
const floor = (n: bigint, d: bigint): bigint => n >= Z ? n/d : -((-n+d-O)/d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const bitLength = (n: bigint) => (n < Z ? -n : n).toString(2).length;
const piCache = new Map<number, Readonly<PhaseInterval>>();

/** Returns bounds in integer units of 2^-bits, including rounding and tails. */
export function realTimePiInterval(bits: number): Readonly<PhaseInterval> {
  if (!Number.isSafeInteger(bits) || bits < 64 || bits > 68096) throw new RangeError('Pi precision outside64..68096bits');
  const found = piCache.get(bits); if (found) return found;
  const scale = O << BigInt(bits);
  function atan(q: bigint): PhaseInterval {
    let power = {lower:floor(scale,q),upper:ceil(scale,q)}, lower = Z, upper = Z, j = 0;
    for (;;) {
      const divisor=BigInt(2*j+1);
      if (power.upper <= O) {
        // The exact positive power remains in this directed interval.
        // Alternation bounds the remainder by the first omitted term.
        const tail = ceil(power.upper, divisor);
        if (j%2) lower -= tail; else upper += tail;
        return {lower, upper};
      }
      const lo = floor(power.lower, divisor), hi = ceil(power.upper, divisor);
      if (j%2) { lower -= hi; upper -= lo; } else { lower += lo; upper += hi; }
      power={lower:floor(power.lower,q*q),upper:ceil(power.upper,q*q)};j++;
    }
  }
  const a = atan(BigInt(5)), b = atan(BigInt(239));
  const result = Object.freeze({lower: BigInt(16)*a.lower-BigInt(4)*b.upper, upper: BigInt(16)*a.upper-BigInt(4)*b.lower});
  // Bounded cache; a long-lived Worker must not retain every requested precision.
  if (piCache.size >= 8) piCache.delete(piCache.keys().next().value!);
  piCache.set(bits, result); return result;
}

export function certifiedRealTimeRotation(phase: RealTimeRational, precisionBits = 512): CertifiedPhaseRotation {
  if (!phase || typeof phase.n !== 'bigint' || typeof phase.d !== 'bigint' || phase.d <= Z
    || bitLength(phase.n) > 65536 || bitLength(phase.d) > 65536) throw new RangeError('Bounded rational phase with positive denominator required');
  if (!Number.isSafeInteger(precisionBits) || precisionBits < 64 || precisionBits > 2048) throw new RangeError('Phase observation requires64..2048bits');
  const outputScale = O << BigInt(precisionBits);
  if (phase.n === Z) return {sin:{lower:Z,upper:Z}, cos:{lower:outputScale,upper:outputScale}, precisionBits, reductionBits:0, taylorTerms:0, turns:'0'};
  const magnitudeBits = Math.max(0, bitLength(phase.n)-bitLength(phase.d)+1);
  const reductionBits = Math.ceil((magnitudeBits+precisionBits+128)/64)*64;
  const pi = realTimePiInterval(reductionBits), reductionScale = O << BigInt(reductionBits);
  // Any integer number of full turns is valid. This midpoint-based choice
  // makes the remainder small without needing an exact rounding decision.
  const piMid = (pi.lower+pi.upper)/T;
  const turns = floor(phase.n*reductionScale+phase.d*piMid, T*phase.d*piMid);
  const x = {lower:floor(phase.n*reductionScale,phase.d),upper:ceil(phase.n*reductionScale,phase.d)};
  const factor=T*turns;
  const r = factor >= Z ? {lower:x.lower-factor*pi.upper,upper:x.upper-factor*pi.lower}
    : {lower:x.lower-factor*pi.lower,upper:x.upper-factor*pi.upper};
  const workBits=precisionBits+64, S=O<<BigInt(workBits), shrink=O<<BigInt(reductionBits-workBits);
  const angle={lower:floor(r.lower,shrink),upper:ceil(r.upper,shrink)};
  if (angle.lower < -BigInt(4)*S || angle.upper > BigInt(4)*S) throw new RangeError('Argument reduction failed to enclose a small phase');
  const add=(a:PhaseInterval,b:PhaseInterval)=>({lower:a.lower+b.lower,upper:a.upper+b.upper});
  const sub=(a:PhaseInterval,b:PhaseInterval)=>({lower:a.lower-b.upper,upper:a.upper-b.lower});
  const mul=(a:PhaseInterval,b:PhaseInterval):PhaseInterval=>{
    const p=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];
    return {lower:floor(p.reduce((a,b)=>a<b?a:b),S),upper:ceil(p.reduce((a,b)=>a>b?a:b),S)};
  };
  const divide=(a:PhaseInterval,d:bigint)=>({lower:floor(a.lower,d),upper:ceil(a.upper,d)});
  const x2=mul(angle,angle);
  let st=angle,ct={lower:S,upper:S},sin=angle,cos=ct;
  // After j=8, subsequent absolute term ratios are <=16/(17*18)<1/16.
  // At j=8 both next sine/cosine terms are <1; each further ratio
  // supplies >=4 bits. After terms steps each tail is <2^-workBits.
  const terms=8+Math.ceil((workBits+8)/4);
  for(let j=1;j<=terms;j++) {
    st=divide(mul(st,x2),BigInt(2*j)*BigInt(2*j+1));
    ct=divide(mul(ct,x2),BigInt(2*j-1)*BigInt(2*j));
    sin=j%2?sub(sin,st):add(sin,st);cos=j%2?sub(cos,ct):add(cos,ct);
  }
  const shrinkOutput=O<<BigInt(64);
  const out=(v:PhaseInterval):PhaseInterval=>({lower:floor(v.lower-O,shrinkOutput)<-outputScale?-outputScale:floor(v.lower-O,shrinkOutput),
    upper:ceil(v.upper+O,shrinkOutput)>outputScale?outputScale:ceil(v.upper+O,shrinkOutput)});
  return {sin:out(sin),cos:out(cos),precisionBits,reductionBits,taylorTerms:terms,turns:String(turns)};
}
