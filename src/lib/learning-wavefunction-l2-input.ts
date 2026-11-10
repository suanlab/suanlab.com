/** Piecewise-affine complex g(u), u=x/L; no Hamiltonian is specified. */
export const wavefunctionL2Keys=['lengthNumerator','lengthDenominator','breakNumerator','breakDenominator','leftRealConstant','leftRealSlope','leftImagConstant','leftImagSlope','rightRealConstant','rightRealSlope','rightImagConstant','rightImagSlope','cutoffDegree','bins','sampleSteps']as const;
export type WavefunctionL2Parameters=Record<typeof wavefunctionL2Keys[number],number>;
export function isWavefunctionL2Parameters(value:unknown):value is WavefunctionL2Parameters{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==wavefunctionL2Keys.length)return false;const p=value as WavefunctionL2Parameters;
 if(!wavefunctionL2Keys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000))return false;
 if(p.lengthNumerator<=0||p.lengthDenominator<=0||p.lengthNumerator>10*p.lengthDenominator||p.breakDenominator<=0||p.breakNumerator<0||p.breakNumerator>p.breakDenominator||p.cutoffDegree<0||p.cutoffDegree>12||p.bins<1||p.bins>32||p.sampleSteps<1||p.sampleSteps>64)return false;
 return(p.breakNumerator>0&&['leftRealConstant','leftRealSlope','leftImagConstant','leftImagSlope'].some(k=>p[k as keyof WavefunctionL2Parameters]!==0))||(p.breakNumerator<p.breakDenominator&&['rightRealConstant','rightRealSlope','rightImagConstant','rightImagSlope'].some(k=>p[k as keyof WavefunctionL2Parameters]!==0));
}
export function snapshotWavefunctionL2Parameters(p:WavefunctionL2Parameters):Readonly<WavefunctionL2Parameters>{if(!isWavefunctionL2Parameters(p))throw new RangeError('Fifteen owned integers of magnitude<=1e6;0<L<=10,positive denominators,0<=break<=1,nonzero L2 norm,degree0..12,bins1..32,sample intervals1..64');return Object.freeze(Object.fromEntries(wavefunctionL2Keys.map(k=>[k,p[k]+0])))as WavefunctionL2Parameters;}
