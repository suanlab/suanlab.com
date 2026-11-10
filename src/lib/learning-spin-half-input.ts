/** Kinematic spin rotations: turns are angle/(2pi), not physical elapsed time. */
export const spinHalfKeys=['stateReal0','stateImag0','stateReal1','stateImag1','axisReal0','axisImag0','axisReal1','axisImag1','rotationReal0','rotationImag0','rotationReal1','rotationImag1','turnNumerator','turnDenominator','hbarNumerator','hbarDenominator','sampleSteps'] as const;
export type SpinHalfParameters=Record<typeof spinHalfKeys[number],number>;
export function isSpinHalfParameters(x:unknown):x is SpinHalfParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==spinHalfKeys.length)return false;const p=x as SpinHalfParameters;
 return spinHalfKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000)&&p.turnDenominator>0&&p.hbarNumerator>0&&p.hbarDenominator>0&&p.sampleSteps>=1&&p.sampleSteps<=16&&['state','axis','rotation'].every(prefix=>[0,1].some(i=>p[`${prefix}Real${i}` as keyof SpinHalfParameters]!==0||p[`${prefix}Imag${i}` as keyof SpinHalfParameters]!==0));
}
export function snapshotSpinHalfParameters(x:SpinHalfParameters):Readonly<SpinHalfParameters>{if(!isSpinHalfParameters(x))throw new RangeError('Seventeen owned bounded integers; nonzero complex state and two axes,positive denominators/hbar,signed turns and1..16 sample intervals');return Object.freeze(Object.fromEntries(spinHalfKeys.map(k=>[k,x[k]+0]))) as SpinHalfParameters;}
