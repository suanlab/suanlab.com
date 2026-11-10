import {binaryProduct as product,binarySum as addBinary,binaryMultiply as multiply,binaryObservation as observe,binaryRatioObservation as ratio,type BinaryValue} from './learning-binary-products';
/** Two smooth teaching families. No shock capturing, boundary solver or turbulence model. */
export type FlowValue = number | null;
export type FlowVector = [FlowValue, FlowValue];
function bound(v: number, lo: number, hi: number, name: string) {
  if (!Number.isFinite(v) || v < lo || v > hi) throw new RangeError(`Invalid ${name}`);
}
const finite = (v: number): FlowValue => Number.isFinite(v) ? v : null;
const sum = (...vs: FlowValue[]): FlowValue => vs.includes(null) ? null : finite((vs as number[]).reduce((a, b) => a + b, 0));
const neg = (v: FlowValue): FlowValue => v === null ? null : -v;
function mul(...vs: FlowValue[]): FlowValue {
  if (vs.includes(0)) return 0;
  if (vs.includes(null)) return null;
  let result = 1;
  for (const v of vs as number[]) { result *= v; if (result === 0 || !Number.isFinite(result)) return null; }
  return result;
}
function div(a: FlowValue, b: FlowValue): FlowValue {
  if (a === null || b === null || b === 0) return null;
  const value = a / b;
  return value === 0 && a !== 0 ? null : finite(value);
}

export interface AcousticParameters {
  density: number; soundSpeed: number; meanVelocity: number; gamma: number;
  waveNumber: number; rightAmplitude: number; leftAmplitude: number;
  contactAmplitude: number; phase: number;
}
function validAcoustic(p: AcousticParameters) {
  bound(p.density, .25, 4, 'density'); bound(p.soundSpeed, .25, 4, 'sound speed');
  bound(p.meanVelocity, -4, 4, 'mean velocity'); bound(p.gamma, 1.1, 1.67, 'gamma');
  bound(p.waveNumber, .25, 4, 'wave number'); bound(p.phase, -Math.PI, Math.PI, 'phase');
  for (const key of ['rightAmplitude', 'leftAmplitude', 'contactAmplitude'] as const) bound(p[key], -.03, .03, key);
}
/** Isentropic acoustic modes plus a pressure-free contact perturbation in a uniform ideal gas. */
export function linearizedEulerAt(p: AcousticParameters, x: number, time: number) {
  validAcoustic(p); bound(x, -8, 8, 'x'); bound(time, 0, 4, 'time');
  const { density: r, soundSpeed: c, meanVelocity: U, waveNumber: k, gamma } = p;
  const rc2=product(r,c,c),backgroundPressure=ratio(rc2,product(gamma)).value!;
  const speeds = [U + c, U - c, U] as const;
  const amplitudes = [p.rightAmplitude, p.leftAmplitude, p.contactAmplitude];
  const rawModes=speeds.map((speed,i)=>{
    const phase=k*(x-speed*time)+p.phase,value=product(amplitudes[i],Math.cos(phase)),dx=product(-amplitudes[i],k,Math.sin(phase));
    return{speed,phase,value,dx,dt:multiply(product(-speed),dx)};
  });
  const component=(key:'value'|'dx'|'dt')=>{
    const [a,b,e]=rawModes.map(mode=>mode[key]),ab=addBinary([a,multiply(product(-1),b)]);
    return{density:multiply(product(r),addBinary([ab,e])),velocity:multiply(product(c),addBinary([a,b])),pressure:multiply(rc2,ab)};
  };
  const project=(v:{density:BinaryValue;velocity:BinaryValue;pressure:BinaryValue})=>({density:observe(v.density).value,velocity:observe(v.velocity).value,pressure:observe(v.pressure).value});
  const values=component('value'),rawX=component('dx'),rawT=component('dt');
  const perturbation=project(values),dx=project(rawX),dt=project(rawT);
  const rho=addBinary([product(r),values.density]),v=addBinary([product(U),values.velocity]);
  const pressureNumerator=addBinary([rc2,multiply(product(gamma),values.pressure)]);
  const density=observe(rho).value,velocity=observe(v).value,pressure=ratio(pressureNumerator,product(gamma)).value;
  const linearResidual={
    mass:observe(addBinary([rawT.density,multiply(product(U),rawX.density),multiply(product(r),rawX.velocity)])).value,
    velocity:ratio(addBinary([multiply(product(r),addBinary([rawT.velocity,multiply(product(U),rawX.velocity)])),rawX.pressure]),product(r)).value,
    pressure:observe(addBinary([rawT.pressure,multiply(product(U),rawX.pressure),multiply(rc2,rawX.velocity)])).value,
  };
  // Evaluate the same unprojected fields in nonlinear primitive equations.
  const nonlinearResidual={
    mass:observe(addBinary([rawT.density,multiply(v,rawX.density),multiply(rho,rawX.velocity)])).value,
    velocity:ratio(addBinary([multiply(rho,addBinary([rawT.velocity,multiply(v,rawX.velocity)])),rawX.pressure]),rho).value,
    pressure:observe(addBinary([rawT.pressure,multiply(v,rawX.pressure),multiply(pressureNumerator,rawX.velocity)])).value,
  };
  const energy=multiply(rc2,addBinary(rawModes.slice(0,2).map(mode=>multiply(mode.value,mode.value))));
  const energyDerivative=(key:'dx'|'dt')=>multiply(product(2,r,c,c),addBinary(rawModes.slice(0,2).map(mode=>multiply(mode.value,mode[key]))));
  const energyFlux=observe(addBinary([multiply(product(U),energy),multiply(values.pressure,values.velocity)])).value;
  const energyResidual=observe(addBinary([energyDerivative('dt'),multiply(product(U),energyDerivative('dx')),multiply(rawX.pressure,values.velocity),multiply(values.pressure,rawX.velocity)])).value;
  const modes=rawModes.map(mode=>({...mode,value:observe(mode.value).value,dx:observe(mode.dx).value,dt:observe(mode.dt).value}));
  return {
    x,time,backgroundPressure,modes,perturbation,density,velocity,pressure,dx,dt,
    characteristicSpeeds:speeds,rightInvariant:observe(multiply(product(2,c),rawModes[0].value)).value,leftInvariant:observe(multiply(product(2,c),rawModes[1].value)).value,
    contactVariable:observe(multiply(product(-r,c,c),rawModes[2].value)).value,
    roundedContactVariable:sum(perturbation.pressure,neg(mul(c,c,perturbation.density))),
    linearResidual,nonlinearResidual,acousticEnergy:observe(energy).value,energyFlux,energyResidual,
    positiveBackgroundAndTotals:density!==null&&pressure!==null&&density>0&&pressure>0,
    maxRelativeDensityBound:Math.abs(p.rightAmplitude)+Math.abs(p.leftAmplitude)+Math.abs(p.contactAmplitude),
    maxRelativePressureBound:gamma*(Math.abs(p.rightAmplitude)+Math.abs(p.leftAmplitude)),
  };
}

export interface TaylorGreenParameters {
  density: number; amplitude: number; waveNumber: number; viscosity: number; pressureOffset: number; depth: number;
}
function validTaylorGreen(p: TaylorGreenParameters) {
  bound(p.density, .25, 4, 'density'); bound(p.amplitude, 0, 4, 'amplitude');
  bound(p.waveNumber, .25, 4, 'wave number'); bound(p.viscosity, 0, 1, 'kinematic viscosity');
  bound(p.pressureOffset, -4, 4, 'pressure offset'); bound(p.depth, .25, 2, 'depth');
}
/** Preserve physical weights before projecting a small exponential change.
 * At very small positive exponents the quadratic correction resolves half-ulp
 * ties that a bare -z approximation would round in the wrong direction.
 */
function rawDecayChange(p:TaylorGreenParameters,time:number,factor:number):BinaryValue {
  const z=product(factor,2,p.viscosity,p.waveNumber,p.waveNumber,time),linear=observe(z).value;
  if(linear===0)return z;
  if(linear===null||linear<1e-200)return addBinary([multiply(product(-1),z),multiply(product(.5),multiply(z,z))]);
  // This change is already representable. Dividing by rounded z and then
  // multiplying by unrounded z can spuriously push a saturated change below -1.
  return product(Math.expm1(-linear));
}
export function taylorGreenDecay(p: TaylorGreenParameters, time: number) {
  validTaylorGreen(p); bound(time, 0, 4, 'time');
  const exponent=observe(product(2,p.viscosity,p.waveNumber,p.waveNumber,time)).value;
  const energyExponent=observe(product(4,p.viscosity,p.waveNumber,p.waveNumber,time)).value;
  const roundedDecay=exponent===null?1:Math.exp(-exponent),roundedEnergyDecay=energyExponent===null?1:Math.exp(-energyExponent);
  const change=rawDecayChange(p,time,1),energyChange=rawDecayChange(p,time,2);
  return {exponent,roundedDecay,stableDecayChange:observe(change).value,roundedEnergyDecay,stableEnergyChange:observe(energyChange).value,
    amplitude:observe(product(p.amplitude,roundedDecay)).value,amplitudeChange:observe(multiply(product(p.amplitude),change)).value,
    status:exponent===null?'change-arithmetic-unavailable' as const:exponent>0&&roundedDecay===1?'change-below-rounded-equality' as const:'available' as const};
}
/** Actual SI field on a periodic square of side 2π/k, with explicitly specified depth. */
export function taylorGreenAt(p: TaylorGreenParameters, x: number, y: number, time: number) {
  const decay=taylorGreenDecay(p,time);bound(x,-32,32,'x');bound(y,-32,32,'y');
  const k=p.waveNumber,sx=Math.sin(k*x),cx=Math.cos(k*x),sy=Math.sin(k*y),cy=Math.cos(k*y);
  const amplitude=product(p.amplitude,decay.roundedDecay),square=multiply(amplitude,amplitude);
  const weighted=(...factors:number[])=>multiply(amplitude,product(...factors));
  const velocityRaw=[weighted(-1,cx,sy),weighted(sx,cy)];
  const gradientRaw=[[weighted(k,sx,sy),weighted(-1,k,cx,cy)],[weighted(k,cx,cy),weighted(-1,k,sx,sy)]];
  const pressureRaw=multiply(product(-.25,p.density),multiply(square,addBinary([product(Math.cos(2*k*x)),product(Math.cos(2*k*y))])));
  const pressureGradientRaw=[Math.sin(2*k*x),Math.sin(2*k*y)].map(v=>multiply(product(.5,p.density,k,v),square));
  const pressureAccelerationRaw=[Math.sin(2*k*x),Math.sin(2*k*y)].map(v=>multiply(product(-.5,k,v),square));
  const convectionRaw=gradientRaw.map(row=>addBinary(row.map((v,i)=>multiply(v,velocityRaw[i]))));
  const timeRaw=velocityRaw.map(v=>multiply(product(-2,p.viscosity,k,k),v));
  const laplacianRaw=velocityRaw.map(v=>multiply(product(-2,k,k),v));
  const viscousRaw=laplacianRaw.map(v=>multiply(product(p.viscosity),v));
  const residualRaw=velocityRaw.map((_,i)=>addBinary([timeRaw[i],convectionRaw[i],multiply(product(-1),pressureAccelerationRaw[i]),multiply(product(-1),viscousRaw[i])]));
  const omittedRaw=velocityRaw.map((_,i)=>addBinary([timeRaw[i],convectionRaw[i],multiply(product(-1),viscousRaw[i])]));
  const strainRaw=[gradientRaw[0][0],gradientRaw[1][1],multiply(product(.5),addBinary([gradientRaw[0][1],gradientRaw[1][0]]))];
  const strainSquared=addBinary(strainRaw.map((v,i)=>multiply(product(i===2?2:1),multiply(v,v))));
  const vector=(vs:BinaryValue[])=>vs.map(v=>observe(v).value)as FlowVector;
  return{x,y,time,decay,velocity:vector(velocityRaw),gradient:gradientRaw.map(vector)as[FlowVector,FlowVector],
    pressureChange:observe(pressureRaw).value,pressure:observe(addBinary([product(p.pressureOffset),pressureRaw])).value,
    pressureGradient:vector(pressureGradientRaw),divergence:observe(addBinary([gradientRaw[0][0],gradientRaw[1][1]])).value,
    vorticity:observe(addBinary([gradientRaw[1][0],multiply(product(-1),gradientRaw[0][1])])).value,
    convection:vector(convectionRaw),timeDerivative:vector(timeRaw),laplacian:vector(laplacianRaw),pressureAcceleration:vector(pressureAccelerationRaw),
    viscousAcceleration:vector(viscousRaw),residual:vector(residualRaw),omittedPressureResidual:vector(omittedRaw),
    kineticEnergyDensity:observe(multiply(product(.5,p.density),addBinary(velocityRaw.map(v=>multiply(v,v))))).value,
    localDissipation:observe(multiply(product(2,p.density,p.viscosity),strainSquared)).value};
}
export function taylorGreenCell(p: TaylorGreenParameters, time: number) {
  const decay = taylorGreenDecay(p, time), side = 2 * Math.PI / p.waveNumber, volume = side * side * p.depth;
  const meanRaw=product(.25,p.density,p.amplitude,p.amplitude,decay.roundedDecay,decay.roundedDecay);
  const energyRaw=multiply(product(volume),meanRaw),initialRaw=product(.25,p.density,p.amplitude,p.amplitude,volume);
  const meanKineticEnergy=observe(meanRaw).value,kineticEnergy=observe(energyRaw).value,initialEnergy=observe(initialRaw).value;
  const dissipation=observe(multiply(product(4,p.viscosity,p.waveNumber,p.waveNumber),energyRaw)).value;
  return {side,volume,mass:p.density*volume,meanKineticEnergy,kineticEnergy,dissipation,
    energyTimeDerivative:neg(dissipation),initialEnergy,stableEnergyChange:observe(multiply(initialRaw,rawDecayChange(p,time,2))).value,
    roundedEnergyChange:sum(kineticEnergy,neg(initialEnergy)),decay};
}
/** RK4 tracer in the actual unwrapped field. Trajectory is an approximation, not a streamline. */
export function taylorGreenParticle(p: TaylorGreenParameters, start: [number, number], endTime: number, steps = 256) {
  validTaylorGreen(p); bound(endTime, 0, 4, 'end time');
  for (const v of start) bound(v, -16, 16, 'initial coordinate');
  if (!Number.isInteger(steps) || steps < 1 || steps > 4096 || start.length !== 2) throw new RangeError('Invalid particle integration');
  const h = endTime / steps, points = [{ time: 0, x: start[0], y: start[1] }];
  let roundedCoordinateUpdates = 0;
  if (endTime === 0) return { status: 'integrated' as const, points, roundedCoordinateUpdates };
  if (h === 0) return { status: 'increment-arithmetic-unavailable' as const, points, roundedCoordinateUpdates };
  let x = start[0], y = start[1];
  for (let i = 0; i < steps; i++) {
    const t = i * h;
    // The tracer needs only velocity, not every stress/energy diagnostic at each
    // RK stage. Evaluate the same declared amplitude and trigonometric factors.
    const field = (px:number,py:number,pt:number):FlowVector => {
      bound(px,-32,32,'x');bound(py,-32,32,'y');
      const exponent=observe(product(2,p.viscosity,p.waveNumber,p.waveNumber,Math.min(endTime,pt))).value;
      const amplitude=product(p.amplitude,exponent===null?1:Math.exp(-exponent));
      return [observe(multiply(amplitude,product(-1,Math.cos(p.waveNumber*px),Math.sin(p.waveNumber*py)))).value,
        observe(multiply(amplitude,product(Math.sin(p.waveNumber*px),Math.cos(p.waveNumber*py)))).value];
    };
    const a = field(x, y, t); if (a.includes(null)) return { status: 'arithmetic-unavailable' as const, points, roundedCoordinateUpdates };
    const b = field(x + h * a[0]! / 2, y + h * a[1]! / 2, t + h / 2); if (b.includes(null)) return { status: 'arithmetic-unavailable' as const, points, roundedCoordinateUpdates };
    const c = field(x + h * b[0]! / 2, y + h * b[1]! / 2, t + h / 2); if (c.includes(null)) return { status: 'arithmetic-unavailable' as const, points, roundedCoordinateUpdates };
    const d = field(x + h * c[0]!, y + h * c[1]!, (i + 1) * h); if (d.includes(null)) return { status: 'arithmetic-unavailable' as const, points, roundedCoordinateUpdates };
    const dx = div(mul(h, sum(a[0], mul(2, b[0]), mul(2, c[0]), d[0])), 6), dy = div(mul(h, sum(a[1], mul(2, b[1]), mul(2, c[1]), d[1])), 6);
    if (dx === null || dy === null) return { status: 'increment-arithmetic-unavailable' as const, points, roundedCoordinateUpdates };
    const nx = x + dx, ny = y + dy;
    if (dx !== 0 && nx === x) roundedCoordinateUpdates++;
    if (dy !== 0 && ny === y) roundedCoordinateUpdates++;
    points.push({ time: (i + 1) * h, x: nx, y: ny }); x = nx; y = ny;
  }
  return { status: 'integrated' as const, points, roundedCoordinateUpdates };
}
