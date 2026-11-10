import type {Vec2} from './learning-simulations';

// The discrete controls define rational initial invariants. Trigonometry and
// implicit Kepler roots are evaluated numerically and reported as such.
const B = BigInt, ZERO = B(0), ONE = B(1);
type Rat = {n: bigint; d: bigint};
function rational(n: bigint, d = ONE): Rat {
  if (d === ZERO) throw new Error('Zero rational denominator.');
  if (d < ZERO) { n = -n; d = -d; }
  let a = n < ZERO ? -n : n, b = d;
  while (b !== ZERO) { const c = a % b; a = b; b = c; }
  return {n: n / a, d: d / a};
}
const R = (n: number, d = 1) => rational(B(n), B(d));
const add = (a: Rat, b: Rat) => rational(a.n*b.d+b.n*a.d, a.d*b.d);
const neg = (a: Rat) => rational(-a.n, a.d);
const sub = (a: Rat, b: Rat) => add(a, neg(b));
const mul = (a: Rat, b: Rat) => rational(a.n*b.n, a.d*b.d);
const div = (a: Rat, b: Rat) => rational(a.n*b.d, a.d*b.n);
const square = (a: Rat) => mul(a, a);
const sign = (a: Rat) => a.n < ZERO ? -1 : a.n > ZERO ? 1 : 0;
function expose(a: Rat) {
  return {numerator: a.n.toString(), denominator: a.d.toString(), label: a.d === ONE ? a.n.toString() : `${a.n}/${a.d}`, value: Number(a.n)/Number(a.d)};
}
const number = (a: Rat) => expose(a).value;
function controls(p: number[], ranges: readonly (readonly [number, number])[]) {
  if (p.length !== ranges.length) throw new Error('Use the stated central-force controls.');
  for (let j = 0; j < ranges.length; j++) {
    if (!Number.isInteger(p[j]) || p[j] < ranges[j][0] || p[j] > ranges[j][1]) throw new Error('Use bounded integer control codes.');
  }
}
const cross = (a: Vec2, b: Vec2) => a[0]*b[1]-a[1]*b[0];

/** Inverse-square geometry and radial energy reduction, not a time integrator.
 * Controls: [mu, 2*r0, 10*vr0, 10*vt0, 10*rQuery, mass].
 * Force is -m*mu*q/r^3 about a fixed origin. No collision continuation.
 */
export function inverseSquareReduction(p: number[]) {
  controls(p, [[1,6],[1,8],[-30,30],[-30,30],[1,120],[1,4]]);
  const [mu, rc, vrc, vtc, qc, mass] = p;
  const MU = R(mu), r0 = R(rc,2), vr = R(vrc,10), vt = R(vtc,10), radius = R(qc,10);
  const h = mul(r0,vt), h2 = square(h), epsilon = sub(div(add(square(vr),square(vt)),R(2)),div(MU,r0));
  const eccentricVector = [sub(div(mul(r0,square(vt)),MU),R(1)),neg(div(mul(r0,mul(vr,vt)),MU))];
  const e2 = add(square(eccentricVector[0]),square(eccentricVector[1]));
  const identity = sub(e2,add(R(1),div(mul(R(2),mul(epsilon,h2)),square(MU))));
  const potential = neg(div(MU,radius)), barrier = div(h2,mul(R(2),square(radius))), effective = add(potential,barrier), radialK = sub(epsilon,effective);
  const radial = sign(h) === 0, en = number(epsilon), hn = number(h), eccentricity = Math.sqrt(number(e2));
  const conic = radial ? null : sign(e2) === 0 ? 'circle' : sign(epsilon) < 0 ? 'ellipse' : sign(epsilon) === 0 ? 'parabola' : 'hyperbola';
  const radialClass = !radial ? null : sign(epsilon) < 0 ? 'bound-collision' : sign(epsilon) === 0 ? 'marginal-collision-or-escape' : 'positive-energy-collision-or-escape';
  const semiLatus = div(h2,MU), periapsis = radial ? null : number(h2)/(mu*(1+eccentricity));
  const apoapsis = sign(epsilon) < 0 ? -mu/en-(periapsis ?? 0) : null;
  const circularRadius = radial ? null : number(h2)/mu;
  const minEnergy = radial ? null : -mu*mu/(2*number(h2));
  const curvature = circularRadius === null ? null : mu/circularRadius**3;
  // Both signs are physically possible at an allowed radius. These are not a
  // selected future state; the initial radial sign and time would be needed.
  const radialSpeed = sign(radialK) < 0 ? null : Math.sqrt(2*number(radialK));
  const rMin = .75*Math.min(number(r0),number(radius),periapsis ?? Infinity), rMax = 12;
  const effectiveCurves: Vec2[][] = [[],[],[],[]];
  const radii = new Set<number>([rMin,rMax,number(r0),number(radius)]);
  for (let j=0;j<=400;j++) radii.add(rMin*(rMax/rMin)**(j/400));
  for (const v of [periapsis,apoapsis,circularRadius]) if (v !== null && v>=rMin && v<=rMax) radii.add(v);
  for (const r of [...radii].sort((a,b)=>a-b)) {
    const u = -mu/r, b = hn*hn/(2*r*r);
    [u+b,en,u,b].forEach((v,j)=>effectiveCurves[j].push([r,v]));
  }
  // The spatial geometry is clipped at a radius, rather than joining through
  // hyperbolic asymptotes or pretending a negative polar radius is allowed.
  const spatialRadius = 12, spatial: Vec2[] = [];
  const argumentOfPeriapsis = radial || eccentricity === 0 ? null : Math.atan2(number(eccentricVector[1]),number(eccentricVector[0]));
  let anomalyLimit: number|null = null;
  const clipped = !radial && (apoapsis === null || apoapsis>spatialRadius);
  if (!radial) {
    const phi = argumentOfPeriapsis ?? 0;
    const bound = eccentricity === 0 ? -1 : (number(semiLatus)/spatialRadius-1)/eccentricity;
    anomalyLimit = Math.acos(Math.max(-1,Math.min(1,bound)));
    const anomalies = new Set<number>([0,-anomalyLimit,anomalyLimit]);
    for(let j=0;j<=800;j++) anomalies.add(-anomalyLimit+2*anomalyLimit*j/800);
    // Explicit initial-point sample avoids losing it in very narrow branches.
    const initialAnomaly = Math.atan2(Math.sin(-phi),Math.cos(-phi));
    if (Math.abs(initialAnomaly)<=anomalyLimit+1e-12) anomalies.add(initialAnomaly);
    for(const f of [...anomalies].sort((a,b)=>a-b)) {
      const denominator = 1+eccentricity*Math.cos(f);
      if (denominator<=0) throw new Error('Nonpositive conic denominator inside the declared finite geometry window.');
      const r = number(semiLatus)/denominator;
      spatial.push([r*Math.cos(f+phi),r*Math.sin(f+phi)]);
    }
  }
  return {
    mass, mu, initialRadius: expose(r0), initialRadialVelocity: expose(vr), initialTangentialVelocity: expose(vt),
    initialPosition: [number(r0),0] as Vec2, initialVelocity: [number(vr),number(vt)] as Vec2,
    specificEnergy: expose(epsilon), totalEnergy: expose(mul(R(mass),epsilon)), specificAngularMomentum: expose(h), angularMomentum: expose(mul(R(mass),h)),
    eccentricityVector: eccentricVector.map(expose), eccentricitySquared: expose(e2), eccentricity, eccentricityIdentityResidual: expose(identity),
    radial, radialClass, conic, semiLatusRectum: radial ? null : expose(semiLatus), argumentOfPeriapsis, periapsis, apoapsis,
    circularRadius, minimumSpecificEnergy: minEnergy, circularStabilityCurvature: curvature,
    query: {radius: expose(radius), physicalPotential: expose(potential), angularBarrier: expose(barrier), effectivePotential: expose(effective), radialKineticPerMass: expose(radialK), allowed: radialSpeed !== null, turning: sign(radialK)===0, radialSpeed, radialVelocityBranches: radialSpeed === null ? [] : radialSpeed === 0 ? [0] : [-radialSpeed,radialSpeed], tangentialVelocity: expose(div(h,radius))},
    radialWindow: [rMin,rMax] as [number,number], effectiveCurves, spatialGeometry: spatial, spatialRadius, clipped, anomalyLimit,
    direction: sign(h), geometryIsTimeTrajectory: false as const, collisionContinuation: false as const,
  };
}

/** Solve monotone elliptic Kepler equation with a guaranteed mathematical
 * bracket [M-e,M+e]. The input M and output E retain complete revolutions.
 * Finite controls use |M|<140, but the standalone solver permits |M|<=1e6.
 * Root/bracket stopping tolerances scale with floating-point spacing.
 */
export function solveEllipticKepler(meanAnomaly: number, eccentricity: number) {
  const M = meanAnomaly, e = eccentricity;
  if (!Number.isFinite(M) || Math.abs(M)>1e6 || !Number.isFinite(e) || e<0 || e>=1) throw new Error('Use finite elliptic Kepler inputs with 0 <= e < 1 and |M| <= 1e6.');
  if (e===0 || M===0) return {eccentricAnomaly:M, residual:0, iterations:0, bracket:[M-e,M+e] as [number,number]};
  // Avoid cancellation in E-sin(E) for the near-parabolic, small-anomaly
  // limit of this standalone solver. The teaching controls keep e<=.95.
  const residual = (x: number) => {
    if (Math.abs(x)<.25) {
      let term=x*x*x/6, difference=term;
      for(let k=2;k<=9;k++) {term *= -x*x/((2*k)*(2*k+1)); difference += term;}
      return (1-e)*x+e*difference-M;
    }
    return (x-M)-e*Math.sin(x);
  };
  let lo=M-e, hi=M+e, E=M;
  const tolerance = 4*Number.EPSILON*Math.max(1,Math.abs(M));
  for(let iterations=1;iterations<=100;iterations++) {
    const f=residual(E);
    if (Math.abs(f)<=tolerance*(1-e)) return {eccentricAnomaly:E,residual:f,iterations,bracket:[lo,hi] as [number,number]};
    if (f>0) hi=E; else lo=E;
    const next = E-f/((1-e)+2*e*Math.sin(E/2)**2);
    const candidate = Number.isFinite(next) && next>lo && next<hi ? next : (lo+hi)/2;
    if (candidate===E || hi-lo<=tolerance) {
      const root=(lo+hi)/2;
      return {eccentricAnomaly:root,residual:residual(root),iterations,bracket:[lo,hi] as [number,number]};
    }
    E=candidate;
  }
  throw new Error('Elliptic Kepler solve did not converge.');
}

/** Actual bound orbital dynamics starting at periapsis (chosen phase for e=0).
 * [mu, 2*a, 20*e, orientation*pi/12, retrograde(0/1), 10*t, mass].
 * All UI combinations have a>0 and 0<=e<=.95, hence r>=a(1-e)>0.
 */
export function ellipticKeplerMotion(p: number[]) {
  controls(p, [[1,6],[1,8],[0,19],[-12,12],[0,1],[0,200],[1,4]]);
  const [mu,ac,ec,orientationCode,retrograde,tc,mass] = p;
  const a=ac/2, e=ec/20, direction=retrograde ? -1 : 1, time=tc/10;
  const orientation=orientationCode*Math.PI/12, c=Math.cos(orientation), s=Math.sin(orientation), b=a*Math.sqrt(1-e*e), n=Math.sqrt(mu/a**3), period=2*Math.PI/n;
  const h=direction*Math.sqrt(mu*a*(1-e*e)), epsilon=-mu/(2*a);
  const rotate = ([x,y]: Vec2): Vec2 => [c*x-s*y,s*x+c*y];
  const stateFromE = (E: number, t: number, root: ReturnType<typeof solveEllipticKepler>) => {
    const co=Math.cos(E), si=Math.sin(E), radius=a*(1-e*co), Edot=direction*n/(1-e*co);
    const position=rotate([a*(co-e),b*si]), velocity=rotate([-a*si*Edot,b*co*Edot]);
    const acceleration: Vec2 = position.map(x=>-mu*x/radius**3) as Vec2;
    const kinetic=mass*(velocity[0]**2+velocity[1]**2)/2, potential=-mass*mu/radius;
    const angularMomentum=mass*cross(position,velocity), sweptArea=h*t/2;
    // This independently evaluated focus-sector expression retains complete
    // turns. It should match h*t/2 to root/roundoff error.
    const geometricSweptArea=a*b*(E-e*si)/2;
    const focusRadius=Math.hypot(...position);
    return {time:t, meanAnomaly:direction*n*t, eccentricAnomaly:E, radius, position, velocity, acceleration,
      force:acceleration.map(v=>mass*v) as Vec2, speed:Math.hypot(...velocity), radialVelocity:(position[0]*velocity[0]+position[1]*velocity[1])/radius,
      kinetic,potential,totalEnergy:kinetic+potential,angularMomentum,arealVelocity:cross(position,velocity)/2,sweptArea,geometricSweptArea,
      rootResidual:root.residual,rootIterations:root.iterations,rootBracket:root.bracket,
      energyResidual:kinetic+potential-mass*epsilon,angularResidual:angularMomentum-mass*h,areaResidual:geometricSweptArea-sweptArea,radiusResidual:focusRadius-radius};
  };
  const state = (t: number) => {
    const root=solveEllipticKepler(direction*n*t,e);
    return stateFromE(root.eccentricAnomaly,t,root);
  };
  const query=state(time), initial=state(0), geometry: Vec2[]=[];
  for(let j=0;j<=512;j++) {const E=2*Math.PI*j/512;geometry.push(rotate([a*(Math.cos(E)-e),b*Math.sin(E)]));}
  // Uniform eccentric anomaly resolves the rapid periapsis passage even for
  // e=.95; each abscissa is its actual Kepler time. Include query and every
  // exact peri/apo E=k*pi. No uniform-angle or uniform-time ellipse animation.
  const endMean=n*20, endE=Math.abs(solveEllipticKepler(endMean,e).eccentricAnomaly);
  const samples=new Set<number>([0,endE,Math.abs(query.eccentricAnomaly)]);
  const steps=Math.max(256,Math.ceil(endE/(Math.PI/128)));
  for(let j=0;j<=steps;j++) samples.add(endE*j/steps);
  for(let k=1;k*Math.PI<endE;k++) samples.add(k*Math.PI);
  const curves: Vec2[][]=Array.from({length:9},()=>[]), actualPath: Vec2[]=[];
  for(const absE of [...samples].sort((x,y)=>x-y)) {
    const E=direction*absE, t=absE===endE ? 20 : absE===Math.abs(query.eccentricAnomaly) ? time : (absE-e*Math.sin(absE))/n;
    const r=stateFromE(E,t,{eccentricAnomaly:E,residual:E-e*Math.sin(E)-direction*n*t,iterations:0,bracket:[E,E]});
    [r.radius,r.speed,r.kinetic,r.potential,r.totalEnergy,r.angularMomentum,r.sweptArea,r.geometricSweptArea,r.eccentricAnomaly].forEach((v,j)=>curves[j].push([t,v]));
    if(absE<=Math.abs(query.eccentricAnomaly)) actualPath.push(r.position);
  }
  // Root-derived query time can round by a few ulps; the last spatial point
  // is the actual queried state, never an extrapolated whole-orbit endpoint.
  if(!actualPath.length || actualPath[actualPath.length-1].some((v,j)=>v!==query.position[j])) actualPath.push([...query.position] as Vec2);
  return {mu,mass,semiMajorAxis:a,eccentricity:e,semiMinorAxis:b,orientation,orientationCode,direction,time,meanMotion:n,period,
    periapsis:a*(1-e),apoapsis:a*(1+e),specificEnergy:epsilon,specificAngularMomentum:h,totalEnergy:mass*epsilon,angularMomentum:mass*h,
    arealVelocity:h/2,periodSquaredOverAxisCubed:4*Math.PI**2/mu,
    initial,query,geometry,actualPath,curves,focus:[0,0] as Vec2,ellipseCenter:rotate([-a*e,0]),
    collisionContinuation:false as const,scope:'bound-elliptic' as const};
}
