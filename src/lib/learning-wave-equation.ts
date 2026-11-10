/** Linear transverse waves, SI units. No time stepping or plotting-grid energy
 * estimates: periodic chains evolve spectrally and compact pulses analytically.
 */
export interface WaveMedium {density:number; tension:number;}
export interface ChainMedium extends WaveMedium {length:number; particles:number;}
export interface WaveHarmonic {mode:number; positionCos:number; positionSin:number; velocityCos:number; velocitySin:number;}
export type WaveBoundary='whole-line'|'fixed'|'free';
export interface PulseMedium extends WaveMedium {
  amplitude:number; velocityAmplitude:number; width:number; center:number;
  direction:'left'|'right'|'split'; boundary:WaveBoundary;
}
function bounded(x:number,min:number,max:number,name:string){
  if(!Number.isFinite(x)||x<min||x>max)throw new RangeError(`${name} outside supported range.`);
}
function medium(s:WaveMedium){bounded(s.density,.25,4,'Line density');bounded(s.tension,.25,4,'Tension');return Math.sqrt(s.tension/s.density);}
function clock(t:number){bounded(t,0,64,'Time');}
function chainData(s:ChainMedium,h:WaveHarmonic[]){
  const c=medium(s);bounded(s.length,4,16,'Length');
  if(![8,16,32,64,128].includes(s.particles))throw new RangeError('Use 8,16,32,64 or128 particles.');
  if(!Array.isArray(h)||h.length<1||h.length>32)throw new RangeError('Supply1–32 distinct resolved harmonics.');
  const seen=new Set<number>();
  for(let i=0;i<h.length;i++){
    const a=h[i];if(!a||!Number.isInteger(a.mode)||a.mode<0||a.mode>=s.particles/2||a.mode>31||seen.has(a.mode))throw new RangeError('Modes must be distinct,nonnegative and below Nyquist.');
    seen.add(a.mode);for(const x of [a.positionCos,a.positionSin,a.velocityCos,a.velocitySin])bounded(x,-1,1,'Initial coefficient');
    if(a.mode===0&&(a.positionSin!==0||a.velocitySin!==0))throw new RangeError('The zero mode has no sine coordinate.');
  }
  return c;
}
const sinc=(x:number)=>x===0?1:Math.sin(x)/x;
function evolution(a:number,v:number,w:number,t:number){return {q:a*Math.cos(w*t)+v*t*sinc(w*t),v:v*Math.cos(w*t)-a*w*Math.sin(w*t)};}

/** The trigonometric interpolant is evaluated continuously; only the returned
 * particle arrays are actual discrete masses. The zero mode drifts affinely.
 */
export function periodicWaveChain(s:ChainMedium,h:WaveHarmonic[],t:number){
  const c=chainData(s,h);clock(t);
  const spacing=s.length/s.particles,mass=s.density*spacing,stiffness=s.tension/spacing;
  const spectrum=h.map(a=>{const waveNumber=2*Math.PI*a.mode/s.length,continuumFrequency=c*waveNumber,frequency=2*c/spacing*Math.sin(waveNumber*spacing/2);
    return {...a,waveNumber,frequency,continuumFrequency,frequencyRatio:a.mode===0?null:sinc(waveNumber*spacing/2),
      discreteCos:evolution(a.positionCos,a.velocityCos,frequency,t),discreteSin:evolution(a.positionSin,a.velocitySin,frequency,t),
      continuumCos:evolution(a.positionCos,a.velocityCos,continuumFrequency,t),continuumSin:evolution(a.positionSin,a.velocitySin,continuumFrequency,t)};});
  const x=Array.from({length:s.particles},(_,i)=>i*spacing);
  const field=(at:number,discrete:boolean)=>{
    let displacement=0,velocity=0,gradient=0,curvature=0,acceleration=0;
    for(const a of spectrum){const co=Math.cos(a.waveNumber*at),si=Math.sin(a.waveNumber*at),p=discrete?a.discreteCos:a.continuumCos,q=discrete?a.discreteSin:a.continuumSin,w=discrete?a.frequency:a.continuumFrequency;
      const u=p.q*co+q.q*si;displacement+=u;velocity+=p.v*co+q.v*si;gradient+=a.waveNumber*(-p.q*si+q.q*co);curvature-=a.waveNumber**2*u;acceleration-=w*w*u;}
    const kineticDensity=s.density*velocity**2/2,strainDensity=s.tension*gradient**2/2;
    return {x:at,displacement,velocity,gradient,curvature,acceleration,kineticDensity,strainDensity,energyDensity:kineticDensity+strainDensity,flux:-s.tension*velocity*gradient};
  };
  const discrete=x.map(at=>field(at,true)),continuum=x.map(at=>field(at,false)),position=discrete.map(a=>a.displacement),velocity=discrete.map(a=>a.velocity);
  const force=x.map((_,i)=>stiffness*(position[(i+1)%s.particles]-2*position[i]+position[(i+s.particles-1)%s.particles]));
  const kinetic=velocity.map(v=>mass*v*v/2),linkEnergy=position.map((q,i)=>stiffness*(position[(i+1)%s.particles]-q)**2/2);
  const linkFlux=position.map((q,i)=>-stiffness*(position[(i+1)%s.particles]-q)*(velocity[(i+1)%s.particles]+velocity[i])/2);
  const assignedEnergy=kinetic.map((v,i)=>v+(linkEnergy[i]+linkEnergy[(i+s.particles-1)%s.particles])/2);
  const assignedEnergyRate=velocity.map((v,i)=>{const j=(i+1)%s.particles,k=(i+s.particles-1)%s.particles;return v*force[i]+stiffness*((position[i]-position[k])*(v-velocity[k])+(position[j]-position[i])*(velocity[j]-v))/2;});
  const modalEnergy=spectrum.map(a=>s.density*s.length/(a.mode===0?2:4)*(a.velocityCos**2+a.velocitySin**2+a.frequency**2*(a.positionCos**2+a.positionSin**2)));
  const continuumModalEnergy=spectrum.map(a=>s.density*s.length/(a.mode===0?2:4)*(a.velocityCos**2+a.velocitySin**2+a.continuumFrequency**2*(a.positionCos**2+a.positionSin**2)));
  return {time:t,c,spacing,mass,stiffness,x,spectrum,position,velocity,force,kinetic,linkEnergy,assignedEnergy,assignedEnergyRate,linkFlux,continuum,
    energy:assignedEnergy.reduce((a,b)=>a+b,0),modalEnergy,continuumModalEnergy,continuumEnergy:continuumModalEnergy.reduce((a,b)=>a+b,0),
    momentum:mass*velocity.reduce((a,b)=>a+b,0),rmsDisplacementError:Math.sqrt(position.reduce((sum,q,i)=>sum+(q-continuum[i].displacement)**2,0)/s.particles),
    initialSlopeBound:spectrum.reduce((sum,a)=>sum+a.waveNumber*Math.hypot(a.positionCos,a.positionSin),0)};
}

/** Compact C² profile and its integral from minus infinity. */
export function compactWaveBump(z:number){
  if(!Number.isFinite(z))throw new RangeError('Finite bump coordinate required.');
  if(Math.abs(z)>=1)return {value:0,first:0,second:0,integral:z<=-1?0:32/35};
  return {value:(1-z*z)**3,first:-6*z*(1-z*z)**2,second:-6+36*z*z-30*z**4,
    integral:z-z**3+3*z**5/5-z**7/7+16/35};
}
function pulseData(s:PulseMedium){
  const c=medium(s);bounded(s.amplitude,-.25,.25,'Displacement amplitude');bounded(s.velocityAmplitude,-1,1,'Velocity amplitude');bounded(s.width,.25,2,'Pulse width');bounded(s.center,-4,4,'Pulse center');
  if(!['left','right','split'].includes(s.direction)||!['whole-line','fixed','free'].includes(s.boundary))throw new RangeError('Unknown wave direction or boundary.');
  if(s.boundary!=='whole-line'&&s.center<s.width+.25)throw new RangeError('Half-line initial support must stay away from the boundary.');
  return c;
}
function pulseInitial(s:PulseMedium,c:number,x:number){
  const parity=s.boundary==='fixed'?-1:1,reflected=s.boundary!=='whole-line'&&x<0,y=reflected?-x:x,b=compactWaveBump((y-s.center)/s.width);
  const sign=reflected?parity:1,derivativeSign=reflected?-parity:1,d=s.direction==='left'?1:s.direction==='right'?-1:0;
  const f=s.amplitude*b.value,fp=s.amplitude*b.first/s.width,fpp=s.amplitude*b.second/s.width**2;
  const g=d*c*fp+s.velocityAmplitude*b.value,gp=d*c*fpp+s.velocityAmplitude*b.first/s.width;
  // Integral of g, not derivative of an extended displacement. The two initial
  // fields must have the same parity, including initially one-way wave data.
  const primitive=d*c*f+s.velocityAmplitude*s.width*b.integral;
  return {f:sign*f,fp:derivativeSign*fp,fpp:sign*fpp,g:sign*g,gp:derivativeSign*gp,primitive:reflected?-parity*primitive:primitive};
}
function pulsePoint(s:PulseMedium,c:number,x:number,t:number){
  const l=pulseInitial(s,c,x-c*t),r=pulseInitial(s,c,x+c*t);
  const rightDisplacement=l.f/2-l.primitive/(2*c),leftDisplacement=r.f/2+r.primitive/(2*c);
  const displacement=(l.f+r.f)/2+(r.primitive-l.primitive)/(2*c),velocity=c*(r.fp-l.fp)/2+(r.g+l.g)/2,
    gradient=(l.fp+r.fp)/2+(r.g-l.g)/(2*c),curvature=(l.fpp+r.fpp)/2+(r.gp-l.gp)/(2*c),acceleration=c*c*(l.fpp+r.fpp)/2+c*(r.gp-l.gp)/2;
  const kineticDensity=s.density*velocity**2/2,strainDensity=s.tension*gradient**2/2,rightEnergyDensity=s.density*(velocity-c*gradient)**2/4,leftEnergyDensity=s.density*(velocity+c*gradient)**2/4;
  return {x,time:t,displacement,velocity,gradient,curvature,acceleration,kineticDensity,strainDensity,energyDensity:kineticDensity+strainDensity,flux:-s.tension*velocity*gradient,rightDisplacement,leftDisplacement,rightEnergyDensity,leftEnergyDensity,
    initialDependencyInterval:[x-c*t,x+c*t] as [number,number]};
}
export function wavePulsePoint(s:PulseMedium,x:number,t:number){
  const c=pulseData(s);clock(t);bounded(x,-512,512,'Position');if(s.boundary!=='whole-line'&&x<0)throw new RangeError('Half-line query must have x>=0.');return pulsePoint(s,c,x,t);
}
// Eight-point Gauss-Legendre integrates degree<=15 exactly. On each interval
// delimited by moving compact-support edges, energy and flux are degree<=12.
const gaussX=[.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363];
const gaussW=[.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
function integrate(f:(x:number)=>number,a:number,b:number,cuts:number[]){
  const points=[a,...cuts.filter(x=>x>a&&x<b),b].sort((x,y)=>x-y);let sum=0;
  for(let i=1;i<points.length;i++){const mid=(points[i-1]+points[i])/2,half=(points[i]-points[i-1])/2;if(half===0)continue;
    for(let j=0;j<4;j++)sum+=half*gaussW[j]*(f(mid-half*gaussX[j])+f(mid+half*gaussX[j]));}
  return sum;
}
function supportEdges(s:PulseMedium){const edges=[s.center-s.width,s.center+s.width];return s.boundary==='whole-line'?edges:[...edges,...edges.map(x=>-x)];}
export function wavePulseEnergyBalance(s:PulseMedium,window:[number,number],t:number){
  const c=pulseData(s);clock(t);
  if(!Array.isArray(window)||window.length!==2)throw new RangeError('Two window endpoints required.');
  const [a,b]=window;bounded(a,-128,128,'Window left');bounded(b,-128,128,'Window right');if(b<=a||s.boundary!=='whole-line'&&a<0)throw new RangeError('Invalid observation window.');
  const edges=supportEdges(s),energy=(left:number,right:number,time:number)=>integrate(x=>pulsePoint(s,c,x,time).energyDensity,left,right,edges.flatMap(y=>[y-c*time,y+c*time]));
  const flow=(x:number)=>integrate(time=>pulsePoint(s,c,x,time).flux,0,t,edges.flatMap(y=>[(x-y)/c,(y-x)/c]));
  const energy0=energy(a,b,0),energyNow=energy(a,b,t),integratedInflow=flow(a),integratedOutflow=flow(b),netBoundaryWork=integratedInflow-integratedOutflow;
  const radius=Math.abs(s.center)+s.width+c*t+1,wholeLeft=s.boundary==='whole-line'?-radius:0;
  return {time:t,window,energy0,energyNow,integratedInflow,integratedOutflow,netBoundaryWork,residual:energyNow-energy0-netBoundaryWork,
    totalEnergy0:energy(wholeLeft,radius,0),totalEnergyNow:energy(wholeLeft,radius,t),
    initialSlopeBound:96/(25*Math.sqrt(5))*Math.abs(s.amplitude)/s.width};
}
