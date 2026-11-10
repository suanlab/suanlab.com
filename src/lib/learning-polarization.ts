/** Normalized transverse plane-wave polarization, exp(-i omega t), +z, right-
 * handed(x,y,z). No Node/data imports. Intensity omits medium calibration.
 * Stokes=(I,Q,U,V)=(Cxx+Cyy,Cxx-Cyy,2Re Cxy,-2Im Cxy).
 */
export type PolarComplex=[number,number];
export type JonesVector=[PolarComplex,PolarComplex];
export type JonesMatrix=[JonesVector,JonesVector];
export type StokesVector=[number,number,number,number];
export interface PolarizationCoherency{xx:number;yy:number;xy:PolarComplex;}
export interface PolarizationPreparation{amplitudeX:number;amplitudeY:number;relativePhase:number;globalPhase:number;}
export type PolarizationOrder='retarder-only'|'analyzer-only'|'retarder-then-analyzer'|'analyzer-then-retarder';
export interface PolarizationOptics{retarderAngle:number;retardance:number;analyzerAngle:number;order:PolarizationOrder;}
export interface JonesPolarizationExperiment extends PolarizationPreparation,PolarizationOptics{observationPhase:number;}
export interface PolarizationPopulation extends PolarizationPreparation{weight:number;}
export interface StokesPolarizationExperiment extends PolarizationOptics{population:PolarizationPopulation[];retainedPolarization:number;}
const add=(a:PolarComplex,b:PolarComplex):PolarComplex=>[a[0]+b[0],a[1]+b[1]];
const mul=(a:PolarComplex,b:PolarComplex):PolarComplex=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const conj=(a:PolarComplex):PolarComplex=>[a[0],-a[1]];
const scale=(a:PolarComplex,s:number):PolarComplex=>[a[0]*s,a[1]*s];
const abs2=(a:PolarComplex)=>a[0]*a[0]+a[1]*a[1];
const identity=():JonesMatrix=>[[[1,0],[0,0]],[[0,0],[1,0]]];
function bounded(x:number,min:number,max:number,name:string){if(!Number.isFinite(x)||x<min||x>max)throw new RangeError(`${name} outside supported range.`);}
/** Only EXACT supplied multiples of pi/2 receive exact quadrantal values.
 * Neighboring angles are never snapped to a dark or singular configuration.
 */
function trig(angle:number){const n=Math.round(angle/(Math.PI/2));if(angle===n*(Math.PI/2)){const i=(n%4+4)%4;return{co:[1,0,-1,0][i],si:[0,1,0,-1][i]};}return{co:Math.cos(angle),si:Math.sin(angle)};}
function vectorData(e:JonesVector){if(!Array.isArray(e)||e.length!==2)throw new RangeError('Two complex field components required.');for(let i=0;i<2;i++){if(!Array.isArray(e[i])||e[i].length!==2)throw new RangeError('Complex components require real and imaginary parts.');for(let j=0;j<2;j++)bounded(e[i][j],-16,16,'Jones component');}}
function matrixData(j:JonesMatrix){if(!Array.isArray(j)||j.length!==2)throw new RangeError('A two-row Jones matrix is required.');vectorData(j[0]);vectorData(j[1]);}
function apply(j:JonesMatrix,e:JonesVector):JonesVector{return[add(mul(j[0][0],e[0]),mul(j[0][1],e[1])),add(mul(j[1][0],e[0]),mul(j[1][1],e[1]))];}
function product(a:JonesMatrix,b:JonesMatrix):JonesMatrix{return[[add(mul(a[0][0],b[0][0]),mul(a[0][1],b[1][0])),add(mul(a[0][0],b[0][1]),mul(a[0][1],b[1][1]))],[add(mul(a[1][0],b[0][0]),mul(a[1][1],b[1][0])),add(mul(a[1][0],b[0][1]),mul(a[1][1],b[1][1]))]];}
function adjoint(j:JonesMatrix):JonesMatrix{return[[conj(j[0][0]),conj(j[1][0])],[conj(j[0][1]),conj(j[1][1])]];}
function asMatrix(c:PolarizationCoherency):JonesMatrix{return[[[c.xx,0],c.xy],[conj(c.xy),[c.yy,0]]];}
function outer(e:JonesVector):PolarizationCoherency{return{xx:abs2(e[0]),yy:abs2(e[1]),xy:mul(e[0],conj(e[1]))};}
function preparation(s:PolarizationPreparation):JonesVector{
 bounded(s.amplitudeX,0,2,'X amplitude');bounded(s.amplitudeY,0,2,'Y amplitude');bounded(s.relativePhase,-4*Math.PI,4*Math.PI,'Relative phase');bounded(s.globalPhase,-4*Math.PI,4*Math.PI,'Global phase');
 const x=trig(s.globalPhase),y=trig(s.globalPhase+s.relativePhase);return[[s.amplitudeX*x.co,s.amplitudeX*x.si],[s.amplitudeY*y.co,s.amplitudeY*y.si]];
}
export function polarizationRetarder(angle:number,retardance:number):JonesMatrix{
 bounded(angle,-Math.PI,Math.PI,'Retarder angle');bounded(retardance,-2*Math.PI,2*Math.PI,'Retardance');const{co,si}=trig(2*angle),d=trig(retardance),p:PolarComplex=[d.co,d.si];
 const mean=scale(add([1,0],p),.5),difference=scale(add([1,0],scale(p,-1)),.5),cross=scale(difference,si);return[[add(mean,scale(difference,co)),cross],[cross,add(mean,scale(difference,-co))]];
}
export function polarizationAnalyzer(angle:number):JonesMatrix{
 bounded(angle,-Math.PI,Math.PI,'Analyzer angle');const{co:c,si:s}=trig(angle);return[[[c*c,0],[c*s,0]],[[c*s,0],[s*s,0]]];
}
function optics(s:PolarizationOptics){
 const retarder=polarizationRetarder(s.retarderAngle,s.retardance),analyzer=polarizationAnalyzer(s.analyzerAngle);
 if(!['retarder-only','analyzer-only','retarder-then-analyzer','analyzer-then-retarder'].includes(s.order))throw new RangeError('Unsupported optical element order.');
 const first=s.order.startsWith('retarder')?retarder:analyzer;
 const second=s.order==='retarder-then-analyzer'?analyzer:s.order==='analyzer-then-retarder'?retarder:identity();
 return{retarder,analyzer,first,second,combined:product(second,first)};
}
export function coherencyFromStokes(s:StokesVector):PolarizationCoherency{
 if(!Array.isArray(s)||s.length!==4)throw new RangeError('Four Stokes components required.');for(let i=0;i<4;i++)bounded(s[i],-128,128,'Stokes component');
 if(s[0]<0||Math.hypot(s[1],s[2],s[3])>s[0])throw new RangeError('Stokes vector lies outside the physical cone.');
 return{xx:(s[0]+s[1])/2,yy:(s[0]-s[1])/2,xy:[s[2]/2,-s[3]/2]};
}
export function polarizationCoherencyState(c:PolarizationCoherency){
 bounded(c.xx,0,128,'Cxx');bounded(c.yy,0,128,'Cyy');if(!Array.isArray(c.xy)||c.xy.length!==2)throw new RangeError('Cxy requires two components.');bounded(c.xy[0],-128,128,'Re Cxy');bounded(c.xy[1],-128,128,'Im Cxy');
 bounded(c.xx+c.yy,0,128,'Total coherency intensity');
 const intensity=c.xx+c.yy,cross=Math.hypot(c.xy[0],c.xy[1]),limit=Math.sqrt(c.xx)*Math.sqrt(c.yy),roundoff=64*Number.EPSILON;
 if(cross>limit*(1+roundoff))throw new RangeError('Coherency matrix is not positive semidefinite.');
 const stokes:StokesVector=[intensity,c.xx-c.yy,2*c.xy[0],-2*c.xy[1]],linear=Math.hypot(stokes[1],stokes[2]),polarized=Math.hypot(linear,stokes[3]);
 // C is preserved exactly. A normalized degree is corrected ONLY within the
 // floating-point tolerance after PSD validation, never for invalid raw Stokes.
 const rawDegree=intensity===0?null:polarized/intensity;
 const degree=rawDegree===null?null:rawDegree>1&&rawDegree<=1+roundoff?1:rawDegree;
 return{coherency:{xx:c.xx,yy:c.yy,xy:[...c.xy]as PolarComplex},stokes,intensity,degree,
  degreeLinear:intensity===0?null:Math.min(1,linear/intensity),degreeCircular:intensity===0?null:stokes[3]/intensity,
  orientation:linear<=roundoff*intensity?null:Math.atan2(stokes[2],stokes[1])/2,
  polarizedEllipticity:polarized<=roundoff*intensity?null:Math.atan2(stokes[3],linear)/2,
  normalizedStokes:intensity===0?null:stokes.slice(1).map(v=>v/intensity),
  determinant:c.xx*c.yy-cross*cross,psdRoundoffTolerance:roundoff,degreeRoundoffCorrection:rawDegree===null?0:rawDegree-degree!};
}
function jonesState(e:JonesVector){
 const state=polarizationCoherencyState(outer(e)),i=state.intensity,linear=Math.hypot(state.stokes[1],state.stokes[2]),major=Math.sqrt((i+linear)/2);
 const minor=major===0?0:Math.abs(state.stokes[3])/(2*major),cross=mul(conj(e[0]),e[1]);
 return{...state,jones:e.map(z=>[...z])as JonesVector,majorAxis:major,minorAxis:minor,
  ellipticityRatio:major===0?null:Math.sign(state.stokes[3])*minor/major,
  relativePhase:abs2(e[0])===0||abs2(e[1])===0?null:Math.atan2(cross[1],cross[0])};
}
const realField=(e:JonesVector,phase:number):[number,number]=>{const{co,si}=trig(phase);return[e[0][0]*co+e[0][1]*si,e[1][0]*co+e[1][1]*si];};
export function jonesPolarizationObservation(s:JonesPolarizationExperiment){
 bounded(s.observationPhase,0,2*Math.PI,'Observation temporal phase');const e=preparation(s),o=optics(s),middle=apply(o.first,e),out=apply(o.second,middle);
 const input=jonesState(e),intermediate=jonesState(middle),output=jonesState(out);
 return{input,intermediate,output,matrices:o,observationPhase:s.observationPhase,
  observed:{input:realField(e,s.observationPhase),intermediate:realField(middle,s.observationPhase),output:realField(out,s.observationPhase)},
  transmittance:input.intensity===0?null:output.intensity/input.intensity,
  meanRealSquare:output.intensity/2,normalization:'sum-of-squared-complex-transverse-amplitudes' as const,
  convention:'exp(-i omega t),right-handed(x,y,z),S3=-2Im(Cxy)' as const};
}
/** Full display trajectory reuses the same validated observation model. */
export function jonesPolarizationExperiment(s:JonesPolarizationExperiment){
 const q=jonesPolarizationObservation(s),e=q.input.jones,middle=q.intermediate.jones,out=q.output.jones;
 const trajectory=Array.from({length:129},(_,i)=>{const phase=2*Math.PI*i/128;return{phase,input:realField(e,phase),intermediate:realField(middle,phase),output:realField(out,phase)};});
 const analyzerScan=Array.from({length:129},(_,i)=>{const angle=Math.PI*i/128,c=outer(apply(polarizationAnalyzer(angle),out));return{angle,intensity:c.xx+c.yy};});
 return{...q,trajectory,analyzerScan};
}
const pauli:JonesMatrix[]=[identity(),[[[1,0],[0,0]],[[0,0],[-1,0]]],[[[0,0],[1,0]],[[1,0],[0,0]]],[[[0,0],[0,-1]],[[0,1],[0,0]]]];
export function polarizationMueller(j:JonesMatrix){
 matrixData(j);const dagger=adjoint(j);return pauli.map(a=>pauli.map(b=>{const m=product(product(product(a,j),b),dagger);return(m[0][0][0]+m[1][1][0])/2;}));
}
function average(population:{field:JonesVector;probability:number}[]):PolarizationCoherency{
 let xx=0,yy=0,xy:PolarComplex=[0,0];for(const r of population){const c=outer(r.field);xx+=r.probability*c.xx;yy+=r.probability*c.yy;xy=add(xy,scale(c.xy,r.probability));}return{xx,yy,xy};
}
function transformed(c:PolarizationCoherency,j:JonesMatrix):PolarizationCoherency{const m=product(product(j,asMatrix(c)),adjoint(j));return{xx:m[0][0][0],yy:m[1][1][0],xy:m[0][1]};}
function analyzerMeasurements(population:{field:JonesVector;probability:number}[]){
 const a:JonesVector[]=[[[1,0],[0,0]],[[0,0],[1,0]],[[Math.SQRT1_2,0],[Math.SQRT1_2,0]],[[Math.SQRT1_2,0],[-Math.SQRT1_2,0]],[[Math.SQRT1_2,0],[0,Math.SQRT1_2]],[[Math.SQRT1_2,0],[0,-Math.SQRT1_2]]];
 // Average positive realized analyzer intensities, avoiding cancellation in
 // an expanded quadratic form at a dark channel.
 const values=a.map(v=>population.reduce((sum,r)=>sum+r.probability*abs2(add(mul(conj(v[0]),r.field[0]),mul(conj(v[1]),r.field[1]))),0));
 const[h,v,d,a45,left,right]=values;return{names:['H','V','D','A','S3+','S3-'],values,reconstructed:[h+v,h-v,d-a45,left-right]as StokesVector};
}
export function stokesPolarizationObservation(s:StokesPolarizationExperiment){
 bounded(s.retainedPolarization,0,1,'Retained polarization');if(!Array.isArray(s.population)||s.population.length<1||s.population.length>16)throw new RangeError('Supply1–16 phase realizations.');
 let total=0;const entries=[];for(let i=0;i<s.population.length;i++){const r=s.population[i];if(!r)throw new RangeError('Sparse populations are invalid.');bounded(r.weight,0,16,'Population weight');total+=r.weight;entries.push({field:preparation(r),weight:r.weight});}if(total===0)throw new RangeError('Positive total population weight required.');
 const population=entries.map(r=>({field:r.field,probability:r.weight/total})),o=optics(s),d=s.retainedPolarization;
 const channelWeights=[(1+3*d)/4,(1-d)/4,(1-d)/4,(1-d)/4];
 const depolarizedPopulation=population.flatMap(r=>pauli.map((j,i)=>({field:apply(j,r.field),probability:r.probability*channelWeights[i],channel:i})));
 const outputPopulation=depolarizedPopulation.map(r=>({...r,field:apply(o.combined,r.field)}));
 const input=polarizationCoherencyState(average(population)),depolarized=polarizationCoherencyState(average(depolarizedPopulation)),output=polarizationCoherencyState(average(outputPopulation));
 const m=polarizationMueller(o.combined).map(row=>row.map((v,i)=>i===0?v:v*d)),muellerStokes=m.map(row=>row.reduce((sum,v,i)=>sum+v*input.stokes[i],0));
 const viaMatrix=transformed(depolarized.coherency,o.combined),measurements=analyzerMeasurements(outputPopulation);
 return{input,depolarized,output,population,depolarizedPopulation,outputPopulation,channelWeights,matrices:o,mueller:m,muellerStokes,
  outputViaMatrix:viaMatrix,measurements,
  transmittance:input.intensity===0?null:output.intensity/input.intensity,
  depolarizationPlacement:'before-the-deterministic-optical-elements' as const,
  normalizedObservation:'population-averaged-quadratic-observations-not-squared-mean-field' as const,
  convention:'exp(-i omega t),right-handed(x,y,z),S3=-2Im(Cxy)' as const};
}

/** Spatial analyzer scans are not recomputed by scalar channel sweeps. */
export function stokesPolarizationExperiment(s:StokesPolarizationExperiment){
 const q=stokesPolarizationObservation(s),outputPopulation=q.outputPopulation;
 const analyzerScan=Array.from({length:129},(_,i)=>{const angle=Math.PI*i/128,j=polarizationAnalyzer(angle);const c=average(outputPopulation.map(r=>({...r,field:apply(j,r.field)})));return{angle,intensity:c.xx+c.yy};});
 return{...q,analyzerScan};
}
