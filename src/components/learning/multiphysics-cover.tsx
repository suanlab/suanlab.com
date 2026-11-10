import{thermoelasticAt,thermoelectricAt}from '@/lib/learning-continuum-multiphysics';
import type{ContinuumMultiphysicsKind}from '@/lib/learning-continuum-multiphysics-lab';
export default function MultiphysicsCover({kind}:{kind:ContinuumMultiphysicsKind}){
 const elastic=kind==='thermoelastic-mode-reciprocal-energy-exchange';
 const values=Array.from({length:81},(_,i)=>{
  const x=i/80;
  if(elastic){const q=thermoelasticAt({elasticModulus:2,density:1,heatCapacity:1,referenceTemperature:300,coupling:.02,conductivity:0,wavenumber:1,area:1,initialDisplacement:.001,initialVelocity:.001,initialTemperatureDeviation:.02},4*x);return[x,q.mechanicalAvailability.value!,q.thermalAvailability.value!];}
  const q=thermoelectricAt({length:1,area:1,resistivity:.01,conductivity:.1,seebeck:.05,leftTemperature:300,rightTemperature:310,currentDensity:10,referencePotential:0},x);return[x,q.temperature.value!-300,10*x];
 });
 const top=Math.max(...values.flatMap(v=>v.slice(1))),path=(index:number)=>values.map((v,i)=>`${i?'L':'M'}${45+270*v[0]} ${195-150*v[index]/top}`).join(' ');
 return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art"><path d="M45 35V195H320" fill="none" stroke="var(--learning-neutral)"/>{[1,2].map((index)=><path key={index} d={path(index)} fill="none" stroke={`var(--learning-${index===1?'blue':'orange'})`} strokeWidth="3" strokeDasharray={index===2?'5 3':undefined}/>)}<g fill="currentColor" fontSize="14"><text x="40" y="25">{elastic?'reciprocal thermal exchange':'Seebeck · Peltier · Joule'}</text><text x="45" y="220">{elastic?'mechanical + thermal availability':'temperature · boundary line'}</text></g></svg>;
}
