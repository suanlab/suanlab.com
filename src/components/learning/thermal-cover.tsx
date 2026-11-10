import React from 'react';
/** Decorative schematic; numerical evidence is computed separately in Workers. */
export default function ThermalCover({kind}:{kind:string}){
 const driven=kind==='driven-two-state-heat-work';
 return <svg data-thermal-cover={kind} viewBox="0 0 480 320" aria-hidden="true" style={{width:'100%',height:'100%'}}><g fill="none" stroke="currentColor" strokeWidth={3}>{driven?<><path d="M40 215H140M185 215H285M330 215H430M40 160H140M185 80H285M330 125H430"/><path d="M90 145V90M80 104L90 90L100 104M235 100V165M225 151L235 165L245 151M40 270H430" strokeDasharray="7 6"/><circle cx={90} cy={160} r={9}/><circle cx={235} cy={215} r={9}/><circle cx={380} cy={125} r={9}/></>:<><rect x={30} y={70} width={100} height={160} rx={15}/><rect x={350} y={70} width={100} height={160} rx={15}/><circle cx={240} cy={150} r={48}/><path d="M145 130H185M170 120L185 130L170 140M295 170H335M320 160L335 170L320 180"/><path d="M185 170H145M160 160L145 170L160 180M335 130H295M310 120L295 130L310 140" strokeDasharray="7 6"/><path d="M220 150H260M250 140L260 150L250 160"/></>}</g></svg>;
}
