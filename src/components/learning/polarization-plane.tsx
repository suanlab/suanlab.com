'use client';
import{waveNumber,type Series,type Mark}from'./waves-plot';
/** Both physical axes use EXACTLY the same pixels per unit. Shared radius
 * across preparation stages preserves attenuation and the shape of circles.
 */
export function PolarizationPlane({marker,title,series,radius,xLabel,yLabel,caption,dots=[],circle}:{marker:string;title:string;series:Series[];radius:number;xLabel:string;yLabel:string;caption:string;dots?:Mark[];circle?:number}){
 const scale=120/radius,x=(v:number)=>190+scale*v,y=(v:number)=>148-scale*v;
 return <figure className="ai-foundation-figure" data-waves-panel={marker}><h3>{title}</h3><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={title}><svg viewBox="0 0 350 345" role="img" aria-label={title} data-waves-plot={marker} data-polarization-scale-x={scale} data-polarization-scale-y={scale} data-polarization-radius={radius}>
  <path d="M70 148H310M190 28V268" stroke="var(--learning-neutral)" fill="none"/>
  {circle!==undefined&&<circle cx="190" cy="148" r={circle*scale} stroke="var(--learning-neutral)" strokeDasharray="4 4" fill="none"/>}
  {series.map((s,i)=><g key={i}>{s.segments.map((points,j)=><path key={j} d={points.map(([a,b],k)=>`${k?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${s.color})`} strokeWidth="2.5" strokeDasharray={i?'5 3':undefined}/>)}</g>)}
  {dots.map((p,i)=><g key={i}><path d={`M190 148L${x(p.point[0])} ${y(p.point[1])}`} stroke={`var(--learning-${p.color})`} strokeWidth="1.5"/><circle cx={x(p.point[0])} cy={y(p.point[1])} r="4" fill={`var(--learning-${p.color})`} data-waves-query="true"/></g>)}
  {[-radius,0,radius].map(v=><text key={v} x="64" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{waveNumber(v)}</text>)}
  <g fill="currentColor" style={{fontSize:10}}><text x="70" y="287">{waveNumber(-radius)}</text><text x="310" y="287" textAnchor="end">{waveNumber(radius)}</text><text x="190" y="309" textAnchor="middle">{xLabel}</text><text x="190" y="332" textAnchor="middle">{yLabel}</text></g>
 </svg></div><div className="ai-legend">{series.map((s,i)=><span key={i}><span aria-hidden="true" style={{display:'inline-block',width:8,height:8,borderRadius:'50%',marginRight:4,background:`var(--learning-${s.color})`}}/>{s.name}</span>)}</div><figcaption>{caption}</figcaption></figure>;
}
