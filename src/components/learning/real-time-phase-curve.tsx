import React,{useId} from 'react';
export interface RealTimeCurve {label:string;color:'blue'|'orange'|'green';points:readonly{x:number;y:number|null}[];dashed?:boolean;}
/** Gaps remain gaps; no line is joined across an unavailable projection. */
export default function RealTimeCurve({title,description,xLabel,yLabel,curves,lines=true}:{title:string;description:string;xLabel:string;yLabel:string;curves:readonly RealTimeCurve[];lines?:boolean}){
  const id=useId(),valid=curves.flatMap(c=>c.points.filter((p):p is {x:number;y:number}=>p.y!==null&&Number.isFinite(p.x)&&Number.isFinite(p.y)));
  if(!valid.length)return <p>{description}</p>;
  const xmin=Math.min(...valid.map(p=>p.x)),xmax=Math.max(...valid.map(p=>p.x)),ymin=Math.min(0,...valid.map(p=>p.y)),ymax=Math.max(0,...valid.map(p=>p.y));
  const x=(v:number)=>70+550*(v-xmin)/(xmax-xmin||1),y=(v:number)=>265-175*(v-ymin)/(ymax-ymin||1);
  return <figure><figcaption id={`${id}-caption`}>{title}</figcaption><div className="qubit-scroll" role="region" tabIndex={0} aria-label={title}><svg viewBox="0 0 700 350" role="img" aria-labelledby={`${id}-caption ${id}-desc`} style={{width:'100%',minWidth:700,fontSize:16}}><desc id={`${id}-desc`}>{description}</desc><path d="M70 75V265H620" fill="none" stroke="var(--learning-neutral)"/>
    <line data-real-time-zero-guide="true" x1={70} x2={620} y1={y(0)} y2={y(0)} stroke="var(--learning-neutral)" strokeDasharray="4 5"/><text x={43} y={y(0)+5} fill="currentColor">0</text>
    {curves.map((c,i)=><g key={i} stroke={`var(--learning-${c.color})`} fill={`var(--learning-${c.color})`}>{c.points.map((p,j)=>{
      if(p.y===null||!Number.isFinite(p.y)||!Number.isFinite(p.x))return null;
      const previous=c.points[j-1],join=lines&&previous?.y!==null&&previous&&Number.isFinite(previous.y)&&Number.isFinite(previous.x);
      return <g key={j}>{join&&<line x1={x(previous.x)} y1={y(previous.y!)} x2={x(p.x)} y2={y(p.y)} strokeWidth={2} strokeDasharray={c.dashed?'6 5':undefined}/>}<circle cx={x(p.x)} cy={y(p.y)} r={2.5}/></g>;
    })}</g>)}
    <text x={70} y={35} fill="currentColor">{yLabel}: {ymin.toPrecision(4)} … {ymax.toPrecision(4)}</text><text x={70} y={300} fill="currentColor">{xmin.toPrecision(4)}</text><text x={620} y={300} textAnchor="end" fill="currentColor">{xmax.toPrecision(4)}</text><text x={350} y={335} textAnchor="middle" fill="currentColor">{xLabel}</text>
  </svg></div><p>{curves.map(c=>c.label).join(' · ')}</p><p>{description}</p></figure>;
}
