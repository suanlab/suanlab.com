'use client';
import { useLanguage } from '@/components/language-provider';
import { formatNumber } from '@/lib/learning-simulations';
import type { FoundationResult } from '@/lib/learning-foundations';
import type { ProbabilityResult } from '@/lib/learning-probability';
export default function ProbabilityFigure({result,label,axis}:{result:FoundationResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko',r=result as ProbabilityResult,points=r.curves.flat(),ellipse=r.plot==='ellipse',cdf=Boolean(r.cdf);
 const radius=Math.max(.1,...points.flat().map(Math.abs)),minX=ellipse?-radius:Math.min(0,...points.map(v=>v[0])),maxX=ellipse?radius:Math.max(.1,...points.map(v=>v[0])),minY=ellipse?-radius:0,maxY=ellipse?radius:cdf?1:Math.max(.01,...points.map(v=>v[1]))*1.05;
 const xy=([x,y]:number[])=>[45+260*(x-minX)/(maxX-minX),305-260*(y-minY)/(maxY-minY)],path=(curve:number[][])=>curve.map((v,i)=>`${i?'L':'M'}${xy(v)}`).join('');
 return <figure className="ai-foundation-figure ai-probability-figure">
 {(points.length>0||(r.grid?.length??0)>0)&&<div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'확률 그림 · 작은 화면에서 가로로 이동':'Probability diagram · Scroll horizontally on narrow screens'}>
 {r.plot==='joint'?<svg viewBox="0 0 350 340" role="img" aria-label={label} data-probability-joint="true">
 {r.grid?.map(cell=><g key={`${cell.x}-${cell.y}`}><rect x={65+110*cell.x} y={55+110*(1-cell.y)} width="105" height="105" rx="6" fill="var(--learning-blue)" fillOpacity={.08+.5*cell.p} stroke="var(--learning-blue)"/><text x={117+110*cell.x} y={115+110*(1-cell.y)} textAnchor="middle" fill="var(--learning-neutral)" fontSize="17">{formatNumber(cell.p)}</text></g>)}
 {[0,1].map(x=><g key={x} fill="var(--learning-neutral)" fontSize="13"><text x={117+110*x} y="291" textAnchor="middle">{r.gridLabels?.[0]??'X'}={x}</text><text x="56" y={115+110*(1-x)} textAnchor="end">{r.gridLabels?.[1]??'Y'}={x}</text></g>)}<text x="175" y="324" textAnchor="middle" fill="var(--learning-neutral)" fontSize="12">{ko?'칸 안 숫자는 결합확률입니다.':'Numbers in cells are joint probabilities.'}</text>
 </svg>:<svg viewBox="0 0 350 350" role="img" aria-label={label} data-probability-plot={r.plot} data-equal-scale={ellipse?'true':undefined}>
 <path d={`M45 ${xy([0,0])[1]}H305M${xy([0,0])[0]} 45V305`} fill="none" stroke="var(--learning-neutral)" strokeWidth="1"/>
 {r.curves.map((curve,i)=><g key={i}>{r.plot==='discrete'?curve.map((v,j)=><g key={j}><path d={`M${xy([v[0],0])}L${xy(v)}`} stroke={`var(--learning-${i?'orange':'blue'})`} strokeWidth="2"/><circle cx={xy(v)[0]} cy={xy(v)[1]} r={i?5:3} fill={i?'none':'var(--learning-blue)'} stroke={`var(--learning-${i?'orange':'blue'})`} strokeWidth="2"/></g>):<path d={path(curve)} fill="none" stroke="var(--learning-blue)" strokeWidth="2.5"/>}</g>)}
 {r.jumps?.map((j,i)=><g key={i}><circle cx={xy([j.x,j.from])[0]} cy={xy([j.x,j.from])[1]} r="4" fill="hsl(var(--card))" stroke="var(--learning-blue)" strokeWidth="2"/><circle cx={xy([j.x,j.to])[0]} cy={xy([j.x,j.to])[1]} r="4" fill="var(--learning-blue)"/></g>)}
 {r.jumps?.map(j=><text key={j.x} x={xy([j.x,0])[0]} y="326" textAnchor="middle" fill="var(--learning-neutral)" fontSize="12">{formatNumber(j.x)}</text>)}
 <g fill="var(--learning-neutral)" fontSize="12"><text x="45" y="326">{formatNumber(minX)}</text><text x="305" y="326" textAnchor="end">{formatNumber(maxX)}</text><text x="40" y="40" textAnchor="end">{formatNumber(maxY)}</text><text x="40" y="305" textAnchor="end">{formatNumber(minY)}</text></g>
 </svg>}
 </div>}
 {points.length===0&&<p className="ai-model-note" role="status">{ko?'전체 가중치가 0이어서 확률분포를 정의할 수 없습니다.':'Zero total weight: a probability distribution is undefined.'}</p>}
 <figcaption>{axis}</figcaption>
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'확률 계산 표 · 좌우 화살표로 이동':'Probability calculation table · Use arrow keys to scroll'}><table><thead><tr>{r.table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{r.table.rows.map((row,i)=><tr key={i}>{row.map((v,j)=><td key={j}>{typeof v==='number'?formatNumber(v):v}</td>)}</tr>)}</tbody></table></div>
 </figure>;
}
