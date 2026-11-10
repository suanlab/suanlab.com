'use client';
import type {Vec2} from '@/lib/learning-simulations';
import {waveVerticalDomain} from '@/lib/learning-wave-axis';

export const waveNumber = (v: number | null | undefined) => v == null ? '—' : v !== 0 && (Math.abs(v) < .001 || Math.abs(v) >= 10000) ? v.toExponential(2) : Number(v.toPrecision(4)).toString();
export type Series = {name: string; color: string; segments: Vec2[][]};
export type Mark = {point: Vec2; color: string};
export function WavePlot({marker, title, series, dots = [], vertical = [], xLabel, yLabel, caption, minimumYPadding=.01, xDomain, yDomain}: {marker: string; title: string; series: Series[]; dots?: Mark[]; vertical?: number[]; xLabel: string; yLabel: string; caption: string; minimumYPadding?:number; xDomain?:[number,number]; yDomain?:[number,number]}) {
  const points = [...series.flatMap(s => s.segments.flat()), ...dots.map(d => d.point)];
  const xs = points.map(v => v[0]), ys = points.map(v => v[1]);
  // Keep the frequency-family domain even when measured phase is unavailable.
  // A declared sweep still has a domain when every ordinate is unavailable.
  const fixedDomain=xDomain&&xDomain.every(Number.isFinite)&&xDomain[0]<xDomain[1]?xDomain:undefined;
  const x0 = fixedDomain?.[0]??Math.min(0, ...xs), x1 = fixedDomain?.[1]??(marker.startsWith('frequency-') ? 8 : Math.max(.01, ...xs));
  const automatic=waveVerticalDomain(ys,minimumYPadding);
  const fixedY=yDomain&&yDomain.every(Number.isFinite)&&yDomain[0]<yDomain[1]?yDomain:undefined;
  const y0=fixedY?.[0]??automatic.minimum,y1=fixedY?.[1]??automatic.maximum;
  const x = (v: number) => 62 + 262 * (v - x0) / (x1 - x0), y = (v: number) => 268 - 240 * (v - y0) / (y1 - y0);
  return <figure className="ai-foundation-figure" data-waves-panel={marker}>
    <h3>{title}</h3><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={title}>
      <svg viewBox="0 0 350 345" role="img" aria-label={title} data-waves-plot={marker} data-wave-x-min={x0} data-wave-x-max={x1} data-wave-y-min={y0} data-wave-y-max={y1}>
        <path d="M62 28V268H324" fill="none" stroke="var(--learning-neutral)"/>
        {y0<=0&&y1>=0&&<path d={`M62 ${y(0)}H324`} fill="none" stroke="var(--learning-neutral)" strokeDasharray="3 4"/>}
        {vertical.filter(v => v >= x0 && v <= x1).map((v,i) => <path key={i} d={`M${x(v)} 28V268`} stroke="var(--learning-neutral)" strokeDasharray="3 5"/>)}
        {series.map((s,i) => <g key={i}>{s.segments.map((segment,j) => segment.length === 1 ? <circle key={j} cx={x(segment[0][0])} cy={y(segment[0][1])} r="3" fill={`var(--learning-${s.color})`}/> : <path key={j} d={segment.map(([a,b],k) => `${k ? 'L' : 'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${s.color})`} strokeWidth="2.5" strokeDasharray={i ? '5 3' : undefined}/>)}</g>)}
        {dots.map((dot,i) => <circle key={i} cx={x(dot.point[0])} cy={y(dot.point[1])} r="4" fill={`var(--learning-${dot.color})`} data-waves-query="true"/>)}
        {[y0, (y0+y1)/2, y1].map(v => <text key={v} x="56" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{waveNumber(v)}</text>)}
        <g fill="currentColor" fontSize="10"><text x="62" y="287">{waveNumber(x0)}</text><text x="324" y="287" textAnchor="end">{waveNumber(x1)}</text><text x="190" y="309" textAnchor="middle">{xLabel}</text><text x="190" y="332" textAnchor="middle">{yLabel}</text></g>
      </svg>
    </div><div className="ai-legend">{series.map((s,i) => <span key={i}><span aria-hidden="true" style={{display:'inline-block',width:8,height:8,borderRadius:'50%',marginRight:4,background:`var(--learning-${s.color})`}}/>{s.name}</span>)}</div><figcaption>{caption}</figcaption>
  </figure>;
}
