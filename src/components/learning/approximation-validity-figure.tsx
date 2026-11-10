'use client';
import {useLanguage} from '@/components/language-provider';
import type {PhysicalModelingResult} from '@/lib/learning-physical-modeling';
import type {Vec2} from '@/lib/learning-simulations';
const fmt=(v:number|null)=>v===null?'—':v!==0&&(Math.abs(v)<.001||Math.abs(v)>10000)?v.toExponential(2):Number(v.toPrecision(4)).toString();
const flag=(v:boolean|null)=>v===null?'—':String(Number(v));
function Plot({curves,colors=['blue','orange','green'],range,title,yLabel,xLabel,rows,marker,dots=[],empty,minimum}:{curves:Vec2[][];colors?:string[];range:number;title:string;yLabel:string;xLabel:string;rows:string[];marker:string;dots?:Vec2[];empty?:string;minimum?:number}){
 const ys=curves.flatMap(c=>c.map(p=>p[1])).concat(dots.map(p=>p[1])),lo=minimum??Math.min(0,...ys),hi=Math.max(lo+1,...ys),pad=(hi-lo)*.05,y0=minimum===undefined?lo-pad:minimum,y1=hi+pad,x=(v:number)=>60+255*v/range,y=(v:number)=>260-200*(v-y0)/(y1-y0);
 return <svg data-approximation-plot={marker} viewBox="0 0 350 560" role="img" aria-label={title}>
 <text x="175" y="25" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{title}</text>
 <path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 {curves.map((c,i)=><path key={i} d={c.map(([a,b],j)=>`${j?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${colors[i]})`} strokeWidth={i===0?3:2} strokeDasharray={i===1?'5 4':i===2?'2 4':undefined}/>)}
 {dots.map(([a,b],i)=><circle key={i} cx={x(a)} cy={y(b)} r="4" fill="currentColor"/>)}
 {[y0,(y0+y1)/2,y1].map(v=><text key={v} x="53" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{fmt(v)}</text>)}
 <text x="60" y="281" fill="currentColor" style={{fontSize:10}}>0</text><text x="315" y="281" textAnchor="end" fill="currentColor" style={{fontSize:10}}>{range} {xLabel}</text>
 <text x="175" y="303" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{yLabel}</text>
 {empty&&<text x="175" y="160" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{empty}</text>}
 {rows.map((v,j)=><text key={j} x="175" y={332+29*j} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 </svg>;
}
export default function ApproximationValidityFigure({result,label,axis}:{result:PhysicalModelingResult;label:string;axis:string}){
 const{language}=useLanguage(),ko=language==='ko',p=result.pendulum,d=result.dragRegimes;
 let first:React.ReactNode,second:React.ReactNode,caption:string;
 if(p){
  first=<Plot curves={p.curves} range={150} minimum={.95} dots={[[p.angleDegrees,p.ratio],[p.angleDegrees,p.approximationRatio]]} title={ko?'진폭에 따른 주기 비율':'Period ratios versus release amplitude'} xLabel={ko?'도':'deg'} yLabel="T/T₀" marker="pendulum-period" rows={[`α=${p.angleDegrees}° · L=${fmt(p.length)} m · g=${fmt(p.gravity)} m/s²`,`${ko?'극한 주기 T₀':'Limiting period T0'}: ${fmt(p.limitPeriod)} s`,`${ko?'비선형 진동 주기':'Nonlinear oscillation period'}: ${fmt(p.period)} s`,`${ko?'선택 근사의 진동 주기':'Selected approximate period'}: ${fmt(p.approximatePeriod)} s`,ko?'파랑 비선형 기준 · 주황 선택 근사':'Blue nonlinear reference · orange selected approximation',ko?'0도 점은 극한이며 정지 궤적의 주기가 아닙니다.':'The0degree point is a limit, not an equilibrium cycle.']} />;
  second=<Plot curves={p.errorCurves} range={150} title={ko?'주기 상대 오차와 요청 기준':'Relative period errors and requested tolerance'} xLabel={ko?'도':'deg'} yLabel={ko?'기준 주기 분모의 상대 오차 · %':'Relative error using reference period · percent'} marker="pendulum-error" dots={p.relativeError===null?[]:[[p.angleDegrees,p.relativeError*100]]} rows={[`${ko?'현재 상대 오차':'Current relative error'}: ${fmt(p.relativeError===null?null:p.relativeError*100)} %`,`${ko?'허용 상대 오차':'Relative tolerance'}: ${fmt(p.tolerance*100)} %`,`${ko?'선택 주기의 허용 오차 만족':'Selected period meets tolerance'}: ${flag(p.meetsTolerance)}`,ko?'파랑 선형 오차 · 주황 보정 오차 · 초록 허용치':'Blue linear error · orange corrected error · green tolerance',ko?'진폭 0의 주기 상대 오차는 미정입니다.':'Period relative error is unavailable at zero amplitude.',ko?'주기 정확도는 전체 파형의 인증이 아닙니다.':'Period accuracy does not certify the whole waveform.']} />;
  caption=ko?'근사 오차의 기준은 명시한 이상 진자입니다. 수치 평가 오차와 실제 진자의 모형 오차는 별도로 확인합니다.':'Approximation error is relative to the declared ideal pendulum. Numerical evaluation and real-model discrepancy require separate checks.';
 }else if(d){
  const q=d.query,unit=d.quantity===0?'m/s':'m';
  first=<Plot curves={d.curves} range={10} dots={[[d.time,q.reference],...(q.selected===null?[]:[[d.time,q.selected] as Vec2])]} title={ko?'물리 시각에서의 기준과 근사':'Parent and approximation in physical time'} xLabel="s" yLabel={`${d.quantity===0?(ko?'속도':'Velocity'):(ko?'변위':'Displacement')} · ${unit}`} marker="drag-time" rows={[`t=${fmt(d.time)} s · τ=${fmt(d.tau)} s`,`${ko?'기준 관측량':'Parent observable'}: ${fmt(q.reference)} ${unit}`,`${ko?'선택 근사':'Selected approximation'}: ${fmt(q.selected)} ${unit}`,`${ko?'상대 오차':'Relative error'}: ${fmt(d.relativeError===null?null:d.relativeError*100)} %`,`${ko?'조회 오차 기준 만족':'Selected query meets tolerance'}: ${flag(d.meetsTolerance)}`,ko?'파랑 기준 · 주황 선택 근사':'Blue parent · orange selected approximation',ko?'음의 장시간 변위는 적용 범위 밖의 근사입니다.':'Negative late displacement is an out-of-regime approximation.']} />;
  second=<Plot curves={d.dimensionlessCurves} range={6} title={ko?'시간 척도로 정규화한 비교':'Comparison in normalized time'} xLabel="t/τ" yLabel={d.quantity===0?'v/(gτ)':'x/(gτ²)'} marker="drag-normalized" empty={d.tau===null?(ko?'항력 0: 유한한 종단 척도가 없습니다.':'Zero drag: no finite terminal scale.'):undefined} rows={[`${ko?'종단 속도':'Terminal velocity'}: ${fmt(d.terminalVelocity)} m/s`,`${ko?'조회 무차원 시각':'Query dimensionless time'}: ${fmt(d.tau===null?null:q.z)}`,`${ko?'허용 상대 오차':'Relative tolerance'}: ${fmt(d.tolerance*100)} %`,ko?'초기 근사와 장시간 근사는 다른 식입니다.':'Early and late approximations use different formulas.',ko?'기준 0의 상대 오차는 미정입니다.':'Relative error is unavailable for a zero reference.',ko?'실제 유동에서 선형 항력을 인증하지 않습니다.':'This does not validate linear drag for every real flow.']} />;
  caption=ko?'같은 시각이라도 속도와 변위의 오차는 다릅니다. 항력이 0이면 유한한 시간 척도와 장시간 근사는 미정입니다.':'Velocity and displacement errors differ even at one time. Zero drag leaves finite terminal scales and late approximations unavailable.';
 }else return null;
 return <div className="ai-game-figure">{[first,second].map((v,j)=><figure className="ai-foundation-figure" key={j}><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={`${label} · ${j+1} · ${ko?'가로 이동 가능':'Horizontal scrolling available'}`}>{v}</div><figcaption>{j===0?axis:caption}</figcaption></figure>)}
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'근사 계산 표 · 가로 이동 가능':'Approximation calculation table · Horizontal scrolling available'}><table><caption>{ko?'현재 조건의 기준과 근사':'Parent and approximation for current settings'}</caption><thead><tr>{result.table.headers.map(v=><th key={v} scope="col">{v}</th>)}</tr></thead><tbody>{result.table.rows.map((row,j)=><tr key={j}>{row.map((v,k)=><td key={k}>{typeof v==='number'?fmt(v):v}</td>)}</tr>)}</tbody></table></div>
 </div>;
}
