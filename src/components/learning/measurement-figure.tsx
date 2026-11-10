'use client';
import {useLanguage} from '@/components/language-provider';
import type {PhysicalModelingResult} from '@/lib/learning-physical-modeling';
const fmt=(v:number|null)=>v===null?'—':Math.abs(v)>=10000||(v!==0&&Math.abs(v)<.001)?v.toExponential(2):Number(v.toPrecision(4)).toString();
export default function MeasurementFigure({result,label,axis}:{result:PhysicalModelingResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko',c=result.measurementCalibration,r=result.propagation,[x0,x1,y0,y1]=result.bounds!,x=(v:number)=>60+255*(v-x0)/(x1-x0),y=(v:number)=>260-200*(v-y0)/(y1-y0),colors=['blue','orange','green'];
 const rows=c?[
  `n_R=${c.n} · n_C=${c.nc} · ${ko?'명목 가우스 95%':'nominal Gaussian 95%'}`,
  `${ko?'설명용 참값':'Oracle truth'} x=${c.truth}, b=${c.bias} cm`,
  `${ko?'조회 평균':'Mean queries'} R=${fmt(c.n===0?null:c.queryMean)}, C=${fmt(c.nc===0?null:c.queryCalibration)} cm`,
  `${ko?'교정값':'Corrected'}: ${fmt(c.correctedEstimate)} cm · s=${fmt(c.correctedSE)} cm`,
  `${ko?'잡음 반폭':'Noise radius'}=${fmt(c.nominalRadius)} cm · D=${fmt(c.driftBound)} cm`,
  `${ko?'확장 구간':'Expanded interval'}: ${c.interval?`[${c.interval.map(fmt).join(', ')}] cm`:'—'}`,
  `${ko?'조회 지지집합 양립 R,C':'Query support compatible R,C'}: ${[c.readingQueryCompatible,c.calibrationQueryCompatible].map(v=>v===null?'—':Number(v)).join(', ')}`,
  ko?'참값은 추정값이 아니며 조회는 생성 표본이 아닙니다.':'Truth is not estimated; queries are not sampled data.',
 ]:r?[
  `${['X + Z','X − Z','X × Z'][r.mode]} · ${ko?'출력 단위':'output unit'} ${r.outputUnit}`,
  `${ko?'잠재 ρ':'Latent ρ'}=${fmt(r.rho)} · Pearson=${fmt(r.observableCorrelation)}`,
  `${ko?'평균':'Mean'}: ${fmt(r.mean)} · ${ko?'일차':'first order'} ${fmt(r.firstOrderMean)}`,
  `${ko?'분산':'Variance'}: ${fmt(r.variance)} ${r.varianceUnit}`,
  `${ko?'일차 분산':'First-order variance'}: ${fmt(r.firstOrderVariance)} ${r.varianceUnit}`,
  `${ko?'빠진 고차 분산':'Omitted higher-order variance'}: ${fmt(r.higherOrderVariance)}`,
  `${ko?'표준불확실성':'Standard uncertainty'}: ${fmt(r.standardUncertainty)} ${r.outputUnit}`,
  ko?'출력의 가우스 분포나 95% 구간은 가정하지 않습니다.':'No Gaussian output or 95% interval is assumed.',
 ]:[];
 return <div className="ai-game-figure"><figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'측정 불확실성 전파 그래프 · 가로 이동 가능':'Measurement uncertainty propagation plot · Horizontal scrolling available'}><svg data-measurement-variance viewBox="0 0 350 560" role="img" aria-label={label}>
 <path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 {result.curves.map((curve,i)=><path key={i} d={curve.map(([a,b],j)=>`${j?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${colors[i]})`} strokeWidth={i===0?3:2} strokeDasharray={i===1?'5 4':i===2?'2 4':undefined}/>)}
 {[y0,(y0+y1)/2,y1].map(v=><text key={v} x="53" y={y(v)+3} fill="currentColor" textAnchor="end" style={{fontSize:9}}>{fmt(v)}</text>)}
 <g fill="currentColor"><text x="60" y="282" style={{fontSize:10}}>{x0}</text><text x="315" y="282" textAnchor="end" style={{fontSize:10}}>{x1}</text></g>
 {result.dots?.map(([a,b],i)=><circle key={i} cx={x(a)} cy={y(b)} r="4" fill={`var(--learning-${result.dotColors?.[i]||'blue'})`}/>)}
 {rows.map((v,i)=><text key={i} x="175" y={310+29*i} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 <text x="175" y="550" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{c?(ko?'파랑 대상 평균 · 주황 교정 · 초록 교정 바닥':'Blue target mean · orange corrected · green floor'):(ko?'파랑 정확 모형 · 주황 일차 근사':'Blue exact model · orange first order')}</text>
 </svg></div><figcaption>{axis}</figcaption></figure>
 {r&&<figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'가우스 입력의 잠재 단위 원 이미지':'Image of the latent unit circle for Gaussian inputs'}><InputContour result={result} label={label}/></div><figcaption>{ko?'잠재 표준 좌표의 단위 원을 측정 좌표로 옮긴 기하적 모양입니다. 입력 X,Z의 단위는 cm이며 이 그림에 68%나 95% 포함 확률을 부여하지 않습니다. 특이 공분산은 선분이나 점이 됩니다.':'This is the geometric image of a unit circle in latent standard coordinates. Inputs X,Z are in cm. No68percent or95percent coverage is assigned to this picture; singular covariance gives a segment or point.'}</figcaption></figure>}
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'측정 불확실성 계산 표 · 가로 이동 가능':'Measurement uncertainty calculation table · Horizontal scrolling available'}><table><caption>{ko?'현재 측정 모형의 계산':'Calculations for the current measurement model'}</caption><thead><tr>{result.table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{result.table.rows.map((row,i)=><tr key={i}>{row.map((v,j)=><td key={j}>{typeof v==='number'?fmt(v):v}</td>)}</tr>)}</tbody></table></div>
 </div>;
}
function InputContour({result,label}:{result:PhysicalModelingResult;label:string}){
 const{language}=useLanguage(),ko=language==='ko',r=result.propagation!,extent=Math.max(1,r.ux,r.uz)*1.3,x=(v:number)=>175+115*(v-r.mx)/extent,y=(v:number)=>155-115*(v-r.mz)/extent;
 return <svg data-measurement-contour data-covariance-rank={r.rank} viewBox="0 0 350 350" role="img" aria-label={label}>
 <path d="M60 40V270H290" fill="none" stroke="var(--learning-neutral)"/><path d={r.contour.map(([a,b],i)=>`${i?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke="var(--learning-blue)" strokeWidth="3"/><circle cx={x(r.mx)} cy={y(r.mz)} r="5" fill="var(--learning-orange)"/>
 <g fill="currentColor" style={{fontSize:9}}><text x="60" y="284" style={{fontSize:9}}>{fmt(r.mx-extent)}</text><text x="290" y="284" textAnchor="end" style={{fontSize:9}}>{fmt(r.mx+extent)}</text><text x="53" y="43" textAnchor="end" style={{fontSize:9}}>{fmt(r.mz+extent)}</text><text x="53" y="273" textAnchor="end" style={{fontSize:9}}>{fmt(r.mz-extent)}</text></g>
 <g fill="currentColor" textAnchor="middle"><text x="175" y="23" style={{fontSize:11}}>{ko?'잠재 단위 원의 상 · 입력 공간':'Latent unit-circle image · input space'}</text><text x="175" y="293" style={{fontSize:10}}>{`X [cm] · center (${r.mx}, ${r.mz})`}</text><text x="175" y="317" style={{fontSize:10}}>{`Z [cm] · rank ${r.rank} · Pearson ${fmt(r.observableCorrelation)}`}</text><text x="175" y="341" style={{fontSize:10}}>{ko?'기하적 기준 모양 · 확률 구간 아님':'Geometric reference shape · no coverage claim'}</text></g>
 </svg>;
}
