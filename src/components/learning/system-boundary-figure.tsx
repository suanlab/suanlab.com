'use client';
import {useLanguage} from '@/components/language-provider';
import type {PhysicalModelingResult} from '@/lib/learning-physical-modeling';
const fmt=(v:number|null)=>v===null?'—':Math.abs(v)>=10000||(v!==0&&Math.abs(v)<.001)?v.toExponential(2):Number(v.toPrecision(4)).toString();
export default function SystemBoundaryFigure({result,label,axis}:{result:PhysicalModelingResult;label:string;axis:string}){
 const{language}=useLanguage(),ko=language==='ko',b=result.balance,o=result.observation,[x0,x1,y0,y1]=result.bounds!,x=(v:number)=>60+255*(v-x0)/(x1-x0),y=(v:number)=>260-200*(v-y0)/(y1-y0),colors=['blue','orange','green'];
 const rows=b?[
  `${ko?'경계':'Boundary'} ${['A','A+B','A+B+E'][b.cut]} · n=${b.query} min`,
  `M=${fmt(b.selected.stored)} kg · ΔM=${fmt(b.selected.change)} kg`,
  `${ko?'누적 유입':'Cumulative in'}=${fmt(b.selected.cumulativeIn)} kg`,
  `${ko?'누적 유출':'Cumulative out'}=${fmt(b.selected.cumulativeOut)} kg`,
  `${ko?'모형의 질량 폐쇄':'Structural mass closure'}: ${Number(b.structurallyMassClosed)}`,
  `${ko?'0~20단계 정확 수지':'Exact balances0..20'}: ${Number(b.allBalancesHold)}`,
 ]:o?[
  `${ko?'센서':'Sensor'}: ${['A','B','A+B','A−B'][o.sensor]} · n=${o.start}..${o.end}`,
  `${ko?'기록 수':'Record count'}: ${o.records.length}`,
  `${ko?'선택한 기록의 정확 일치':'Exact selected-record match'}: ${o.sameRecordedOutputs===null?'—':Number(o.sameRecordedOutputs)}`,
  `${ko?'처음 구별하는 기록':'First distinguishing record'}: ${fmt(o.firstDistinguishingStep)}`,
  `${ko?'정확 정수에서 구한 최대 차이':'Maximum gap from exact integers'}: ${fmt(o.maxRecordedDifference)} kg`,
  ko?'두 가설 비교이며 전체 상태 식별 정리가 아닙니다.':'A two-hypothesis comparison, not full identification.',
 ]:[];
 return <div className="ai-game-figure"><figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'계의 질량과 관측 기록 · 가로 이동 가능':'System mass and observation records · Horizontal scrolling available'}><svg data-system-history viewBox="0 0 350 560" role="img" aria-label={label}>
 <path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 {result.curves.map((c,i)=><path key={i} d={c.map(([a,v],j)=>`${j?'L':'M'}${x(a)} ${y(v)}`).join(' ')} fill="none" stroke={`var(--learning-${colors[i]})`} strokeWidth={i===0?3:2} strokeDasharray={i===1?'5 4':i===2?'2 4':undefined}/>)}
 {[y0,(y0+y1)/2,y1].map(v=><text key={v} x="53" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{fmt(v)}</text>)}
 <text x="60" y="282" fill="currentColor" style={{fontSize:10}}>0</text><text x="315" y="282" textAnchor="end" fill="currentColor" style={{fontSize:10}}>20 min</text>
 {result.dots?.map(([a,v],i)=><circle key={i} cx={x(a)} cy={y(v)} r={o?2.5:4} fill={`var(--learning-${result.dotColors?.[i]||'blue'})`}/>)}
 {o&&!o.hasObservations&&<text x="175" y="155" textAnchor="middle" fill="currentColor" style={{fontSize:12}}>{ko?'이 창에는 관측 기록이 없습니다.':'No observation records in this window.'}</text>}
 {rows.map((v,i)=><text key={i} x="175" y={325+34*i} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 <text x="175" y="540" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{b?(ko?'파랑 저장량 · 주황 수지 · 초록 확장 총량':'Blue stored · orange flow balance · green expanded total'):(ko?'파랑 가설 0 · 주황 가설 1 · 기록은 이산 시각':'Blue world0 · orange world1 · discrete records')}</text>
 </svg></div><figcaption>{axis}</figcaption></figure>
 <figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'선택 경계와 가설의 내부 상태':'Selected boundary and hypothesized internal states'}>{b?<BoundaryDrawing result={result} label={label}/>:<LatentDrawing result={result} label={label}/>}</div><figcaption>{b?(ko?'점선은 계산 경계이고 화살표는 모형의 이동 연결이며 비율 0인 연결은 흐리게 표시합니다. 거리나 화살표 폭은 질량을 나타내지 않습니다. 질량 폐쇄만 검사하며 열적 고립을 주장하지 않습니다.':'The dashed outline is the accounting boundary; arrows show model links, faded when their fraction is0. Distances and arrow widths do not encode mass. Only mass closure is assessed, not thermal isolation.'):(ko?'두 가설이 예측하는 내부 A 질량입니다. 관측값이나 복원한 상태가 아닙니다. 연결선은 이산 단계의 읽기를 돕고 연속 시간 법칙을 주장하지 않습니다.':'These are hypothesized internal A masses, not observations or reconstructed states. Connecting lines guide discrete steps without asserting a continuous-time law.')}</figcaption></figure>
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'경계 수지와 관측 비교 계산 표':'Boundary-balance and observation-comparison table'}><table><caption>{ko?'현재 모형의 계산 결과':'Calculation results for the current model'}</caption><thead><tr>{result.table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{result.table.rows.map((row,i)=><tr key={i}>{row.map((v,j)=><td key={j}>{typeof v==='number'?fmt(v):v}</td>)}</tr>)}</tbody></table></div>
 </div>;
}
function BoundaryDrawing({result,label}:{result:PhysicalModelingResult;label:string}){
 const{language}=useLanguage(),ko=language==='ko',r=result.balance!,q=r.selected,boxColors=['blue','orange','green'];const rows=[`${ko?'마지막 구간 유입':'Last interval in'}: ${fmt(q.lastIn)} kg`,`${ko?'마지막 구간 유출':'Last interval out'}: ${fmt(q.lastOut)} kg`,`${ko?'누적 수지 잔차 분자':'Cumulative balance residual numerator'}: ${q.exactResidualNumerator}`,`${ko?'확장 총량 보존':'Expanded total conserved'}: ${Number(r.allExpandedTotalsConserved)}`,ko?'0분에는 마지막 구간이 정의되지 않습니다.':'At minute0 no last interval is defined.',ko?'구조적 폐쇄와 0인 순유량은 다릅니다.':'Structural closure differs from zero net crossing.'];
 return <svg data-system-boundary data-boundary-code={r.cut} viewBox="0 0 350 560" role="img" aria-label={label}><text x="175" y="24" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{ko?'계산 경계와 가능한 질량 이동':'Accounting boundary and allowed mass transfers'}</text>
 <rect x="15" y="40" width={[100,210,320][r.cut]} height="140" rx="8" fill="none" stroke="currentColor" strokeDasharray="5 3"/>
 {q.amounts.map((v,i)=><g key={i}><rect x={25+110*i} y="65" width="80" height="80" rx="5" fill={`var(--learning-${boxColors[i]})`} fillOpacity=".1" stroke={`var(--learning-${boxColors[i]})`}/><text x={65+110*i} y="91" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{i===2?(ko?'환경 E':'environment E'):['A','B'][i]}</text><text x={65+110*i} y="120" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{fmt(v)} kg</text></g>)}
 <path data-system-exchange-active={r.exchange>0?'true':'false'} d="M105 82H135l-5 -4m5 4l-5 4M135 132H105l5 -4m-5 4l5 4" fill="none" stroke="currentColor" opacity={r.exchange>0?1:.2}/><path data-system-leak-active={r.leak>0?'true':'false'} d="M215 106H245l-5 -4m5 4l-5 4" fill="none" stroke="currentColor" opacity={r.leak>0?1:.2}/>
 <text x="120" y="161" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{`p=${r.exchange}%`}</text><text x="230" y="161" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{`ℓ=${r.leak}%`}</text>
 <text x="175" y="211" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{`${['A','A+B','A+B+E'][r.cut]} · n=${r.query} min`}</text>
 {rows.map((v,i)=><text key={i} x="175" y={260+43*i} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 </svg>;
}
function LatentDrawing({result,label}:{result:PhysicalModelingResult;label:string}){
 const{language}=useLanguage(),ko=language==='ko',r=result.observation!,x=(n:number)=>60+255*n/20,y=(v:number)=>260-200*v/10.5;
 return <svg data-system-latent viewBox="0 0 350 560" role="img" aria-label={label}><path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 {r.latentA.map((c,i)=><path key={i} d={c.map(([n,v],j)=>`${j?'L':'M'}${x(n)} ${y(v)}`).join(' ')} fill="none" stroke={`var(--learning-${i?'orange':'blue'})`} strokeWidth={i?2:3} strokeDasharray={i?'5 4':undefined}/>)}
 {[0,5,10].map(v=><text key={v} x="53" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{v}</text>)}
 <text x="60" y="282" fill="currentColor" style={{fontSize:10}}>0</text><text x="315" y="282" textAnchor="end" fill="currentColor" style={{fontSize:10}}>20 min</text>
 {[ko?'두 가설의 내부 A 질량 · kg':'Internal A mass in the two hypotheses · kg',ko?'초기 상태 0:(8,2,0), 1:(2,8,0) kg':'Initial states0:(8,2,0), 1:(2,8,0) kg',`${ko?'마지막 시각 A':'End-time A'}: ${r.endStates.map(s=>fmt(s[0])).join(', ')} kg`,`${ko?'마지막 전체 상태의 정확 일치':'Exact end-state match'}: ${Number(r.sameEndState)}`,`${ko?'반올림 기록의 일치':'Rounded record match'}: ${r.roundedSamplesCoincide===null?'—':Number(r.roundedSamplesCoincide)}`,ko?'보이지 않는 상태는 가설이지 복원 결과가 아닙니다.':'Hidden states are hypotheses, not reconstructions.'].map((v,i)=><text key={i} x="175" y={325+34*i} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 </svg>;
}
