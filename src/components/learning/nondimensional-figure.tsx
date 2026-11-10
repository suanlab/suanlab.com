'use client';
import {useLanguage} from '@/components/language-provider';
import type {PhysicalModelingResult} from '@/lib/learning-physical-modeling';
import type {Vec2} from '@/lib/learning-simulations';
const fmt=(v:number)=>v!==0&&(Math.abs(v)<.001||Math.abs(v)>10000)?v.toExponential(2):Number(v.toPrecision(4)).toString();
function Plot({curves,range,title,xLabel,yLabel,rows,marker,dots=[]}:{curves:Vec2[][];range:number;title:string;xLabel:string;yLabel:string;rows:string[];marker:string;dots?:Vec2[]}){
 const ys=curves.flatMap(c=>c.map(p=>p[1])).concat(dots.map(p=>p[1])),lo=Math.min(0,...ys),hi=Math.max(1,...ys),pad=(hi-lo)*.05,y0=lo-pad,y1=hi+pad,x=(v:number)=>60+255*v/range,y=(v:number)=>260-200*(v-y0)/(y1-y0);
 return <svg data-nondimensional-plot={marker} viewBox="0 0 350 560" role="img" aria-label={title}>
 <text x="175" y="25" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{title}</text><path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 {curves.map((c,i)=><path key={i} d={c.map(([a,b],j)=>`${j?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${i?'orange':'blue'})`} strokeWidth={i?2:3} strokeDasharray={i?'5 4':undefined}/>)}
 {dots.map(([a,b],i)=><circle key={i} cx={x(a)} cy={y(b)} r="4" fill="var(--learning-green)"/>)}
 {[y0,(y0+y1)/2,y1].map(v=><text key={v} x="53" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{fmt(v)}</text>)}
 <text x="60" y="281" fill="currentColor" style={{fontSize:10}}>0</text><text x="315" y="281" textAnchor="end" fill="currentColor" style={{fontSize:10}}>{range} {xLabel}</text><text x="175" y="303" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{yLabel}</text>
 {rows.map((v,j)=><text key={j} x="175" y={332+29*j} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 </svg>;
}
function Defect({result}:{result:PhysicalModelingResult}){
 const{language}=useLanguage(),ko=language==='ko',b=result.buckingham!,extent=Math.max(1,...b.defect.map(Math.abs)),y=(v:number)=>260-200*(v+extent)/(2*extent);
 const rows=[`${ko?'질량·길이·시간 차이 지수':'M,L,T exponent defect'}: (${b.defect.join(', ')})`,`${ko?'후보 단항식의 차원 없음':'Candidate monomial dimensionless'}: ${Number(b.dimensionless)}`,`${ko?'차원 행렬 rank와 영공간 차원':'Matrix rank and nullity'}: ${b.rank}, ${b.nullity}`,`${ko?'기저 지수 a,b':'Basis powers a,b'}: ${b.basisCoefficients?b.basisCoefficients.join(', '):'—'}`,'ΠF=F/(ρV²D²), Πμ=μ/(ρVD)=1/Re',ko?'차원의 일치만으로 실제 법칙을 정하지 못합니다.':'Dimensional agreement does not identify the real law.'];
 return <svg data-nondimensional-plot="buckingham-defect" viewBox="0 0 350 560" role="img" aria-label={ko?'후보 단항식의 차원 지수 차이':'Dimension exponent defects of the candidate monomial'}><text x="175" y="25" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{ko?'차원 지수와 무차원 기저':'Dimension exponents and dimensionless basis'}</text><path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/><path d={`M60 ${y(0)}H315`} fill="none" stroke="currentColor" opacity=".3"/>
 {b.defect.map((v,j)=><g key={j}><path d={`M${90+90*j} ${y(0)}V${y(v)}`} stroke="var(--learning-blue)" strokeWidth="22"/><circle cx={90+90*j} cy={y(v)} r="3" fill="var(--learning-orange)"/><text x={90+90*j} y="282" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{['M','L','T'][j]}</text></g>)}
 {[-extent,0,extent].map(v=><text key={v} x="53" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{v}</text>)}
 {rows.map((v,j)=><text key={j} x="175" y={332+29*j} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 </svg>;
}
export default function NondimensionalFigure({result,label,axis}:{result:PhysicalModelingResult;label:string;axis:string}){
 const{language}=useLanguage(),ko=language==='ko',o=result.nondimensionalOscillator,b=result.buckingham;
 let first:React.ReactNode,second:React.ReactNode,caption:string;
 if(o){
  first=<Plot curves={o.normalizedCurves} range={8} title={ko?'같은 무차원 시각의 두 계':'Two systems at the same normalized time'} xLabel="s" yLabel="u=x/X" marker="oscillator-normalized" dots={o.query.map(q=>[q.normalizedTime,q.normalizedDisplacement])} rows={[`s=${fmt(o.queryTime)} · ζA=${fmt(o.worlds[0].zeta)}, ζB=${fmt(o.worlds[1].zeta)}`,`${ko?'같은 무차원 법칙':'Same normalized law'}: ${Number(o.sameNormalizedLaw)}`,`${ko?'같은 무차원 초기 상태':'Same normalized initial data'}: ${Number(o.sameNormalizedInitialData)}`,`${ko?'같은 무차원 문제':'Same normalized problem'}: ${Number(o.sameNormalizedProblem)}`,`${ko?'초기 자극 없음':'Zero excitation'}: ${Number(o.zeroExcitation)}`,`${ko?'표시 격자의 최대 간격':'Maximum sampled gap'}: ${fmt(o.maxSampledNormalizedGap)}`,ko?'곡선의 일치만으로 법칙을 인증하지 않습니다.':'Coincident curves do not certify matching laws.']} />;
  second=<Plot curves={o.physicalCurves} range={20} title={ko?'같은 초 단위 시계의 물리 궤적':'Physical trajectories on a shared clock'} xLabel={ko?'초':'seconds'} yLabel={ko?'변위 · m':'Displacement · m'} marker="oscillator-physical" dots={o.query.map(q=>[q.physicalTime,q.physicalDisplacement])} rows={[`${ko?'조회 물리 시간 A,B':'Query physical times A,B'}: ${o.query.map(q=>fmt(q.physicalTime)).join(', ')} s`,`${ko?'조회 물리 변위 A,B':'Query displacements A,B'}: ${o.query.map(q=>fmt(q.physicalDisplacement)).join(', ')} m`,`h=${o.massScale}, r=${o.rateScale}, ℓ=${o.lengthScale}`,ko?'파랑 A · 주황 B · 초록 점은 같은 s의 조회':'Blue A · orange B · green queries use the same s',ko?'같은 무차원 조회는 다른 물리 시각입니다.':'A shared normalized query uses different clocks.',ko?'SI 단위는 같고 물리 계의 매개변수가 다릅니다.':'SI units agree; the physical parameters differ.']} />;
  caption=ko?'초기 상태도 함께 변환해야 상사 문제를 얻습니다. 독립 길이 척도는 초기 변위 0에서도 양수입니다.':'Initial data must scale coherently for similarity. Independent length scales stay positive when initial displacement is zero.';
 }else if(b){
  first=<Defect result={result}/>;
  second=<Plot curves={b.curves} range={10} title={ko?'구성된 두 함수와 조회점':'Two constructed functions and a query'} xLabel="Re" yLabel="ΠF" marker="buckingham-hypotheses" dots={[[b.reynolds,b.forceGroup]]} rows={[`Re=${fmt(b.reynolds)}, ΠF=${fmt(b.forceGroup)}`,'Φ₁=1/Re, Φ₂=1+1/Re',`${ko?'첫 예시 함수와 정확 일치':'Exact match with toy function1'}: ${Number(b.hypothesisMatches[0])}`,`${ko?'둘째 예시 함수와 정확 일치':'Exact match with toy function2'}: ${Number(b.hypothesisMatches[1])}`,ko?'파랑 Φ₁ · 주황 Φ₂ · 초록은 조회':'Blue Phi1 · orange Phi2 · green query',ko?'실제 항력 법칙이나 실험 데이터가 아닙니다.':'These are not empirical drag laws or measured data.',ko?'차원 판정과 조회 일치는 다른 질문입니다.':'Dimensions and query agreement are separate questions.']} />;
  caption=ko?'관련 변수의 선택은 물리적 가정입니다. 무차원 기저의 완전성이 실제 변수 목록의 완전성을 증명하지 않습니다.':'Selecting relevant variables is a physical assumption. Completeness of the basis does not establish completeness of the physical variable list.';
 }else return null;
 return <div className="ai-game-figure">{[first,second].map((v,j)=><figure className="ai-foundation-figure" key={j}><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={`${label} · ${j+1} · ${ko?'가로 이동 가능':'Horizontal scrolling available'}`}>{v}</div><figcaption>{j===0?axis:caption}</figcaption></figure>)}
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'무차원 계산 표 · 가로 이동 가능':'Nondimensional calculation table · Horizontal scrolling available'}><table><caption>{ko?'현재 설정의 물리 변수와 계산':'Physical variables and calculations for current settings'}</caption><thead><tr>{result.table.headers.map(v=><th key={v} scope="col">{v}</th>)}</tr></thead><tbody>{result.table.rows.map((row,j)=><tr key={j}>{row.map((v,k)=><td key={k}>{typeof v==='number'?fmt(v):v}</td>)}</tr>)}</tbody></table></div>
 </div>;
}
