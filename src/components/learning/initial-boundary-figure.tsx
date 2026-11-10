'use client';
import {useLanguage} from '@/components/language-provider';
import type {PhysicalModelingResult} from '@/lib/learning-physical-modeling';
import type {Vec2} from '@/lib/learning-simulations';
const fmt=(v:number)=>v!==0&&(Math.abs(v)<.001||Math.abs(v)>10000)?v.toExponential(2):Number(v.toPrecision(4)).toString();
const flag=(v:boolean|null)=>v===null?'—':String(Number(v));

function Plot({curves,colors=['blue','orange','green'],range,title,xLabel,yLabel,rows,marker,dots=[]}:{curves:Vec2[][];colors?:string[];range:number;title:string;xLabel:string;yLabel:string;rows:string[];marker:string;dots?:Vec2[]}){
 const extent=Math.max(1,...curves.flatMap(c=>c.map(p=>Math.abs(p[1]))),...dots.map(p=>Math.abs(p[1])))*1.1;
 const x=(v:number)=>60+255*v/range,y=(v:number)=>260-200*(v+extent)/(2*extent);
 return <svg data-initial-boundary-plot={marker} viewBox="0 0 350 560" role="img" aria-label={title}>
 <text x="175" y="25" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{title}</text>
 <path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 <path d={`M60 ${y(0)}H315`} fill="none" stroke="var(--learning-neutral)" strokeDasharray="2 4" opacity=".4"/>
 {curves.map((c,i)=><path key={i} d={c.map(([a,b],j)=>`${j?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${colors[i]})`} strokeWidth={i===0?3:2} strokeDasharray={i===1?'5 4':i===2?'2 4':undefined}/>)}
 {dots.map(([a,b],i)=><circle key={i} cx={x(a)} cy={y(b)} r="4" fill="var(--learning-green)"/>)}
 {[-extent,0,extent].map(v=><text key={v} x="53" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{fmt(v)}</text>)}
 <text x="60" y="280" fill="currentColor" style={{fontSize:10}}>0</text><text x="315" y="280" textAnchor="end" fill="currentColor" style={{fontSize:10}}>{range} {xLabel}</text>
 <text x="175" y="302" textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{yLabel}</text>
 {rows.map((row,i)=><text key={i} x="175" y={332+29*i} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{row}</text>)}
 </svg>;
}

export default function InitialBoundaryFigure({result,label,axis}:{result:PhysicalModelingResult;label:string;axis:string}){
 const{language}=useLanguage(),ko=language==='ko',i=result.initialData,h=result.heatBoundary,p=result.poissonBoundary;
 let first:React.ReactNode,second:React.ReactNode,caption:string;
 if(i){
  const unit=i.mode===0?'K':'m',mainRows=[`${ko?'시각':'Time'}: ${fmt(i.time)} s`,`${ko?'초기 진폭 일치':'Initial amplitude match'}: ${flag(i.displacementMatches)}`,`${ko?'필요한 초기 속도 일치':'Required initial velocity match'}: ${flag(i.velocityMatches)}`,`${ko?'필요한 전체 초기 데이터 일치':'All required data match'}: ${flag(i.allRequiredDataMatch)}`,ko?'파랑 후보 · 주황 지정 데이터':'Blue candidate · orange prescribed data',i.mode===0?(ko?'열: 두 속도 입력 비활성':'Heat: both velocity inputs are inactive'):(ko?'파동: 변위와 속도 모두 필요':'Wave: displacement and velocity are both required')];
  first=<Plot curves={i.temporalCurves} range={4} title={ko?'시간에 따른 한 모드의 진폭':'One-mode amplitude over time'} xLabel="s" yLabel={`${ko?'진폭':'Amplitude'} · ${unit}`} rows={mainRows} marker="initial-time"/>;
  second=<Plot curves={i.spatialCurves} range={1} title={ko?'현재 시각의 공간 모양':'Spatial profile at the selected time'} xLabel="m" yLabel={`${i.mode===0?(ko?'온도 차':'Temperature difference'):(ko?'변위':'Displacement')} · ${unit}`} rows={[`t=${fmt(i.time)} s · L=1 m`,`${ko?'후보 진폭':'Candidate amplitude'}: ${fmt(i.query.q)} ${unit}`,`${ko?'지정 데이터 진폭':'Prescribed-data amplitude'}: ${fmt(i.target.q)} ${unit}`,`${ko?'후보 초기 시간 미분':'Candidate initial time derivative'}: ${fmt(i.candidateInitialTimeDerivative)} ${unit}/s`,`${ko?'지정 초기 시간 미분':'Prescribed initial time derivative'}: ${fmt(i.targetInitialTimeDerivative)} ${unit}/s`,ko?'한 시각의 일치는 전체 조건의 인증이 아닙니다.':'A query match does not certify the complete data.']} marker="initial-space"/>;
  caption=ko?'끝 값 0인 한 모드의 해석식입니다. 열과 파동의 단위 및 필요한 초기 데이터를 구분합니다.':'Analytic single modes with zero endpoint values. Heat and wave units and required initial data are distinct.';
 }else if(h){
  first=<Plot curves={h.spatialCurves} range={1} title={ko?'후보 온도와 독립 초기 데이터':'Candidate temperature and independent initial data'} xLabel="m" yLabel={ko?'온도 차 · K':'Temperature difference · K'} rows={[`t=${fmt(h.time)} s · κ=${fmt(h.diffusivity)} m²/s`,`${ko?'후보의 모든 시각 경계 만족':'All-time candidate boundary match'}: ${flag(h.candidateBoundaryMatches)}`,`${ko?'후보의 초기 함수 일치':'Candidate initial profile match'}: ${flag(h.initialProfileMatches)}`,`${ko?'지정 초기 모서리 양립':'Prescribed corner compatibility'}: ${flag(h.targetCornerCompatible)}`,ko?'파랑 현재 · 주황 후보 초기 · 초록 지정 초기':'Blue current · orange candidate initial · green prescribed',ko?'내부 방정식은 이 함수족에서 항상 만족합니다.':'This family always satisfies the interior equation.']} marker="heat-space"/>;
  second=<Plot curves={h.traceCurves} range={4} title={h.boundary===0?(ko?'양 끝의 값 추적':'Endpoint value traces'):(ko?'양 끝의 좌표 미분 추적':'Endpoint coordinate-derivative traces')} xLabel="s" yLabel={h.boundary===0?'u · K':'ux · K/m'} rows={[ko?'파랑 왼쪽 · 주황 오른쪽 · 목표 0':'Blue left · orange right · target0',`${ko?'모든 초기·경계 조건 일치':'All initial and boundary conditions match'}: ${flag(h.allDeclaredConditionsMatch)}`,ko?'지수 감쇠는 정확한 경계 판정을 바꾸지 않습니다.':'Decay does not change exact boundary flags.',ko?'모서리 판정은 요구 추적의 연속성을 가정합니다.':'Corner checks assume continuous required traces.',ko?'약해·mild 해의 부존재 판정이 아닙니다.':'This does not rule out weak or mild solutions.']} marker="heat-traces"/>;
  caption=ko?'현재 함수와 초기 함수는 서로 다른 시각의 양입니다. 모서리 양립은 명시한 정칙성에서 검사합니다.':'Current and initial profiles refer to different times. Corner compatibility is checked in the stated regularity class.';
 }else if(p){
  const status=p.solutionClass==='none'?(ko?'정적 해 없음':'No steady solution'):p.solutionClass==='affine-family'?(ko?'임의 상수족':'Arbitrary-constant family'):(ko?'유일한 해':'Unique solution');
  first=<Plot curves={p.curves} colors={[p.hasSolution?'blue':'orange']} range={1} title={p.hasSolution?(ko?'경계 데이터가 허용한 선택 해':'Selected solution allowed by boundary data'):(ko?'오른쪽 조건에 실패한 내부 후보':'Interior candidate failing the right datum')} xLabel="m" yLabel={ko?'온도 차 · K':'Temperature difference · K'} rows={[status,`u=−(${p.source})x²/2+(${fmt(p.a)})x+(${fmt(p.b)})`,`${ko?'해 존재':'Solution exists'}: ${flag(p.hasSolution)}`,`${ko?'미분 양립 잔차':'Derivative compatibility residual'}: ${p.compatibilityResidual}`,`${ko?'현재 후보 평균':'Current candidate mean'}: ${fmt(p.selectedMean)} K`,p.ungaugedConstantFree?'u(x)=−fx²/2+q₀x+C, C∈R':(ko?'그림의 후보와 해의 판정을 구분하세요.':'Separate the drawn candidate from solution status.')]} marker="poisson-profile"/>;
  const gradient=Array.from({length:81},(_,n)=>[n/80,p.a-p.source*n/80] as Vec2),targets:Vec2[]=p.boundary===2?[[0,p.left],[1,p.right]]:p.boundary===1?[[1,p.right]]:[];
  second=<Plot curves={[gradient]} colors={[p.hasSolution?'blue':'orange']} dots={targets} range={1} title={ko?'좌표 미분과 끝 데이터':'Coordinate derivative and endpoint data'} xLabel="m" yLabel="ux · K/m" rows={[ko?'초록 점: 지정한 미분 데이터':'Green dots: prescribed derivative data',`${ko?'왼쪽 좌표 미분':'Left coordinate derivative'}: ${fmt(p.leftDerivative)} K/m`,`${ko?'오른쪽 좌표 미분':'Right coordinate derivative'}: ${fmt(p.rightDerivative)} K/m`,`${ko?'왼쪽 바깥 법선 미분':'Left outward derivative'}: ${fmt(p.leftOutwardDerivative)} K/m`,`${ko?'평균 제약 활성':'Mean constraint active'}: ${flag(p.gaugeActive)}`,`${ko?'임의 상수 입력 활성':'Free constant control active'}: ${flag(p.offsetActive)}`,ko?'평균 제약으로 미분 양립 실패를 고칠 수 없습니다.':'A mean cannot repair incompatible derivatives.']} marker="poisson-gradient"/>;
  caption=ko?'양 끝 미분 입력은 같은 +x 방향입니다. 양립 실패 시 주황 후보를 정적 해로 인증하지 않습니다.':'Both derivative inputs use the +x direction. An orange incompatible candidate is not certified as a steady solution.';
 }else return null;
 return <div className="ai-game-figure">{[first,second].map((figure,n)=><figure className="ai-foundation-figure" key={n}><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={`${label} · ${n+1} · ${ko?'가로 이동 가능':'Horizontal scrolling available'}`}>{figure}</div><figcaption>{n===0?axis:caption}</figcaption></figure>)}
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'초기·경계 계산 표 · 가로 이동 가능':'Initial and boundary calculation table · Horizontal scrolling available'}><table><caption>{ko?'현재 조건의 계산 결과':'Calculation results for current conditions'}</caption><thead><tr>{result.table.headers.map(v=><th key={v} scope="col">{v}</th>)}</tr></thead><tbody>{result.table.rows.map((row,n)=><tr key={n}>{row.map((v,j)=><td key={j}>{typeof v==='number'?fmt(v):v}</td>)}</tr>)}</tbody></table></div>
 </div>;
}
