'use client';
import {useLanguage}from'@/components/language-provider';
import type {ElectromagnetismResult}from'@/lib/learning-electromagnetism';
import type {Vec2}from'@/lib/learning-simulations';
import {WavePlot, waveNumber as fmt, type Series, type Mark}from'./waves-plot';
import ElectroMagnetostaticsFigure from './electro-magnetostatics-figure';
import ElectromagneticEffectiveFigure from './electromagnetic-effective-models-figure';
import CircuitSensorsFigure from './circuit-sensors-figure';
import GaugeFreedomFigure from './gauge-freedom-figure';
import MaterialInterfacesFigure from './material-interfaces-figure';
import PoyntingEnergyFigure from './poynting-energy-figure';
import InductionWavesFigure from './induction-waves-figure';
import MaxwellFigure from './maxwell-figure';
import PotentialsFigure from './potentials-figure';

export default function ElectromagnetismFigure({result}:{result:ElectromagnetismResult}){
 const{language}=useLanguage(), t=(ko:string, en:string)=>language==='ko'?ko:en;
 const split=(points:[number, number|null][])=>{const segments:Vec2[][]=[];let part:Vec2[]=[];for(const[x, y]of points){if(y===null||!Number.isFinite(y)){if(part.length)segments.push(part);part=[];}else part.push([x, y]);}if(part.length)segments.push(part);return segments;};
 const line=(name:string, points:[number, number|null][], color='blue'):Series=>({name, segments:split(points), color});
 const plot=(marker:string, title:string, series:Series[], xLabel:string, yLabel:string, caption:string, dots:Mark[]=[], vertical:number[]=[])=>{
  const values=[...series.flatMap(s=>s.segments.flat().map(p=>Math.abs(p[1]))),...dots.map(d=>Math.abs(d.point[1]))], magnitude=Math.max(0,...values);
  // Normalize only for an explicitly disclosed display scale at extreme ranges.
  // Original physical quantities remain in the model and numeric tables.
  const scale=magnitude>1e100||magnitude>0&&magnitude<1e-100?magnitude:1;
  const drawn=series.map(s=>({...s, segments:s.segments.map(segment=>segment.map(([x, y])=>[x, y/scale]as Vec2))}));
  return <div key={marker}><WavePlot marker={marker} title={title} series={drawn} xLabel={xLabel} yLabel={scale===1?yLabel:`y / s · ${yLabel.split(' · ').at(-1)}`} caption={scale===1?caption:`${caption} ${t('표시 배율','Display scale')} s=${fmt(scale)}; ${t('실제 값 = 세로 좌표 × s.','physical value = ordinate × s.')}`} dots={dots.map(d=>({...d, point:[d.point[0], d.point[1]/scale]as Vec2}))} vertical={vertical} minimumYPadding={1e-100}/></div>;
 };
 const table=(marker:string, title:string, headers:string[], rows:(number|string|null)[][])=><figure key={marker} data-em-table={marker}><div className="ai-discrete-table" tabIndex={0} role="region" aria-label={title}><table><caption>{title}</caption><thead><tr>{headers.map((h, i)=><th key={i} scope="col">{h}</th>)}</tr></thead><tbody>{rows.map((row, i)=><tr key={i}>{row.map((v, j)=>j===0?<th key={j} scope="row">{typeof v==='number'?fmt(v):v}</th>:<td key={j}>{typeof v==='number'||v===null?fmt(v):v}</td>)}</tr>)}</tbody></table></div></figure>;
 const state=(s:string)=>s==='finite'?t('유한한 값','Finite value'):s==='zero-source'?t('영원천','Zero source'):s==='source-singularity'?t('원천 특이점','Source singularity'):s==='numerical-underflow'?t('수치 underflow','Numeric underflow'):t('수치 표현 범위','Numerical range');
 const arrow=(x:number, y:number, dx:number, dy:number, color:string, key:string)=>{const length=Math.hypot(dx, dy);if(!Number.isFinite(length)||length===0)return null;const ux=dx/length, uy=dy/length, head=Math.min(5, length/3), ex=x+dx, ey=y+dy;return <g key={key} stroke={`var(--learning-${color})`} fill="none"><path d={`M${x} ${y}L${ex} ${ey}`} strokeWidth="1.6"/><path d={`M${ex-head*ux+head*.55*uy} ${ey-head*uy-head*.55*ux}L${ex} ${ey}L${ex-head*ux-head*.55*uy} ${ey-head*uy+head*.55*ux}`} strokeWidth="1.6"/></g>;};
 const frame=(marker:string, title:string, children:React.ReactNode, caption:string)=><figure key={marker} className="ai-foundation-figure"><h3>{title}</h3><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={title}><svg viewBox="0 0 350 345" role="img" aria-label={title} data-waves-plot={marker}>{children}</svg></div><figcaption>{caption}</figcaption></figure>;
 if(result.unavailable)return <p role="status">{t('이 입력은 SI 단위 변환 또는 원천 위치 구성에서 숫자 표현 범위를 벗어납니다. 입력을 0이나 원천 충돌로 바꾸지 않고 물리 관측을 미정으로 유지합니다.','This input exceeds numeric representation during SI conversion or source positioning. Physical observations remain unavailable; the input is not replaced by zero or a source collision.')}</p>;
 if(result.statics)return <div data-electromagnetism-kind={result.kind}><ElectroMagnetostaticsFigure lab={result.statics}/></div>;
 if(result.transmissionLine||result.layeredMedium)return <div data-electromagnetism-kind={result.kind}><ElectromagneticEffectiveFigure result={result}/></div>;
 if(result.rcCircuit||result.coilSensor)return <div data-electromagnetism-kind={result.kind}><CircuitSensorsFigure result={result}/> </div>;
 if(result.gaugeTransform||result.lorenzConstraint)return <div data-electromagnetism-kind={result.kind}><GaugeFreedomFigure result={result}/></div>;
 if(result.dielectricBoundary||result.normalInterface)return <div data-electromagnetism-kind={result.kind}><MaterialInterfacesFigure result={result}/></div>;
 if(result.sourceEnergy||result.pulseEnergy)return <div data-electromagnetism-kind={result.kind}><PoyntingEnergyFigure result={result}/></div>;
 if(result.inductionWave||result.transversePulse)return <div data-electromagnetism-kind={result.kind}><InductionWavesFigure result={result}/></div>;
 if(result.planeMaxwell||result.radialMaxwell)return <div data-electromagnetism-kind={result.kind}><MaxwellFigure result={result}/></div>;
 if(result.scalarPotential||result.vectorPotential)return <div data-electromagnetism-kind={result.kind}><PotentialsFigure result={result}/></div>;
 const panels:React.ReactNode[]=[];
 if(result.charges){
  const r=result.charges, q=r.query, X=(x:number)=>190+30*x, Y=(y:number)=>148-30*y;
  panels.push(frame('charge-field-plane', t('원천·관측점과 전기장 단면','Sources, observation and a field slice'),<>
   <path d="M70 148H310M190 28V268" stroke="var(--learning-neutral)" strokeDasharray="3 4"/>
   {r.grid.map((p, i)=>{const e=p.value.field, m=p.value.fieldMagnitude;if(!e||!m)return null;const length=19*(m/(r.arrowReference+m));return arrow(X(p.x), Y(p.y), length*(e[0]/m),-length*(e[1]/m),'blue', String(i));})}
   {r.sources.map((s, i)=><g key={i}><circle cx={X(s.position[0])} cy={Y(s.position[1])} r={i?5:8} fill={`var(--learning-${i?'orange':'green'})`} stroke="hsl(var(--card))"/><text x={X(s.position[0])+10} y={Y(s.position[1])+18} fill="currentColor" fontSize="10">q{i+1}</text></g>)}
   <circle cx={X(r.observation[0])} cy={Y(r.observation[1])} r="4" fill="hsl(var(--card))" stroke="currentColor" strokeWidth="2"/>
   {[-4,0,4].map(x=><text key={x} x={X(x)} y="287" textAnchor="middle" fill="currentColor" fontSize="10">{x}</text>)}
   {[-4,0,4].map(y=><text key={y} x="64" y={Y(y)+3} textAnchor="end" fill="currentColor" fontSize="10">{y}</text>)}
   <text x="190" y="309" textAnchor="middle" fill="currentColor" fontSize="10">x · m</text><text x="190" y="332" textAnchor="middle" fill="currentColor" fontSize="10">y · m</text>
  </>, t('삼차원 Coulomb 장의 평면 단면입니다. 화살표 길이는 표시용으로 압축하며 실제 크기는 표에서 읽습니다. 빈 화살표는 영장 또는 미정일 수 있으므로 상태를 확인하세요.','A planar slice of the 3D Coulomb field. Arrow lengths are compressed for display; read actual magnitudes in the tables. Missing arrows can mean zero or unavailable field: inspect the states.')));
  const vectors=[...r.sourceFields, q.field??[null, null]], maximum=Math.max(0,...vectors.map(v=>v.every(x=>x!==null&&Number.isFinite(x))?Math.max(...v.map(x=>Math.abs(x!))):0));
  panels.push(frame('charge-vector-sum', t('관측점의 원천별 벡터와 합','Source contributions and their vector sum'),<>
   <path d="M70 148H310M190 28V268" stroke="var(--learning-neutral)" strokeDasharray="3 4"/>
   {q.field&&vectors.map((v, i)=>v[0]!==null&&v[1]!==null&&maximum>0?arrow(190,148, v[0]/maximum*105,-v[1]/maximum*105,['blue','orange','green'][i], String(i)):null)}
   <text x="190" y="287" textAnchor="middle" fill="currentColor" fontSize="11">{q.field?(maximum===0?'E = 0':'E1 + E2 = E'):t('상태 표를 확인하세요','See the state table')}</text>
   <text x="190" y="309" textAnchor="middle" fill="currentColor" fontSize="10">{t('파랑 E1 · 주황 E2 · 초록 합','Blue E1 · orange E2 · green sum')}</text>
   <text x="190" y="332" textAnchor="middle" fill="currentColor" fontSize="10">{t('공통 선형 표시 배율','One shared linear display scale')}</text>
  </>, t('세 벡터에 같은 선형 배율을 적용합니다. 방향을 먼저 비교하고 실제 성분은 원래 단위의 표에서 확인하세요.','All three vectors share one linear scale. Compare directions, then inspect original components and units in the table.')));
  const poles=r.observation[1]===0?r.sources.filter(s=>s.charge!==0).map(s=>s.position[0]):[];
  const sliced=(fn:(p:typeof r.slice[number])=>number|null)=>{const points:[number, number|null][]=[];let previous:number|undefined;for(const p of r.slice){if(previous!==undefined&&poles.some(c=>c>previous!&&c<p.x))points.push([(previous+p.x)/2, null]);points.push([p.x, p.displayExcluded?null:fn(p)]);previous=p.x;}return points;};
  panels.push(plot('charge-field-slice', t('관측 높이에서의 장 성분','Field components at the observation height'),[line('E x', sliced(p=>p.value.field?.[0]??null)), line('E y', sliced(p=>p.value.field?.[1]??null),'orange')],'x · m','E · N/C', t('높이 y를 고정한 0.05 m 간격의 그림 표본입니다. y=0에서는 표시만 원천 ±0.0125 m를 제외합니다. 관측값은 원래 식으로 계산합니다.','Plot samples at fixed y, spaced 0.05 m. At y=0, only the display omits ±0.0125 m around sources. Query values use the original law.')));
  panels.push(plot('charge-potential-slice', t('같은 단면의 스칼라 전위','Scalar potential on the same slice'),[line('V', sliced(p=>p.value.potential))],'x · m','V · V', t('전위는 부호 있는 스칼라입니다. 위 장 그림과 같은 표시 제외 구간을 사용하며 전위0과 장0은 다릅니다.','Potential is a signed scalar, with the same display omissions as the field slice. Zero potential differs from zero field.')));
  panels.push(plot('charge-force-slice', t('시험 전하에 작용하는 힘','Force on the test charge'),[line('F x', sliced(p=>p.forceNano[0])), line('F y', sliced(p=>p.forceNano[1]),'orange')],'x · m','F · nN', t('시험 전하는 장이 아닌 F=q test E를 바꿉니다. 같은 표시 제외 구간을 쓰며 미정 힘을0으로 대체하지 않습니다.','The test charge changes F=q test E, not the field. The same display gaps apply; unavailable force is not replaced by zero.')));
  panels.push(plot('charge-single-source-distance', t('첫 원천만의 거리별 장 크기','Distance dependence of source 1 alone'),[line('|E1|', r.radial.map(p=>[p.radius, p.fieldMagnitude]))],'r · m','|E1| · N/C', t('첫 원천만 비교하는 별도 준비입니다. 거리는0.25 m부터이며 r=0의 값을 그린 것이 아닙니다.','A separate comparison of source 1 alone. Distances start at 0.25 m; no r=0 value is drawn.')));
  panels.push(plot('charge-single-source-potential', t('첫 원천만의 거리별 전위','Distance dependence of source 1 potential'),[line('V1', r.radial.map(p=>[p.radius, p.potential]))],'r · m','V1 · V', t('장 크기는1/r², 부호 있는 전위는1/r입니다. 서로 다른 단위를 한 세로축에 섞지 않습니다.','Field magnitude follows 1/r²; signed potential follows 1/r. Their different units are not mixed on one ordinate.')));
  panels.push(table('charge-contributions', t('원천별 원래 성분','Original source contributions'),['i','q · nC','r · m','E x · N/C','E y · N/C', t('상태','State')], r.sources.map((s, i)=>[i+1, s.charge*1e9, q.contributions[i].distance, r.sourceFields[i][0], r.sourceFields[i][1], state(r.sourceFields[i].some(x=>x===null)&&q.contributions[i].status==='finite'?'numerical-range':q.contributions[i].status)])));
  panels.push(table('charge-observables', t('장·전위·힘의 별도 관측','Separate field, potential and force observations'),[t('항목','Quantity'), t('값','Value'), t('단위·의미','Unit or meaning')],[['E x', q.field?.[0]??null,'N/C'],['E y', q.field?.[1]??null,'N/C'],['|E|', q.fieldMagnitude,'N/C'],['V', q.potential,'V'],['F x', q.force?.[0]??null,'N'],['F y', q.force?.[1]??null,'N'],[t('장 상태','Field state'), state(q.fieldStatus),''],[t('전위 상태','Potential state'), state(q.potentialStatus),''],[t('힘 상태','Force state'), state(q.forceStatus),''],['k', q.constant, t('반올림한 N m²/C²','Rounded N m²/C²')]]));
  panels.push(<p key="field-scope" role="status">{t('정지한 지정 원천·삼차원 커널의 평면 관측입니다. 화면 화살표 압축은 물리 법칙이 아닙니다. 이 평면의 두 성분에 이차원 무원천 발산 조건을 강제하지 마세요.','Static prescribed sources with a planar observation of the 3D kernel. Arrow compression is not a physical law. Do not enforce a 2D source-free divergence constraint on this slice.')}</p>);
 }
 if(result.transport){
  const r=result.transport, q=r.query, b=r.balance, edges=[r.left, r.right];
  const transportScaled=(value:number|null,factor=1)=>value===null?null:value*factor;
  panels.push(plot('charge-density-transport', t('이동 밀도와 초기 준비','Transported density and initial preparation'),[line(t('현재 밀도','Current density'), r.spatial.map(p=>[p.x, transportScaled(p.current.density, 1e6)])), line(t('초기 밀도','Initial density'), r.spatial.map(p=>[p.x, transportScaled(p.initial.density, 1e6)]),'orange')],'x · m','rho · µC/m³', t('세로선은 고정 제어 구간입니다. 화면을 벗어난 Gaussian 꼬리도 원래 모델에 남습니다.','Vertical lines bound the fixed control interval. Off-screen Gaussian tails remain in the model.'),[], edges));
  panels.push(plot('charge-current-density', t('부호 있는 전류 밀도','Signed current density'),[line('J x', r.spatial.map(p=>[p.x, transportScaled(p.current.currentDensity, 1e6)]))],'x · m','J · µA/m²', t('음전하의 속도와 전류 방향은 반대일 수 있습니다. 밀도와 전류 밀도의 단위도 다릅니다.','Negative-charge velocity can oppose conventional current. Density and current density also have different units.'),[], edges));
  panels.push(plot('charge-control-volume', t('고정 구간에 저장된 전하','Charge stored in a fixed interval'),[line('Q inside', r.temporal.map(p=>[p.time, transportScaled(p.chargeInside, 1e6)]))],'t · s','Q · µC', t('분포를 보기 창 안에서 다시 정규화하지 않습니다. 점은 선택한 시각입니다.','The distribution is not renormalized inside the view. The dot marks the selected time.'),r.interval.chargeInside===null?[]:[{point:[r.time, r.interval.chargeInside*1e6], color:'orange'}]));
  panels.push(plot('charge-boundary-currents', t('양쪽 경계의 양의 x 방향 전류','Positive-x currents at both boundaries'),[line('A J left', r.temporal.map(p=>[p.time,transportScaled(p.leftOutwardCurrent, -1e6)])), line('A J right', r.temporal.map(p=>[p.time, transportScaled(p.rightOutwardCurrent, 1e6)]),'orange')],'t · s','I · µA', t('순유입은 왼쪽 값−오른쪽 값입니다. 왼쪽의 바깥 법선은−x입니다.','Net inflow is left minus right. The outward normal on the left points toward−x.')));
  panels.push(plot('charge-integrated-balance', t('유한 시간 변화의 두 계산','Two calculations of finite-time charge change'),[line(`Delta Q · ${t('점','point')}`,[[1, transportScaled(b.chargeChange, 1e6)]]), line(`integral inflow · ${t('점','point')}`,[[2, transportScaled(b.integratedNetInflow, 1e6)]],'orange')],'method · 1 / 2','change · µC', t('1은 끝점 전하 차,2는 시간 적분입니다. 같은 좌표가 아닌 두 방법을 점으로 비교하며 잔차는 표에 남깁니다.','1 is the endpoint charge change;2 is the time integral. Discrete points compare two methods; the table retains the residual.')));
  const patchMicro=r.surface.current===null?null:r.surface.current*1e6;
  const jSign=q.currentDensity===null?null:Math.sign(q.currentDensity), nx=r.surface.normal[0], ny=r.surface.normal[1];
  panels.push(frame('charge-directed-surface', t('전류 방향과 선택한 면 법선','Current direction and the chosen surface normal'),<>
   <path d="M70 148H310M190 28V268" stroke="var(--learning-neutral)" strokeDasharray="3 4"/>
   <path d={`M${190+70*ny} ${148+70*nx}L${190-70*ny} ${148-70*nx}`} stroke="var(--learning-orange)" strokeWidth="4"/>
   {arrow(190,148,85*nx,-85*ny,'green','normal')}{jSign!==null&&arrow(190,148,105*jSign,0,'blue','current')}
   <text x="190" y="287" textAnchor="middle" fill="currentColor" fontSize="11">{jSign===null?t('J 사용 불가','J unavailable'):jSign?'J · n':'J = 0'}</text>
   <text x="190" y="309" textAnchor="middle" fill="currentColor" fontSize="10">{t('파랑 J · 초록 n · 주황 면','Blue J · green n · orange surface')}</text>
   <text x="190" y="332" textAnchor="middle" fill="currentColor" fontSize="10">{t('방향 그림 · 실제 값은 표','Direction diagram · values in table')}</text>
  </>, t('독립 측정 면입니다. 회전한 법선이 제어 구간 양끝의 바깥 법선을 바꾸지는 않습니다. 면적0이면 전류0입니다.','An independent measurement patch. Rotating its normal does not alter control-volume end normals. Zero patch area gives zero current.')));
  panels.push(plot('charge-surface-angle', t('법선 회전과 보고 전류','Normal rotation and reported current'),[line('I patch', r.angular.map(p=>[p.angle, p.current===null?null:p.current*1e6]))],'theta · rad','I patch · µA', t('같은 전류 밀도에서 법선 방향만 바꾸는 비교입니다.','Only the patch normal changes; the local current-density vector stays fixed.'),patchMicro===null?[]:[{point:[r.normalAngle, patchMicro], color:'orange'}]));
  panels.push(plot('charge-local-continuity', t('국소 보존의 해석 미분','Analytic derivatives in local continuity'),[line('rho t', r.spatial.map(p=>[p.x, transportScaled(p.current.timeDerivative, 1e6)])), line('-div J', r.spatial.map(p=>[p.x,transportScaled(p.current.currentDivergence, -1e6)]),'orange')],'x · m','rate · µC/(m³ s)', t('두 해석 항이 같은 곡선입니다. 구성상 일치하며 독립 검증을 대신하지 않습니다.','These analytic terms coincide by construction. Their agreement does not replace independent checks.')));
  panels.push(table('charge-current-observables', t('원래 밀도·전류·면 관측','Original density, current and surface observations'),[t('항목','Quantity'), t('값','Value'), t('단위·의미','Unit or meaning')],[['Q total', r.system.totalCharge*1e6,'µC'],['A stream', r.system.area,'m²'],['rho query', transportScaled(q.density, 1e6),'µC/m³'],['J query', transportScaled(q.currentDensity, 1e6),'µA/m²'],[t('밀도 상태','Density state'),state(q.densityStatus),''],[t('전류 밀도 상태','Current-density state'),state(q.currentDensityStatus),''],['A patch', r.patchArea,'m²'],['n x', nx,'1'],['n y', ny,'1'],['I patch', patchMicro,'µA'],[t('면 통과 전류 상태','Patch-current state'),state(r.surface.status),''],['left outward', transportScaled(r.interval.leftOutwardCurrent, 1e6),'µA'],['right outward', transportScaled(r.interval.rightOutwardCurrent, 1e6),'µA'],['net inflow', transportScaled(r.interval.netInflow, 1e6),'µA']]));
  panels.push(table('charge-balance', t('별도로 계산한 전하 수지','Separately computed charge balance'),[t('항목','Quantity'), t('값','Value'), t('단위·의미','Unit or meaning')],[['start time',0,'s'],['selected time', r.time,'s'],['Q start', transportScaled(b.initial.chargeInside, 1e6),'µC'],['Q selected', transportScaled(b.final.chargeInside, 1e6),'µC'],['Delta Q', transportScaled(b.chargeChange, 1e6),'µC'],['integral inflow', transportScaled(b.integratedNetInflow, 1e6),'µC'],['residual', b.balanceResidual,'C'],['flow order difference', b.flowQuadratureDifference,'C']]));
  panels.push(table('charge-quadrature', t('모델·꼬리·구적 진단','Model, tail and quadrature diagnostics'),[t('항목','Quantity'), t('값','Value'), t('단위·의미','Unit or meaning')],[['left', r.left,'m'],['right', r.right,'m'],['fraction', r.interval.fraction,'1'],[t('질량 상태','Mass state'), state(r.interval.fractionStatus),''],['order difference', r.interval.quadratureDifference,'1'],['panels', r.interval.quadraturePanels,'1'],['rho t', q.timeDerivative,'C/(m³ s)'],['div J', q.currentDivergence,'C/(m³ s)'],['local residual', q.localContinuityResidual, t('해석 항등식','Analytic identity')]]));
  panels.push(<p key="current-scope" role="status">{t('고정된 폭·단면적의 지정 수송입니다. 구적 차이는 엄밀한 오차 보증이 아니며 국소0은 해석 항등식입니다. 힘·구성 법칙·Maxwell 장의 자기 일관성은 계산하지 않습니다.','Prescribed transport with fixed width and cross-section. Quadrature differences are not certified error bounds; the local zero is an analytic identity. Forces, constitutive laws and self-consistent Maxwell fields are not computed.')}</p>);
 }
 return <div data-electromagnetism-kind={result.kind}>{panels}</div>;
}
