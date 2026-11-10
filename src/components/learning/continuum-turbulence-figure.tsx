'use client';
import { useLanguage } from '@/components/language-provider';
import type { ContinuumResult } from '@/lib/learning-continuum';
import type { TurbulenceObservation } from '@/lib/learning-continuum-turbulence';
import type { Vec2 } from '@/lib/learning-simulations';
import { WavePlot, waveNumber as fmt, type Series } from './waves-plot';
type TurbulenceSeries = Series & { extent: [number, number] };
type Point = [number|null, number|null];
export default function ContinuumTurbulenceFigure({result,conceptStep}:{result:ContinuumResult;conceptStep?:number}) {
  const {language}=useLanguage(),t=(ko:string,en:string)=>language==='ko'?ko:en,panels:React.ReactNode[]=[];
  const prefix=result.reynolds?'continuum-reynolds':'continuum-gaussian-filter';
  const line=(name:string,points:Point[],color='blue',discrete=false):TurbulenceSeries=>{
    const xs=points.map(([x])=>x).filter((x):x is number=>x!==null&&Number.isFinite(x));
    const segments:Vec2[][]=[];let current:Vec2[]=[];
    for(const [x,y]of points){if(x===null||y===null||!Number.isFinite(x)||!Number.isFinite(y)){if(current.length)segments.push(current);current=[];}else if(discrete)segments.push([[x,y]]);else current.push([x,y]);}
    if(current.length)segments.push(current);return{name,color,segments,extent:[Math.min(0,...xs),Math.max(0,...xs)]};
  };
  const plot=(id:string,title:string,series:TurbulenceSeries[],xLabel:string,unit:string,caption:string,vertical:number[]=[])=>{
    const left=Math.min(0,...series.map(s=>s.extent[0])),right=Math.max(0,...series.map(s=>s.extent[1]));
    const magnitude=Math.max(0,...series.flatMap(s=>s.segments.flat().map(p=>Math.abs(p[1])))),scale=magnitude>1e5||magnitude>0&&magnitude<.001?magnitude:1;
    return <WavePlot key={id} marker={`${prefix}-${id}`} title={title} series={series.map(s=>({...s,segments:s.segments.map(part=>part.map(([x,y])=>[x,y/scale]as Vec2))}))} xDomain={[left,right>left?right:left+.01]} xLabel={xLabel} yLabel={scale===1?unit:`y × ${fmt(scale)} · ${unit}`} minimumYPadding={magnitude===0?.01:1e-100} vertical={vertical} caption={caption+(scale===1?'':` ${t('실제 값 = 세로 좌표 ×','Physical value = ordinate ×')} ${fmt(scale)}.`)}/>;
  };
  const state=(q:TurbulenceObservation|null)=>q===null?t('현재 조건에서 미정','Undefined at current conditions'):q.sign===0?t('정확한0','Exact zero'):q.value===null?t('비영·일반 값 불가','Nonzero; linear value unavailable'):fmt(q.value);
  type Row=[string,TurbulenceObservation|null,string];
  const table=(id:string,title:string,rows:Row[])=><figure key={id} data-continuum-table={`${prefix}-${id}`}><div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption>{title}</caption><thead><tr>{[t('관측','Observation'),t('일반 값·상태','Linear value or state'),t('부호','Sign'),'ln |q / unit|',t('단위','Unit')].map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{rows.map(([name,q,unit],i)=><tr key={i}><th scope="row">{name}</th><td>{state(q)}</td><td>{q===null?'—':q.sign}</td><td>{fmt(q?.logMagnitude)}</td><td>{unit}</td></tr>)}</tbody></table></div></figure>;
  const note=(id:string,text:string)=><p key={id} className="ai-logic-statement" data-turbulence-note={id}>{text}</p>;
  if(result.reynolds){
    const r=result.reynolds,q=r.query,b=r.budget;
    const spatial=(name:string,get:(v:typeof q)=>number|null,color='blue')=>line(name,r.spatial.map(s=>[s.xi,get(s.value)]),color);
    panels.push(plot('ensemble',t('실제8위상 속도 편차','Departures of eight actual phases'),[line('u′ / v',r.ensemble.map(s=>[s.value.uPrime.value,s.value.v.value]),'blue',true)],'u′ · m/s','v · m/s',t('같은 높이·같은 수평 위치에서8개 균등 앙상블 위상을 표본화합니다. 점들의 곱의 평균이Ruv입니다. 위상은 시간이 아닙니다.','Eight equally weighted ensemble phases at one height and streamwise position. Mean pointwise products give Ruv. Phase is not time.')));
    panels.push(plot('forcing',t('같은 평균에 필요한 실제 외력','Actual forcing required at the same mean'),[
      spatial('Gx',v=>v.meanForceX.value),spatial('Gy',v=>v.meanForceY.value,'orange'),line(t('영 섭동Gx','Zero-departure Gx'),r.spatial.map(s=>[s.xi,s.quiet.meanForceX.value]),'green')],
      'ξ = y/H','N/m³',t('영 섭동 대조는 같은U를 유지하지만 공분산·외력은 다릅니다. 압력0으로 만든 전체2차원 강제이며 외력 없는DNS가 아닙니다.','The zero-departure comparison keeps U while covariance and forcing differ. This is a complete two-dimensional manufactured force at zero pressure, not an unforced DNS.')));
    panels.push(plot('viscosity',t('역산 동점성의 정의 가능한 구간','Defined intervals of inferred viscosity'),[
      {name:'−Ruv/Uy',color:'blue',extent:[0,1],segments:r.viscositySegments.map(branch=>branch.map(s=>[s.xi,s.value.inferredEddyViscosity!.value!]as Vec2))}],'ξ = y/H','m²/s',t('전단0/비영 응력에는 유한 값이 없고 둘 다0이면 비가 미정입니다. 정의된 비영 값도 일반 숫자로 표현할 수 없으면 선을 잇지 않습니다.','Zero shear with nonzero stress admits no finite value; two zeros leave an indeterminate ratio. A defined nonzero value outside linear range also leaves a gap.'),[r.xi,.5]));
    panels.push(plot('budget',t('평균 비대칭과 전체 셀 일 수지','Mean asymmetry and whole-cell work budget'),[
      line(t('외력 일','Forcing work'),r.asymmetry.map(s=>[s.c,s.value.meanForcingWork.value])),line(t('분자 손실','Molecular loss'),r.asymmetry.map(s=>[s.c,s.value.molecularMeanLoss.value]),'orange'),line(t('평균→요동','Mean→fluctuation'),r.asymmetry.map(s=>[s.c,s.value.meanToFluctuationTransfer.value]),'green')],
      'c','W',t('c를 바꾸면 평균 프로필과 실제 외력도 바뀝니다. 주기2π/k·높이H·폭W를 모두 적분한 전력입니다.','Changing c changes the mean profile and actual force. Powers integrate period2π/k, height H and span W.'),[r.parameters.meanAsymmetry]));
    panels.push(plot('velocity',t('한 구성원의 실제 속도','Actual velocity of one member'),[
      line('u',r.spatial.map(s=>[s.xi,s.point.u.value])),line('v',r.spatial.map(s=>[s.xi,s.point.v.value]),'orange'),spatial('U',v=>v.meanU.value,'green')],
      'ξ = y/H','m/s',t('모든 구성원은 발산0과 양쪽 벽 속도0을 만족합니다. 실제 속도와 평균을 구분하세요.','Every member satisfies zero divergence and zero velocity at both walls. Distinguish actual velocity from its mean.'),[r.xi]));
    panels.push(plot('covariance',t('세 공분산 성분','Three covariance components'),[
      spatial('Ruu',v=>v.covarianceUU.value),spatial('Rvv',v=>v.covarianceVV.value,'orange'),spatial('Ruv',v=>v.covarianceUV.value,'green')],
      'ξ = y/H','m²/s²',t('대각 분산은 비음수지만 교차 공분산은 부호가 있습니다. 중앙의 계수 감소가 모든 분산0을 뜻하지는 않습니다.','Diagonal variances are nonnegative; cross covariance is signed. Centre rank loss does not imply all variances vanish.')));
    panels.push(plot('stress',t('분자·Reynolds·총 평균 전단','Molecular, Reynolds and total mean shear'),[
      spatial('μUy',v=>v.molecularShearStress.value),spatial('−ρRuv',v=>v.reynoldsShearStress.value,'orange'),spatial('τ mean',v=>v.totalMeanShearStress.value,'green')],
      'ξ = y/H','Pa',t('공분산R은 밀도를 포함하지 않습니다. 운동량 응력의 부호·밀도 관례를 확인하세요.','Covariance R excludes density. Check stress signs and density conventions.')));
    panels.push(plot('covariance-defect',t('후보 공분산의 결손','Defect of candidate covariance'),[
      spatial('Ruv',v=>v.covarianceUV.value),spatial('−νm Uy',v=>v.modeledCovarianceUV.value,'orange'),spatial('δR',v=>v.covarianceDefect.value,'green')],
      'ξ = y/H','m²/s²',t('일정한 비음 후보 점성으로 실제 공분산을 비교합니다. 완전한3차원 응력 모델은 아닙니다.','Compare a constant nonnegative viscosity candidate with actual covariance. This is not a complete three-dimensional stress model.')));
    panels.push(plot('force-defect',t('실제 외력과 후보 외력의 결손','Actual/candidate forcing defect'),[
      spatial('G',v=>v.meanForceX.value),spatial('Gm',v=>v.modeledForceX.value,'orange'),spatial('G−Gm',v=>v.forceDefect.value,'green')],
      'ξ = y/H','N/m³',t('같은 평균을 유지하려면 공분산을 바꿀 때 외력도 바뀝니다. 작은 자기 잔차만으로 독립 인증하지 않습니다.','At fixed mean, replacing covariance also changes required forcing. A small self residual is not independent certification.')));
    panels.push(plot('local-power',t('국소 평균 에너지 전달과 손실','Local mean energy transfer and loss'),[
      spatial('−ρRuv Uy',v=>v.meanToFluctuationTransferDensity.value),spatial('μUy²',v=>v.molecularMeanLossDensity.value,'orange'),spatial('ρνm Uy²',v=>v.modeledTransferDensity.value,'green')],
      'ξ = y/H','W/m³',t('평균→요동 교환은 양·음일 수 있지만 분자 손실과 비음 후보의 전달은 비음수입니다.','Mean-to-fluctuation exchange can have either sign; molecular loss and nonnegative-candidate transfer are nonnegative.')));
    panels.push(plot('small',t('벽 근처의 비영 평균 속도','Nonzero mean velocity near the wall'),[
      line('ln |U|',r.small.map(s=>[s.logPosition,s.value.meanU.logMagnitude]))],'log10 ξ','ln(|U| / (m/s))',t('정확한 벽0에는 로그가 없습니다. c1에서는 평균 속도가이차로 시작하며 작은 양의 좌표를0으로 대체하지 않습니다.','Exact wall zero has no log. At c=1 mean velocity begins quadratically; tiny positive positions are not replaced by zero.')));
    panels.push(plot('phase',t('조회 높이의 실제 위상 변화','Actual phase variation at the query height'),[
      line('u′',r.phases.map(s=>[s.phase,s.value.uPrime.value])),line('v',r.phases.map(s=>[s.phase,s.value.v.value]),'orange')],
      'α / (2π)','m/s',t('연속 위상 프로필과8개 평균 표본을 구분하세요. 시간 진화나 애니메이션 속도가 아닙니다.','Distinguish continuous phase profiles from the eight averaging samples. This is not time evolution or animation speed.'),[r.phaseCycles]));
    if(conceptStep===undefined){
      panels.push(table('query',t('현재 앙상블 관측','Current ensemble observations'),[
        ['U',q.meanU,'m/s'],['Uy',q.meanShear,'1/s'],['u′',r.point.uPrime,'m/s'],['v',r.point.v,'m/s'],['Ruu',q.covarianceUU,'m²/s²'],['Rvv',q.covarianceVV,'m²/s²'],['Ruv',q.covarianceUV,'m²/s²'],['det R',q.covarianceDeterminant,'m⁴/s⁴'],['Gx',q.meanForceX,'N/m³'],['Gy',q.meanForceY,'N/m³'],['δR',q.covarianceDefect,'m²/s²'],['δG',q.forceDefect,'N/m³'],['−Ruv/Uy',q.inferredEddyViscosity,'m²/s'],['Re bulk',q.reynoldsNumber,'1']]));
      panels.push(table('budget',t('실제 전체 셀의 평균 예산','Mean budget of the actual whole cell'),[
        ['mean kinetic energy',b.meanKineticEnergy,'J'],['fluctuation kinetic energy',b.fluctuationKineticEnergy,'J'],['mean forcing work',b.meanForcingWork,'W'],['molecular mean loss',b.molecularMeanLoss,'W'],['mean→fluctuation',b.meanToFluctuationTransfer,'W'],['candidate forcing work',b.modeledForcingWork,'W'],['candidate transfer',b.modeledTransfer,'W'],['forcing work defect',b.forcingWorkDefect,'W']]));
      const inferred=q.inferredEddyViscosityStatus==='defined'?t('점별 비가 정의됩니다. 물성의 보편성을 뜻하지 않습니다.','The pointwise ratio is defined, not a universal material property.'):q.inferredEddyViscosityStatus==='indeterminate'?t('전단과 응력이 모두0이므로0/0 미정입니다.','Both shear and stress vanish: the ratio is indeterminate0/0.'):t('전단0에 비영 응력이 있어 유한 스칼라 점성으로 맞출 수 없습니다.','Nonzero stress at zero shear cannot be matched by finite scalar viscosity.');
      const re=q.reynoldsNumberStatus==='defined'?t('Re는 관성·점성 척도의 비이며 난류 자동 분류가 아닙니다.','Re compares inertial and viscous scales; it is not an automatic turbulence classifier.'):q.reynoldsNumberStatus==='indeterminate'?t('평균 유속과 분자 점성이 모두0이라Re 비가 미정입니다.','Zero bulk speed and molecular viscosity leave Re indeterminate.'):t('점성0·비영 유속이라 유한Re 비를 제공하지 않습니다.','Zero viscosity and nonzero speed admit no finite Re ratio.');
      panels.push(note('conditions',`${inferred} ${re} ${t('주기 길이','Period length')}: ${fmt(b.cellLength)} m; ${t('체적','volume')}: ${fmt(b.cellVolume)} m³. ${t('위상은 앙상블의 표지이며 시간은 아닙니다. 완전한 요동 외력 일 예산은 아직 별도 계산하지 않았습니다.','Phase labels ensemble members, not time. A full fluctuation forcing-work budget is not separately computed.')}`));
    }
  }
  if(result.gaussianFilter){
    const r=result.gaussianFilter,q=r.query,b=r.budget;
    const spatial=(name:string,get:(v:typeof q)=>number|null,color='blue')=>line(name,r.spatial.map(s=>[s.xCycles,get(s.value)]),color);
    const temporal=(name:string,get:(v:typeof b)=>number|null,color='blue')=>line(name,r.temporal.map(s=>[s.time,get(s.value)]),color);
    panels.push(plot('variance',t('실제 분산과 네 비음 항','Actual variance and four nonnegative terms'),[
      spatial('τ',v=>v.variance.value),...['even square','even remainder','odd square','odd remainder'].map((name,i)=>spatial(name,v=>v.positiveVarianceTerms[i].value,['orange','green','neutral','blue'][i]))],
      'x/L','m²/s²',t('정확한 유한 양의 표현이며 음수를0으로 자른 결과가 아닙니다. 폭0과 작은 비영 분산을 구별하세요.','An exact finite positive representation, not clipping negative results. Distinguish zero width from tiny nonzero variance.'),[r.xCycles]));
    panels.push(plot('residual',t('원래 잔차와 필터한 잔차','Original and filtered residuals'),[
      spatial('u−ū',v=>v.residualVelocity.value),spatial('overline(u−ū)',v=>v.filteredResidualVelocity.value,'orange')],
      'x/L','m/s',t('Gaussian은 멱등 투영이 아닙니다. 필터한 잔차가 보통0이 아니며 두 번 적용하면 계수가제곱됩니다.','A Gaussian is not an idempotent projection. The filtered residual generally remains; a second application squares the multiplier.')));
    panels.push(plot('transfer',t('부호 전달과 분자 손실','Signed transfer and molecular loss'),[
      spatial('Π',v=>v.transferPerMass.value),spatial('νūx²',v=>v.resolvedMolecularLossPerMass.value,'orange')],
      'x/L','m²/s³',t('Π=−τūx/2이므로 양의 분산만으로 전달 방향을 정할 수 없습니다. 음의 전달이 전체 가역성을 뜻하지는 않습니다.','Π=−τūx/2: positive variance alone does not determine transfer direction. Negative transfer does not imply overall reversibility.')));
    panels.push(plot('power',t('실제 시간의 해상 전력 수지','Resolved power budget over actual time'),[
      temporal('Fr',v=>v.resolvedForcingWork.value),temporal('P',v=>v.resolvedToResidualTransfer.value,'orange'),temporal('Dr',v=>v.resolvedMolecularLoss.value,'green'),temporal('Kr,t',v=>v.resolvedEnergyRate.value,'neutral')],
      't · s','W',t('현재 제조 가족에서는 Fr=P가 상쇄되어Kr,t=−Dr입니다. 외력의 일을 빼고 전달만으로 감쇠율을 해석하지 마세요.','This manufactured family has Fr=P, leaving Kr,t=−Dr. Transfer alone does not determine decay when forcing work is present.'),[r.time]));
    panels.push(plot('velocity',t('실제·필터·두 번 필터 속도','Actual, filtered and twice-filtered velocity'),[
      spatial('u',v=>v.velocity.value),spatial('ū',v=>v.filteredVelocity.value,'orange'),spatial('overline(ū)',v=>v.twiceFilteredVelocity.value,'green')],
      'x/L','m/s',t('u=C+v(x−Ct,t)이며 시간 미분에 이동 항도 포함합니다. C만 속도에 더하는 모델과 구별하세요.','u=C+v(x−Ct,t), including drift in the time derivative. This differs from adding C to velocity alone.'),[r.xCycles]));
    panels.push(plot('filter',t('물리 필터 폭과 모드 전달함수','Physical filter width and mode multipliers'),[
      line('g1',r.widths.map(s=>[s.width,s.value.firstTransferFunction.value])),line('g2',r.widths.map(s=>[s.width,s.value.secondTransferFunction.value]),'orange'),line('g1²',r.widths.map(s=>[s.width,s.value.firstTransferFunction.value===null?null:s.value.firstTransferFunction.value**2]),'green')],
      'ℓ · m','1',t('ℓ는Gaussian 표준편차이며 격자 간격·픽셀 수가 아닙니다. 모드 파수의 제곱으로 감쇠합니다.','ℓ is Gaussian standard deviation, not grid spacing or pixel count. Attenuation depends on squared mode wavenumber.'),[r.parameters.filterWidth]));
    panels.push(plot('gradient',t('응력 기울기와 실제 잔여 힘','Stress gradient and actual residual force'),[
      spatial('τx',v=>v.varianceGradient.value),spatial('−τx/2',v=>v.subgridForce.value,'orange')],
      'x/L','m/s²',t('Burgers의u²/2플럭스에 따른1/2입니다. 실제Navier–Stokes 응력 관례에 그대로 복사하지 않습니다.','The one-half comes from Burgers flux u²/2. Do not copy it into a Navier–Stokes stress convention.')));
    panels.push(plot('local-energy',t('국소 외력 일·변화율·플럭스 발산','Local forcing work, rate and flux divergence'),[
      spatial('ū fbar',v=>v.filteredForcingWorkPerMass.value),spatial('e,t',v=>v.resolvedEnergyRatePerMass.value,'orange'),spatial('Jx',v=>v.resolvedEnergyFluxGradient.value,'green')],
      'x/L','m²/s³',t('e,t+Jx=−νūx²+ū fbar−Π입니다. 작은 폭의 모멘트 항등식 자체는 독립 인증이 아닙니다.','e,t+Jx=−νūx²+ū fbar−Π. The small-width moment identity itself is not independent certification.')));
    panels.push(plot('energy',t('전체·해상·잔여 이차 에너지','Total, resolved and residual quadratic energy'),[
      temporal('K',v=>v.totalKineticEnergy.value),temporal('Kr',v=>v.resolvedKineticEnergy.value,'orange'),temporal('Ks',v=>v.residualKineticEnergy.value,'green')],
      't · s','J',t('rho·S는 스칼라 에너지의 기준 가중치입니다. 완전한 압축성 질량 모델이 아닙니다. C의 배경 에너지는 잔여 에너지와 구분합니다.','rho and S weight this scalar energy, not a complete compressible mass model. Distinguish the C background from residual energy.'),[r.time]));
    panels.push(plot('small',t('작은 양의 폭과 실제 분산 로그','Actual variance log at small positive width'),[
      line('ln τ',r.small.map(s=>[s.logWidth,s.value.variance.logMagnitude]))],
      'log10(ℓ / m)','ln(τ / (m²/s²))',t('폭0에는 로그가 없습니다. 단일 모드 극값의4차,초기A1/B−1/2·위상0의6차 법칙은 해당 조건에서만 적용합니다.','Zero width has no log. Fourth order at a single-mode extremum and sixth order at initial A=1/B=−1/2, phase0 apply only under those conditions.')));
    panels.push(plot('tail',t('넓은 필터의 강제·기울기 꼬리','Forcing and gradient tails under broad filtering'),[
      line('ln |fbar|',r.widths.map(s=>[s.width,s.value.filteredForce.logMagnitude])),line('ln |τx|',r.widths.map(s=>[s.width,s.value.varianceGradient.logMagnitude]),'orange')],
      'ℓ · m','ln(|q| / (m/s²))',t('분산이 평평하게 반올림되어도 작은 실제 기울기·외력은 비영일 수 있습니다. 정확한 영점은 선으로 잇지 않습니다.','A rounded-flat variance can retain nonzero gradient and forcing tails. Exact zeros are not bridged.'),[r.parameters.filterWidth]));
    panels.push(plot('frames',t('이동 좌표와 전체 전달의 불변성','Moving frames and invariant whole-cell transfer'),[
      line('P',r.frames.map(s=>[s.frameVelocity,s.value.resolvedToResidualTransfer.value])),line('Dr',r.frames.map(s=>[s.frameVelocity,s.value.resolvedMolecularLoss.value]),'orange')],
      'C · m/s','W',t('좌표 이동·시간 미분을 포함한 가족의 주기 전체 적분입니다. 고정 위치의 모든 국소 값이 같다는 뜻은 아닙니다.','Full-period integrals for the family including coordinate drift and time derivative. This does not make all fixed-position local values equal.'),[r.parameters.frameVelocity]));
    if(conceptStep===undefined){
      panels.push(table('query',t('현재 Gaussian 필터 관측','Current Gaussian-filter observations'),[
        ['A',q.firstAmplitude,'m/s'],['B',q.secondAmplitude,'m/s'],['ΔA stable',q.firstAmplitudeChange,'m/s'],['ΔB stable',q.secondAmplitudeChange,'m/s'],['g1',q.firstTransferFunction,'1'],['g2',q.secondTransferFunction,'1'],['u',q.velocity,'m/s'],['v departure',q.velocityDeparture,'m/s'],['ū',q.filteredVelocity,'m/s'],['ūx',q.filteredGradient,'1/s'],['u,t',q.velocityTimeRate,'m/s²'],['ū,t',q.filteredTimeRate,'m/s²'],['u−ū',q.residualVelocity,'m/s'],['overline(u−ū)',q.filteredResidualVelocity,'m/s'],['τ',q.variance,'m²/s²'],['τx',q.varianceGradient,'m/s²'],['−τx/2',q.subgridForce,'m/s²'],['filtered force',q.filteredForce,'m/s²'],['Π',q.transferPerMass,'m²/s³'],['molecular loss',q.resolvedMolecularLossPerMass,'m²/s³'],['forcing work',q.filteredForcingWorkPerMass,'m²/s³']]));
      panels.push(table('budget',t('전체·해상·잔여 에너지 예산','Total, resolved and residual energy budgets'),[
        ['K',b.totalKineticEnergy,'J'],['Kr',b.resolvedKineticEnergy,'J'],['Ks',b.residualKineticEnergy,'J'],['D',b.totalMolecularLoss,'W'],['Dr',b.resolvedMolecularLoss,'W'],['Ds',b.residualMolecularLoss,'W'],['P',b.resolvedToResidualTransfer,'W'],['F total',b.totalForcingWork,'W'],['Fr',b.resolvedForcingWork,'W'],['Fs',b.residualForcingWork,'W'],['K,t',b.totalEnergyRate,'W'],['Kr,t',b.resolvedEnergyRate,'W'],['Ks,t',b.residualEnergyRate,'W']]));
      panels.push(note('conditions',`${t('주기 길이','Period length')}: ${fmt(b.cellLength)} m; ${t('체적 기준','Volume weight')}: ${fmt(b.cellVolume)} m³. ${t('직접 제곱 뺄셈의 분산','Variance from direct rounded-square subtraction')}: ${fmt(q.ordinarySubtractionVariance)} m²/s². ${t('비영 양의 분산은 별도 안정적 관측을 사용합니다. 유한 두 모드의Gaussian 역연산은 수학적으로 가역이나 작은 계수에서 수치 조건이 나빠집니다.','Nonzero positive variance uses the separate stable observation. Gaussian inversion on the finite two-mode family is mathematically injective but poorly conditioned at small multipliers.')}`));
    }
  }
  return <div className="ai-matrix-panels">{conceptStep===undefined?panels:panels[conceptStep]}</div>;
}
