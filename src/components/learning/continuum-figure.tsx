'use client';
import ContinuumConservativeFigure from './continuum-conservative-figure';
import ContinuumTurbulenceFigure from './continuum-turbulence-figure';
import ContinuumMultiphysicsFigure from './continuum-multiphysics-figure';
import ContinuumDimensionlessFigure from './continuum-dimensionless-figure';
import ContinuumConstitutiveFigure from './continuum-constitutive-figure';
import ContinuumReactionFigure from './continuum-reaction-transport-figure';
import ContinuumDiffusionFigure from './continuum-diffusion-figure';
import ContinuumFlowFigure from './continuum-flow-figure';
import ContinuumAssumptionsFigure from './continuum-assumptions-figure';
import ContinuumStressStrainFigure from './continuum-stress-strain-figure';
import {useLanguage} from '@/components/language-provider';
import type {ContinuumResult} from '@/lib/learning-continuum';
import type {Vec2} from '@/lib/learning-simulations';
import {WavePlot, waveNumber as fmt, type Series} from './waves-plot';

export default function ContinuumFigure({result,conceptStep}:{result:ContinuumResult;conceptStep?:number}) {
  const {language}=useLanguage(),t=(ko:string,en:string)=>language==='ko'?ko:en,panels:React.ReactNode[]=[];
  if(result.volumeDiscretization||result.godunovDiscretization)return <ContinuumConservativeFigure result={result} conceptStep={conceptStep}/>;
  if(result.reynolds||result.gaussianFilter)return <ContinuumTurbulenceFigure result={result} conceptStep={conceptStep}/>;
  if(result.thermoelastic||result.thermoelectric)return <ContinuumMultiphysicsFigure result={result} conceptStep={conceptStep}/>;
  if(result.peclet||result.damkohler)return <ContinuumDimensionlessFigure result={result} conceptStep={conceptStep}/>;
  if(result.viscoelastic||result.neoHookean)return <ContinuumConstitutiveFigure result={result} conceptStep={conceptStep}/>;
  if(result.reversibleTransport||result.fisherTransport)return <ContinuumReactionFigure result={result} conceptStep={conceptStep}/>;
  if(result.heatDiffusion||result.gaussianDiffusion)return <ContinuumDiffusionFigure result={result} conceptStep={conceptStep}/>;
  if(result.acousticFlow||result.taylorGreenFlow)return <ContinuumFlowFigure result={result} conceptStep={conceptStep}/>;
  if(result.stressTraction||result.finiteDeformation)return <ContinuumStressStrainFigure result={result} conceptStep={conceptStep}/>;
  if(result.particleAveraging||result.scaleSeparation)return <ContinuumAssumptionsFigure result={result} conceptStep={conceptStep}/>;
  const prefix=result.massTransport?'continuum-mass':'continuum-gas';
  const line=(name:string,points:[number,number|null][],color='blue',discrete=false):Series=>{
    const segments:Vec2[][]=[];let part:Vec2[]=[];
    for(const[x,y]of points){if(y===null||!Number.isFinite(y)){if(part.length)segments.push(part);part=[];}else if(discrete)segments.push([[x,y]]);else part.push([x,y]);}
    if(part.length)segments.push(part);return{name,color,segments};
  };
  const plot=(id:string,title:string,series:Series[],axis:string,unit:string,caption:string,vertical:number[]=[])=>{
    const magnitude=Math.max(0,...series.flatMap(s=>s.segments.flat().map(p=>Math.abs(p[1]))));
    const scale=magnitude>1e5||magnitude>0&&magnitude<.001?magnitude:1;
    return <WavePlot key={id} marker={prefix+'-'+id} title={title}
      series={series.map(s=>({...s,segments:s.segments.map(part=>part.map(([x,y])=>[x,y/scale]as Vec2))}))}
      xLabel={axis} yLabel={scale===1?unit:'y × '+fmt(scale)+' · '+unit} minimumYPadding={magnitude===0?.01:1e-100}
      vertical={vertical} caption={scale===1?caption:caption+' '+t('실제 값 = 세로 좌표 ×','Physical value = ordinate ×')+' '+fmt(scale)+' '+unit+'.'}/>;
  };
  const table=(id:string,title:string,rows:(string|number|null)[][])=><figure key={id} data-continuum-table={prefix+'-'+id}>
    <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={title}><table><caption>{title}</caption>
      <thead><tr>{[t('항목','Quantity'),t('값','Value'),t('단위·의미','Unit or meaning')].map(h=><th key={h} scope="col">{h}</th>)}</tr></thead>
      <tbody>{rows.map((row,i)=><tr key={i}>{row.map((v,j)=>j===0?<th scope="row" key={j}>{v}</th>:<td key={j}>{typeof v==='number'||v===null?fmt(v):v}</td>)}</tr>)}</tbody>
    </table></div></figure>;
  const diagnostic=t('공유 해석 가족의 진단입니다. 독립 인증이나 엄밀한 오차 상한이 아닙니다.','Shared analytic-family diagnostics, not independent certificates or rigorous error bounds.');
  const geometryStatus=(s:string)=>s==='zero-length'?t('정확한 영폭: 실제 면 관측 없음','Exact zero width: no actual face observation'):s==='width-below-coordinate-resolution'?t('좌표 해상도 아래의 양의 폭','Positive width below coordinate resolution'):s==='arithmetic-range'?t('산술 범위: 기하 사용 불가','Arithmetic range: geometry unavailable'):t('유한한 기하','Finite geometry');
  const geometry=(left:number|null,right:number|null,particles:{initial:number;current:number}[],caption:string)=>{
    const X=(x:number)=>190+20*x,title=t('입자와 실제 관측 구간','Particles and actual observation interval');
    return <figure key="geometry" className="ai-foundation-figure"><h3>{title}</h3>
      <div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={title}>
      <svg viewBox="0 0 350 345" role="img" aria-label={title} data-waves-plot={prefix+'-geometry'}>
        <path d="M70 160H310" stroke="var(--learning-neutral)"/>
        {particles.map((p,i)=><g key={i}><path d={'M'+X(p.initial)+' 95L'+X(p.current)+' 155'} stroke="var(--learning-neutral)" strokeDasharray="3 4"/>
          <circle cx={X(p.initial)} cy="95" r="3" fill="var(--learning-neutral)"/><circle cx={X(p.current)} cy="155" r="4" fill="var(--learning-blue)"/></g>)}
        {left!==null&&right!==null?<><path d={'M'+X(left)+' 195V225H'+X(right)+'V195'} fill="none" stroke="var(--learning-orange)" strokeWidth="3"/>
          <path d={'M'+X(left)+' 160V245M'+X(right)+' 160V245'} stroke="var(--learning-orange)" strokeDasharray="3 4"/></>:null}
        <g fontSize="10" fill="currentColor"><text x="190" y="65" textAnchor="middle">{t('회색: 초기 위치 · 파랑: 현재 입자','Gray: initial position · Blue: current particles')}</text>
          <text x="190" y="270" textAnchor="middle">{t('주황: 현재 창 · 폭을 임의 확대하지 않음','Orange: current window · no width enlargement')}</text>
          <text x="70" y="290">−6</text><text x="310" y="290" textAnchor="end">6</text><text x="190" y="315" textAnchor="middle">x · m</text>
          <text x="190" y="337" textAnchor="middle">{t('세로 간격은 상태를 구분하는 도식','Vertical spacing separates states schematically')}</text></g>
      </svg></div><figcaption>{caption}</figcaption></figure>;
  };
  if(result.massTransport){
    const r=result.massTransport,q=r.query,b=r.balance;
    const time=(get:(x:typeof q)=>number|null,name:string,color='blue')=>line(name,r.temporal.map(p=>[p.time,get(p.value)]),color);
    panels.push(geometry(q.left,q.right,r.particles,t('같은 초기 입자의 이동과 현재 창을 비교합니다. 세로축은 물리 좌표·면적이 아닙니다. 영폭·해상도 손실은 아래 상태 표를 읽으세요.','Compare motion of the same initial particles with the current window. Vertical spacing is not physical position or area. Read zero-width/resolution states below.')));
    panels.push(plot('density',t('현재 밀도와 적분 시작의 밀도','Current density and density at the integration start'),[
      line(t('현재 밀도','Current density'),r.spatial.map(p=>[p.x,p.value.density])),
      line(t('시작 밀도','Start density'),r.spatial.map(p=>[p.x,p.initial.density]),'orange')],'x · m','kg/m³',
      t('세로선은 현재 창입니다. 무한 축의 유한 총질량이나 화면 끝의 자동 주기 경계를 가정하지 않습니다.','Vertical lines mark the current window. No finite whole-line mass or automatic periodic screen boundary is assumed.'),[q.left,q.right].filter((x):x is number=>x!==null)));
    panels.push(plot('storage',t('관측 창에 저장된 질량','Mass stored in the observation window'),[time(x=>x.mass,'M')],'t · s','kg',
      t('시간별 실제 창 안의 질량입니다. 같은 위치의 창과 같은 입자의 집합은 다릅니다.','Mass in the actual window at each time. A window at the same position differs from the same material set.')));
    panels.push(plot('faces',t('양쪽 바깥 법선의 상대 유출','Relative transport through outward faces'),[
      time(x=>x.leftOutward,t('왼쪽 바깥','Left outward')),time(x=>x.rightOutward,t('오른쪽 바깥','Right outward'),'orange')],'t · s','kg/s',
      t('왼쪽 법선은−x, 오른쪽은+x입니다. 음의 바깥 유출은 유입입니다. 정확한 영폭에서는 실제 면 관측을 만들지 않습니다.','Left normal is−x, right is+x. Negative outward transport is inflow. Exact zero width does not create actual face observations.')));
    panels.push(plot('relative-control',t('올바른 상대 유입과 잘못된 유속','Correct relative inflow and incorrect flux'),[
      time(x=>x.stableNetInflow,t('실제 상대 유입','Actual relative inflow')),
      time(x=>x.wrongLabFrameNetInflow,t('잘못된 실험실 유속','Incorrect laboratory flux'),'orange')],'t · s','kg/s',
      t('같은 실제 장·창에서 경계 속도를 누락한 대조입니다. 대조에 맞추려고 밀도를 바꾸지 않습니다.','A control omitting boundary velocity for the same actual field/window. Density is not changed to fit the control.')));
    panels.push(plot('rate-methods',t('저장 변화율과 두 유입 계산','Storage rate and two inflow evaluations'),[
      time(x=>x.massRate,t('해석 변화율','Analytic storage rate')),
      time(x=>x.ordinaryNetInflow,t('반올림 면 차이','Rounded face difference'),'orange')],'t · s','kg/s',diagnostic));
    panels.push(plot('finite-change',t('유한 시간 질량 변화의 세 관측','Three observations of finite-time mass change'),[
      line(t('독립 방법 점','Separate method points'),[[1,b.massChange],[2,b.integratedNetInflow],[3,b.roundedMassChange]],'blue',true)],
      '1=stable;2=time integral;3=rounded endpoints','kg',
      t('적분 시작과 관측 시각을 비교합니다. 점은 방법이며 시간 경로가 아닙니다.','Compare integration start and selected time. Points are methods, not a time trajectory.')));
    panels.push(plot('small-change',t('안정 변화와 반올림 차감','Stable change and rounded subtraction'),[
      line(t('서로 다른 계산','Separate evaluations'),[[1,b.roundedMassChange],[2,b.massChange]],'orange',true)],
      '1=rounded;2=stable','kg',t('작은 비영 변화와 실제 영시간·영대비·물질 창을 구분합니다.','Separate a tiny nonzero change from exact zero duration, zero contrast or a material window.')+' '+diagnostic));
    panels.push(plot('local',t('국소 연속방정식의 두 변화율','Two rates in local continuity'),[
      line('rho t',r.spatial.map(p=>[p.x,p.value.densityT])),
      line('−(rho v) x',r.spatial.map(p=>[p.x,p.value.massFluxX===null?null:-p.value.massFluxX]),'orange')],'x · m','kg/(m³ s)',diagnostic));
    panels.push(plot('mean',t('관측 평균과 해석적 평균','Observed and analytic mean density'),[
      time(x=>x.observedMeanDensity,t('질량/부피','Mass/volume')),time(x=>x.analyticMeanDensity,t('해석 평균·극한','Analytic mean/limit'),'orange')],'t · s','kg/m³',
      t('영폭의 해석적 극한을 관측0/0으로 바꾸지 않습니다. k=0의 실제 밀도는 rho0(1+alpha)입니다.','Do not turn an analytic zero-width limit into an observed0/0. At k=0 actual density is rho0(1+alpha).')));
    if(conceptStep===undefined){
      panels.push(table('window',t('기하·속도·원래 SI 관측','Geometry, velocities and original SI observations'),[
        ['start',r.start,'s'],['time',r.time,'s'],['center',q.center,'m'],['left',q.left,'m'],['right',q.right,'m'],
        ['half width',q.halfWidth,'m'],['area',q.area,'m²'],['volume',q.volume,'m³'],['fluid speed',q.fluidVelocity,'m/s'],
        ['window speed',q.boundaryVelocity,'m/s'],['relative speed',q.relativeSpeed,'m/s'],
        [t('기하 상태','Geometry status'),geometryStatus(q.geometryStatus),''],
        ['mass',q.mass,'kg'],['observed mean',q.observedMeanDensity,'kg/m³'],['analytic mean/limit',q.analyticMeanDensity,'kg/m³']]));
      panels.push(table('balance',t('방향 있는 유출과 시간 수지','Oriented transport and time balance'),[
        ['left outward',q.leftOutward,'kg/s'],['right outward',q.rightOutward,'kg/s'],['stable net inflow',q.stableNetInflow,'kg/s'],
        ['rounded net inflow',q.ordinaryNetInflow,'kg/s'],['wrong laboratory inflow',q.wrongLabFrameNetInflow,'kg/s'],
        ['stable mass change',b.massChange,'kg'],['time integral',b.integratedNetInflow,'kg'],['rounded endpoint change',b.roundedMassChange,'kg'],
        ['balance residual',b.residual,'kg'],['quadrature order difference',b.quadratureOrderDifference,'kg'],
        [t('의미','Meaning'),t('해석 가족·선택 구적의 진단','Analytic-family/selected-quadrature diagnostics'),'']]));
    }
  }
  if(result.gasConservation){
    const r=result.gasConservation,q=r.query,b=r.balance;
    const time=(get:(x:typeof q)=>number|null,name:string,color='blue')=>line(name,r.temporal.map(p=>[p.time,get(p.value)]),color);
    panels.push(geometry(q.left,q.right,r.particles,t('물질 모드는 폭도 변합니다. 고정·평행 이동 모드와 같은 초기 입자를 비교합니다. 세로 간격은 상태 구분용입니다.','Material mode changes width. Compare the same initial particles across fixed/translating/material modes. Vertical spacing separates states.')));
    panels.push(plot('density',t('밀도와 부피 변형','Density and volume deformation'),[time(x=>x.query?.density??null,'rho')],'t · s','kg/m³',
      t('밀도는 rho0/(1+a t)입니다. 밀도 감소를 곧바로 물질 질량 손실로 해석하지 않습니다.','Density is rho0/(1+a t). A lower density does not automatically mean loss of material mass.')));
    panels.push(plot('velocity',t('현재 속도의 공간 프로필','Current spatial velocity profile'),[line('u',r.spatial.map(p=>[p.x,p.value.velocity]))],'x · m','m/s',
      t('고정 위치의 시간 미분과 입자를 따르는 가속도는 다릅니다.','A time derivative at fixed position differs from acceleration along a particle.')));
    panels.push(plot('pressure',t('공간적으로 균일한 시간 의존 압력','Spatially uniform, time-dependent pressure'),[time(x=>x.query?.pressure??null,'p')],'t · s','Pa',
      t('p=p0 s^(−gamma)입니다. gamma는 교육 모형 계수이며 특정 기체 물성을 가정하지 않습니다.','p=p0 s^(−gamma). Gamma is a teaching-model coefficient, not assumed measured properties of a particular gas.')));
    panels.push(plot('energy-storage',t('내부·운동·총 저장 에너지','Stored internal, kinetic and total energy'),[
      time(x=>x.internalEnergy,t('내부','Internal')),time(x=>x.kineticEnergy,t('운동','Kinetic'),'orange'),time(x=>x.storage[2],t('총 에너지','Total'),'green')],'t · s','J',
      t('선택 모드의 실제 구간 적분입니다. 밀도 J/m³와 저장 J를 구분하세요.','Integrals over the actual interval in the selected mode. Distinguish density J/m³ from storage J.')));
    panels.push(plot('energy-rate',t('전체 에너지 변화율과 압력 일','Total energy rate and pressure work'),[
      time(x=>x.storageRates[2],t('실제 저장 변화율','Actual storage rate')),time(x=>x.pressureWorkNet,t('압력 일 순유입','Net pressure work'),'orange')],'t · s','W',
      t('물질 모드에서 두 값이 일치합니다. 고정·평행 이동 모드에는 대류 에너지 유출도 있습니다.','They match in material mode. Fixed/translating modes also have advective energy transport.')));
    panels.push(plot('mass-storage',t('선택한 구간의 질량','Mass in the selected interval'),[time(x=>x.storage[0],'M')],'t · s','kg',
      t('물질 구간에서는 일정하며 고정 창에서는 달라질 수 있습니다.','Constant for a material interval; it can vary in a fixed window.')));
    panels.push(plot('momentum-storage',t('부호 있는 전체 운동량','Signed total momentum'),[time(x=>x.storage[1],'P')],'t · s','kg m/s',
      t('부호를 지우지 않습니다. 양쪽 압력이 같으면 물질 운동량 변화는0입니다.','Signs are retained. Equal pressures give zero material-momentum change.')));
    panels.push(plot('energy-faces',t('양쪽 경계의 바깥 에너지 유출','Outward energy transport at both boundaries'),[
      time(x=>x.leftOutward[2],t('왼쪽 바깥','Left outward')),time(x=>x.rightOutward[2],t('오른쪽 바깥','Right outward'),'orange')],'t · s','W',
      t('상대 에너지 유속은 E(u−w)+p u입니다. 물질 경계에서도 p u는 남습니다.','Relative energy flux is E(u−w)+p u. Material boundaries retain p u.')));
    panels.push(plot('energy-change',t('유한 시간 에너지 변화의 세 계산','Three evaluations of finite-time energy change'),[
      line(t('서로 다른 방법 점','Separate method points'),[[1,b.changes[2]],[2,b.integratedNetInflows[2]],[3,b.roundedChanges[2]]],'blue',true)],
      '1=stable;2=time integral;3=rounded','J',diagnostic));
    panels.push(plot('missing-work',t('압력 일을 누락한 에너지 유속','Energy transport with pressure work omitted'),[
      time(x=>x.storageRates[2],t('실제 변화율','Actual rate')),
      time(x=>x.missingPressureWorkNetInflow,t('압력 일 누락 대조','Omitted-work control'),'orange')],'t · s','W',
      t('같은 실제 장에서 p u만 제거한 대조입니다. 실패에 맞춰 원천이나 장을 바꾸지 않습니다.','The same actual fields with only p u removed. Sources and fields are not changed to fit the failure.')));
    panels.push(plot('local-energy',t('국소 에너지의 두 편미분 항','Two partial-derivative terms in local energy'),[
      line('E t',r.spatial.map(p=>[p.x,p.value.timeDerivatives[2]])),
      line('−F E x',r.spatial.map(p=>[p.x,p.value.fluxDerivatives[2]===null?null:-p.value.fluxDerivatives[2]]),'orange')],'x · m','W/m³',diagnostic));
    if(conceptStep===undefined){
      const mode=q.mode==='material'?t('물질','Material'):q.mode==='fixed'?t('고정','Fixed'):t('평행 이동','Translating');
      panels.push(table('geometry',t('모드·실제 경계·입력 단위','Mode, actual boundaries and input units'),[
        [t('모드','Mode'),mode,''],['start',r.start,'s'],['time',r.time,'s'],['stretch',q.scale,'1'],['left',q.left,'m'],['right',q.right,'m'],
        ['left speed',q.leftSpeed,'m/s'],['right speed',q.rightSpeed,'m/s'],['initial half width',r.window.halfWidth,'m'],
        ['current half width',q.halfWidth,'m'],['area',q.area,'m²'],['volume',q.volume,'m³'],
        [t('기하 상태','Geometry status'),geometryStatus(q.geometryStatus),''],
        [t('비활성 입력','Inactive input'),q.mode==='translating'?t('없음','None'):t('평행 이동 속도는 이 모드에서 사용하지 않습니다.','Translating speed is inactive in this mode.'),'']]));
      panels.push(table('storage',t('세 저장량과 순간 변화율','Three stored quantities and instantaneous rates'),[
        ['density',q.query?.density??null,'kg/m³'],['velocity at center',q.query?.velocity??null,'m/s'],['pressure',q.query?.pressure??null,'Pa'],
        ['mass',q.storage[0],'kg'],['momentum',q.storage[1],'kg m/s'],['internal energy',q.internalEnergy,'J'],
        ['kinetic energy',q.kineticEnergy,'J'],['total energy',q.storage[2],'J'],
        ['mass rate',q.storageRates[0],'kg/s'],['momentum rate',q.storageRates[1],'N'],['energy rate',q.storageRates[2],'W'],
        ['pressure work net',q.pressureWorkNet,'W'],['omitted-work prediction',q.missingPressureWorkNetInflow,'W']]));
      panels.push(table('balance',t('세 예산의 유한 시간 검사','Finite-time checks of three budgets'),[0,1,2].flatMap(i=>[
        [[t('질량','Mass'),t('운동량','Momentum'),t('에너지','Energy')][i]+' · stable',b.changes[i],['kg','kg m/s','J'][i]],
        [[t('질량','Mass'),t('운동량','Momentum'),t('에너지','Energy')][i]+' · integral',b.integratedNetInflows[i],['kg','kg m/s','J'][i]],
        [[t('질량','Mass'),t('운동량','Momentum'),t('에너지','Energy')][i]+' · residual',b.residuals[i],['kg','kg m/s','J'][i]],
        [[t('질량','Mass'),t('운동량','Momentum'),t('에너지','Energy')][i]+' · order difference',b.quadratureOrderDifferences[i],['kg','kg m/s','J'][i]]
      ])));
    }
  }
  const selections=result.massTransport?[[0,1],[0,3],[4,5],[6,7]]:[[0,4],[3,5],[6,8],[9,10]];
  const shown=conceptStep===undefined?panels:(selections[conceptStep]??[]).map(i=>panels[i]);
  return <div data-continuum-kind={result.massTransport?'mass':'gas'}>{shown}
    <p role="status">{t('—는 사용 불가이며0이 아닙니다. 정확한 영·표현 손실·관측 정보와 해석 가족의 진단을 구분하세요.','— means unavailable, not zero. Separate exact zeros, representation loss, observation information and analytic-family diagnostics.')}</p></div>;
}
