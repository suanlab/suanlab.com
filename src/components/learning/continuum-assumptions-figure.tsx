'use client';
import {useLanguage} from '@/components/language-provider';
import type {ContinuumResult} from '@/lib/learning-continuum';
import {WavePlot,waveNumber as fmt,type Series} from './waves-plot';
import type {Vec2} from '@/lib/learning-simulations';
export default function ContinuumAssumptionsFigure({result,conceptStep}:{result:ContinuumResult;conceptStep?:number}) {
  const {language}=useLanguage(),t=(ko:string,en:string)=>language==='ko'?ko:en;
  const panels:React.ReactNode[]=[],prefix=result.particleAveraging?'continuum-particles':'continuum-scales';
  const line=(name:string,points:[number,number|null][],color='blue'):Series=>{
    const segments:Vec2[][]=[];let part:Vec2[]=[];
    for(const[x,y]of points){if(y===null||!Number.isFinite(y)){if(part.length)segments.push(part);part=[];}else part.push([x,y]);}if(part.length)segments.push(part);
    return{name,color,segments};
  };
  const plot=(id:string,title:string,series:Series[],xLabel:string,yLabel:string,caption:string,vertical:number[]=[])=>{
    const max=Math.max(0,...series.flatMap(s=>s.segments.flat().map(p=>Math.abs(p[1])))),scale=max>1e5||max>0&&max<.001?max:1;
    return <WavePlot key={id} marker={prefix+'-'+id} title={title} series={series.map(s=>({...s,segments:s.segments.map(part=>part.map(([x,y])=>[x,y/scale]as Vec2))}))}
      xLabel={xLabel} yLabel={scale===1?yLabel:'y × '+fmt(scale)+' · '+yLabel} minimumYPadding={max===0?.01:1e-100} vertical={vertical}
      caption={caption+(scale===1?'':' '+t('실제 세로 값 = 좌표 ×','Physical ordinate = coordinate ×')+' '+fmt(scale)+'.')}/>;
  };
  const table=(id:string,title:string,rows:(string|number|null)[][])=><figure key={id} data-continuum-table={prefix+'-'+id}>
    <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption>{title}</caption>
      <thead><tr>{[t('항목','Quantity'),t('값','Value'),t('단위·의미','Unit or meaning')].map(h=><th key={h} scope="col">{h}</th>)}</tr></thead>
      <tbody>{rows.map((row,i)=><tr key={i}>{row.map((value,j)=>j===0?<th scope="row" key={j}>{value}</th>:<td key={j}>{typeof value==='number'||value===null?fmt(value):value}</td>)}</tr>)}</tbody>
    </table></div></figure>;
  const negative=(value:number|null)=>value===null?null:-value;
  const missing=t('미정 값은 곡선을 끊습니다. 0으로 대체하지 않습니다.','Unavailable values break the curve; they are not replaced by zero.');
  if(result.particleAveraging){
    const r=result.particleAveraging,q=r.query,X=(x:number)=>62+262*(x+6)/12;
    const geometryTitle=t('같은 입자·커널 지지·관측 창','Particles, kernel supports and observation window');
    panels.push(<figure key="geometry" className="ai-foundation-figure"><h3>{geometryTitle}</h3><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={geometryTitle}>
      <svg viewBox="0 0 350 345" role="img" aria-label={geometryTitle} data-waves-plot={prefix+'-geometry'}>
        <path d="M62 165H324" stroke="var(--learning-neutral)"/>
        {q.contributions.map((p,i)=><g key={i}><path d={'M'+X(r.particles[i].position)+' 95L'+X(p.position)+' 155'} stroke="var(--learning-neutral)" strokeDasharray="3 4"/>
          <circle cx={X(r.particles[i].position)} cy="95" r="3" fill="var(--learning-neutral)"/><circle cx={X(p.position)} cy="155" r="5" fill={'var(--learning-'+(i?'orange':'blue')+')'}/>
          <path d={'M'+X(p.position-r.h)+' '+(190+i*20)+'v10H'+X(p.position+r.h)+'v-10'} stroke={'var(--learning-'+(i?'orange':'blue')+')'} fill="none"/></g>)}
        <path d={'M'+X(r.left)+' 235v10H'+X(r.right)+'v-10'} stroke="var(--learning-orange)" fill="none"/>
        <path d={'M'+X(r.x)+' 75V260'} stroke="var(--learning-green)" strokeDasharray="3 4"/>
        <g fill="currentColor" fontSize="10"><text x="190" y="45" textAnchor="middle">{t('회색: 초기 · 색 점: 현재 · 녹색: 관측 위치','Gray: initial · Color: current · Green: query')}</text>
          <text x="190" y="280" textAnchor="middle">{t('위: 커널 지지 · 아래: 고정 창','Upper brackets: supports · Lower: fixed window')}</text>
          <text x="62" y="300">−6</text><text x="324" y="300" textAnchor="end">6</text><text x="190" y="320" textAnchor="middle">x · m</text>
          <text x="190" y="340" textAnchor="middle">{t('세로 간격·점 반경은 물리량을 나타내지 않음','Vertical spacing and radii are schematic')}</text></g>
      </svg></div><figcaption>{t('입자는 충돌 없이 서로 통과합니다. 같은 초기 입자와 현재 지지 구간을 비교하며 폭을 임의 확대하지 않습니다.','Particles pass without collisions. Compare the same initial particles and current support; widths are not artificially enlarged.')}</figcaption></figure>);
    const spatial=(get:(p:typeof q)=>number|null,name:string,color='blue')=>line(name,r.spatial.map(p=>[p.x,get(p.value)]),color);
    panels.push(plot('density',t('합한 선밀도와 입자별 기여','Line density and individual contributions'),[
      spatial(p=>p.density,'rho'),...r.particles.map((_,i)=>spatial(p=>p.contributions[i].weight,'particle '+(i+1),i?'green':'orange'))],'x · m','kg/m',
      t('일차원 선밀도입니다. 각 커널은 전체 직선에서 질량을 보존하며 화면 안에서 다시 정규화하지 않습니다.','This is line density. Each kernel conserves mass on the full line; the visible region is not renormalized.'),[r.x,r.left,r.right]));
    panels.push(plot('flux',t('질량 유속과 평균 속도의 곱','Mass flux and density times mean velocity'),[
      spatial(p=>p.flux,'j'),spatial(p=>p.velocity===null?null:p.flux,'rho u','orange')],'x · m','kg/s',
      t('같은 가중 합의 정의를 비교합니다. 독립 검증이나 충돌 유체의 인증이 아닙니다.','These are definitions from the same weighted sums, not an independent check or a certificate for a collisional fluid.')));
    panels.push(plot('velocity',t('질량으로 가중한 관측 속도','Mass-weighted observed velocity'),[spatial(p=>p.velocity,'u')],'x · m','m/s',missing+' '+t('빈 지지 구간과 정지한 물질을 구분합니다.','Separate empty support from material observed at rest.')));
    panels.push(plot('momentum',t('운동량 유속·평균 수송·분산','Momentum flux, mean transport and dispersion'),[
      spatial(p=>p.momentumFlux,'Q'),spatial(p=>p.meanMomentumTransport,'rho u²','orange'),spatial(p=>p.dispersion,'S','green')],'x · m','N',
      t('Q=rho u²+S는 밀도가 양수인 곳의 모멘트 분해입니다. S를 열역학적 압력이나 점성으로 자동 해석하지 않습니다.','Q=rho u²+S is a moment decomposition where density is positive. S is not automatically thermodynamic pressure or viscosity.')));
    panels.push(plot('continuity',t('국소 질량 변화와 물리적 유입','Local mass change and physical inflow'),[
      spatial(p=>p.densityT,'rho t'),spatial(p=>negative(p.fluxX),'−j x','orange'),spatial(p=>p.smoothingRate,t('관측 폭의 항','Width term'),'green')],'x · m','kg/(m s)',
      t('폭 변화율이 비영이면 두 변화율의 차이는 관측 폭의 항입니다. 물리적 질량 원천이 아닙니다.','With nonzero width rate, the difference between rates is the width term, not a physical mass source.')));
    panels.push(plot('momentum-balance',t('탄도 운동량 변화와 관측 항','Ballistic momentum change and observation term'),[
      spatial(p=>p.momentumT,'j t'),spatial(p=>negative(p.momentumFluxX),'−Q x','orange'),spatial(p=>p.momentumSmoothingRate,t('관측 폭의 항','Width term'),'green')],'x · m','kg/s²',
      t('고정 폭에서 탄도 운동량 수지가 맞습니다. 시간에 따라 변하는 폭은 운동량 관측에도 항을 추가합니다.','Fixed width obeys ballistic momentum balance. Time-dependent width adds a term to momentum observations too.')));
    panels.push(plot('width-density',t('같은 상태에서 폭별 관측 밀도','Density observed at different widths'),[line('rho',r.widths.map(p=>[p.h,p.value.density]))],'h · m','kg/m',
      t('같은 입자·시각·관측 위치를 유지합니다. 가로축은 시간이나 물리 팽창이 아니라 관측 반폭입니다.','The particles, time and query position stay fixed. The horizontal axis is observation half-width, not time or physical expansion.')));
    panels.push(plot('width-dispersion',t('폭이 바꾸는 관측 속도 분산','Velocity dispersion at different widths'),[line('S',r.widths.map(p=>[p.h,p.value.dispersion]))],'h · m','N',missing));
    panels.push(plot('window-mass',t('고정 폭 시간 스윕의 창 질량','Window mass in a fixed-width time sweep'),[
      line(t('고정 창 질량','Fixed-window mass'),r.temporal.map(p=>[p.time,p.mass])),line(t('전체 질량','Global mass'),r.temporal.map(p=>[p.time,r.totalMass]),'orange')],'t · s','kg',
      t('이 스윕에서는 h를 현재 값에 고정합니다. 폭 변화율 입력은 선택 시각의 국소 미분이며 이 곡선의 경로가 아닙니다.','This sweep fixes h at its current value. The width-rate input describes a local derivative at the selected state, not the path of this curve.')));
    if(conceptStep===undefined){
      panels.push(table('observation',t('선밀도·속도와 실제 관측값','Line density, velocity and actual observations'),[
        ['time',r.time,'s'],['x',r.x,'m'],['h',r.h,'m'],['width rate',r.bandwidthRate,'m/s'],['rho',q.density,'kg/m'],['j',q.flux,'kg/s'],['u',q.velocity,'m/s'],['Q',q.momentumFlux,'N'],['S',q.dispersion,'N'],['rho t',q.densityT,'kg/(m s)'],['−j x',negative(q.fluxX),'kg/(m s)'],['width term',q.smoothingRate,'kg/(m s)'],[t('지원 상태','Support status'),q.status==='empty-support'?t('빈 지지: 관측 속도 미정','Empty support: velocity undefined'):t('양의 관측 밀도','Positive observed density'),'']]));
      panels.push(<figure key="arithmetic" data-assumption-table="arithmetic"><div className="ai-discrete-table" role="region" tabIndex={0} aria-label={t('정확한 영과 비영 산술 관측','Exact zeros and nonzero arithmetic observations')}><table><caption>{t('정확한 영과 비영 산술 관측','Exact zeros and nonzero arithmetic observations')}</caption><thead><tr>{[t('관측','Observation'),t('값·상태','Value or state'),t('부호','Sign'),'ln |q / unit|'].map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{Object.entries(q.observations).map(([name,value])=><tr key={name} data-assumption-observation={name}><th scope="row">{name}</th><td>{value===null?t('빈 지지: 미정','Empty support: undefined'):value.sign===0?t('정확한0','Exact zero'):value.value===null?t('비영·일반 값 불가','Nonzero; linear value unavailable'):fmt(value.value)}</td><td>{value?.sign??'—'}</td><td>{fmt(value?.logMagnitude)}</td></tr>)}</tbody></table></div><figcaption>{t('이진 입력의 곱·합과 비를 투영하기 전에 계산합니다. 표시 범위 밖의 이차 모멘트·분산·관측 항을0으로 바꾸지 않습니다. 위치와 지지 경계의 기하는 일반 부동소수점입니다.','Products, sums and ratios of binary inputs are evaluated before projection. Out-of-range second moments, dispersion and observation terms are not replaced by zero. Position and support geometry use ordinary floating point.')}</figcaption></figure>);
      panels.push(table('particles',t('실제 입자별 입력과 기여','Actual particle inputs and contributions'),q.contributions.flatMap((p,i)=>[
        ['particle '+(i+1)+' mass',p.mass,'kg'],['particle '+(i+1)+' initial',r.particles[i].position,'m'],['particle '+(i+1)+' current',p.position,'m'],['particle '+(i+1)+' velocity',p.velocity,'m/s'],['particle '+(i+1)+' density',p.weight,'kg/m']])));
      panels.push(table('window',t('유한 창과 전체 질량','Finite window and global mass'),[
        ['left',r.left,'m'],['right',r.right,'m'],['requested half-width',r.halfWidth,'m'],['rounded width',r.interval.width,'m'],['window mass',r.interval.mass,'kg'],['window mean',r.interval.meanDensity,'kg/m'],['global mass',r.totalMass,'kg'],
        [t('관측 상태','Observation status'),r.widthStatus==='zero-length'?t('영길이: 평균 미정','Zero length: mean undefined'):r.widthStatus==='unsupported-positive-width'?t('양의 폭이 지원 범위 아래: 관측 불가','Positive width below supported range: unavailable'):t('지원되는 유한 창','Supported finite window'),'']]));
    }
  }
  if(result.scaleSeparation){
    const r=result.scaleSeparation,q=r.query,log=Math.log10;
    const limits=[q.countLowerLength,q.collisionLowerLength,q.upperLength,q.averagingLength,q.meanSpacing,q.meanFreePath,q.macroLength];
    const min=Math.min(-12,...limits.map(log)),max=Math.max(6,...limits.map(log)),X=(value:number)=>62+262*(log(value)-min)/(max-min);
    const geometryTitle=t('로그 길이 축의 선언한 후보 구간','Declared feasible window on a log-length axis');
    panels.push(<figure key="geometry" className="ai-foundation-figure"><h3>{geometryTitle}</h3><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={geometryTitle}>
      <svg viewBox="0 0 350 345" role="img" aria-label={geometryTitle} data-waves-plot={prefix+'-geometry'}>
        <path d="M62 250H324" stroke="var(--learning-neutral)"/>
        {[['count lower',q.countLowerLength,'blue'],['collision lower',q.collisionLowerLength,'orange'],['resolution upper',q.upperLength,'green'],['selected length',q.averagingLength,'blue']].map(([name,value,color],i)=><g key={String(name)}>
          <text x="62" y={45+i*45} fill="currentColor" fontSize="10">{name}</text><circle cx={X(Number(value))} cy={60+i*45} r="4" fill={'var(--learning-'+color+')'}/><path d={'M62 '+(60+i*45)+'H324'} stroke="var(--learning-neutral)" strokeDasharray="2 4"/>
        </g>)}
        {q.windowExists?<path d={'M'+X(q.lowerLength)+' 220H'+X(q.upperLength)} stroke="var(--learning-green)" strokeWidth="6"/>:<text x="190" y="224" textAnchor="middle" fill="currentColor" fontSize="10">{t('선언한 후보 구간 없음','No declared feasible interval')}</text>}
        <g fill="currentColor" fontSize="10"><text x="62" y="273">{fmt(min)}</text><text x="324" y="273" textAnchor="end">{fmt(max)}</text><text x="190" y="295" textAnchor="middle">log10(length / m)</text>
          <text x="190" y="320" textAnchor="middle">{t('녹색 선분: 선언한 조건의 교집합','Green segment: declared proxy intersection')}</text><text x="190" y="339" textAnchor="middle">{t('실제 물리적 유효성의 인증이 아님','Not a certificate of physical validity')}</text></g>
      </svg></div><figcaption>{t('좌표는 실제 길이의 상용로그입니다. 입자 수·충돌·해상도 대리 조건을 나누고 후보 구간과 현재 선택을 구분합니다.','Coordinates are base-ten logarithms of physical lengths. Separate count, collision and resolution proxies, interval existence and the selected width.')}</figcaption></figure>);
    const sweep=(get:(value:typeof q)=>number,name:string,color='blue')=>line(name,r.lengths.map(p=>[p.logLength,get(p.value)]),color);
    const selected=[log(q.averagingLength)],xLabel='log10(ell / m)';
    panels.push(plot('counts',t('평균 창 길이별 기대 입자 수','Expected count across averaging lengths'),[sweep(p=>log(p.expectedCount),'log10(mu)')],xLabel,'log10(count)',t('삼차원 정육면체의 기대 개수입니다. 한 번 관측한 정수 개수나 일차원 선분의 개수와 다릅니다.','Expected counts in a three-dimensional cube; these are not observed integer counts or counts in a line segment.'),selected));
    panels.push(plot('noise',t('Poisson 계수 요동과 선택한 허용값','Poisson noise and the declared tolerance'),[
      sweep(p=>log(p.relativeCountStd),'log10(CV)'),sweep(()=>log(q.relativeFluctuationTolerance),'log10(epsilon)','orange')],xLabel,'log10(relative noise)',t('균일 Poisson 기준이며 상관된 실제 물질의 분산을 인증하지 않습니다.','This homogeneous Poisson benchmark does not certify fluctuations in a correlated material.'),selected));
    panels.push(plot('collision',t('평균 자유 행로 대비 관측 길이','Observation length relative to mean free path'),[
      sweep(p=>log(p.ratioToMeanFreePath),'log10(ell/lambda)'),sweep(()=>log(q.collisionSeparationFactor),'log10(c)','orange')],xLabel,'log10(length ratio)',t('사용자가 선언한 충돌 분리 대리 조건입니다. 통과해도 국소 평형·시간 척도는 별도 확인합니다.','This is a user-declared collision proxy. Local equilibrium and temporal scales require separate evidence.'),selected));
    panels.push(plot('resolution',t('거시 길이 대비 관측 길이','Observation length relative to macro length'),[
      sweep(p=>log(p.ratioToMacro),'log10(ell/L)'),sweep(()=>log(q.relativeResolutionTolerance),'log10(eta)','orange')],xLabel,'log10(length ratio)',t('상한은 선택한 해상도 비율입니다. 임의 장의 엄밀한 평균 편향 상한이 아닙니다.','The upper limit is a chosen resolution ratio, not a rigorous averaging-bias bound for arbitrary fields.'),selected));
    const fractions=(get:(value:typeof r.count)=>number|null,name:string,color='blue')=>line(name,r.fractions.map(p=>[p.p,get(p.value)]),color);
    panels.push(plot('finite-noise',t('고정 저장소와 Poisson 계수 요동','Fixed-reservoir and Poisson count noise'),[
      fractions(p=>p.relativeStd,t('이항 계수','Binomial count')),fractions(p=>p.poissonRelativeStd,t('Poisson 기준','Poisson benchmark'),'orange')],'p','relative std / mean',missing+' '+t('p=1의 고정 총개수는 정확한 영요동입니다. p=0은 미정입니다.','At p=1 the fixed total has exact zero noise; p=0 is undefined.'),[r.p]));
    panels.push(plot('finite-moments',t('고정 저장소의 평균과 분산','Fixed-reservoir mean and variance'),[
      fractions(p=>p.mean,'mean'),fractions(p=>p.variance,'variance','orange')],'p','dimensionless count moments',t('N개의 독립 균일 위치와 관측 부피 비율p를 가정합니다. 수밀도와 길이 입력의 별도 비교 예제입니다.','Assumes N independently uniform positions and observed volume fraction p. This is a separate comparison from the density/length inputs.'),[r.p]));
    panels.push(plot('fano',t('고정 총개수와 Poisson의 분산 비','Variance-to-mean ratio for two count models'),[
      fractions(p=>p.mean===0?null:p.variance/p.mean,t('이항 Fano 비','Binomial Fano ratio')),fractions(p=>p.mean===0?null:1,t('Poisson Fano 비','Poisson Fano ratio'),'orange')],'p','variance / mean',t('평균이 양수일 때만 정의합니다. 실제 상관 함수나 Fano 계수를 추정한 결과가 아닙니다.','Defined only for positive mean. This does not estimate a physical correlation function or measured Fano factor.'),[r.p]));
    if(conceptStep===undefined){const yes=(value:boolean)=>value?t('통과','Pass'):t('실패','Fail');
      panels.push(table('limits',t('실제 SI 길이와 선언한 한계','Physical SI lengths and declared limits'),[
        ['number density',q.numberDensity,'/m³'],['mean spacing',q.meanSpacing,'m'],['mean free path',q.meanFreePath,'m'],['macro length',q.macroLength,'m'],['count lower',q.countLowerLength,'m'],['collision lower',q.collisionLowerLength,'m'],['combined lower',q.lowerLength,'m'],['resolution upper',q.upperLength,'m'],['selected length',q.averagingLength,'m'],['Knudsen number',q.knudsen,'dimensionless']]));
      panels.push(table('selection',t('후보 구간과 현재 관측의 판정','Interval existence and selected observation'),[
        [t('후보 구간 존재','Declared interval exists'),yes(q.windowExists),''],[t('계수 조건','Count condition'),yes(q.countSatisfied),''],[t('충돌 조건','Collision condition'),yes(q.collisionSatisfied),''],[t('해상도 조건','Resolution condition'),yes(q.resolutionSatisfied),''],[t('현재 폭의 통과','Selected width passes'),yes(q.selectedSatisfies),''],['expected count',q.expectedCount,''],['Poisson relative noise',q.relativeCountStd,''],[t('해석 범위','Interpretation'),t('선언한 대리 조건만 검사·물리 인증 아님','Declared proxies only; not physical certification'),'']]));
      panels.push(table('reservoir',t('별도 고정 저장소의 계수 관측','Counts in a separate fixed reservoir'),[
        ['N',r.totalCount,'fixed total'],['p',r.p,'volume fraction'],['mean count',r.count.mean,''],['binomial variance',r.count.variance,''],['binomial relative noise',r.count.relativeStd,''],['Poisson relative noise',r.count.poissonRelativeStd,'']]));
    }
  }
  const selection=result.particleAveraging?[[0,1],[3,4],[1,3],[5,7]]:[[0,2],[0,3],[1,2],[5,7]];
  return <div className="ai-foundation-plots" data-continuum-kind={result.particleAveraging?'particles':'scales'}>{conceptStep===undefined?panels:selection[conceptStep].map(i=>panels[i])}</div>;
}
