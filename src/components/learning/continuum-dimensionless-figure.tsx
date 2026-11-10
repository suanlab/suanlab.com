'use client';
import { useLanguage } from '@/components/language-provider';
import type { ContinuumResult } from '@/lib/learning-continuum';
import type { TransportObservation } from '@/lib/learning-continuum-dimensionless';
import type { Vec2 } from '@/lib/learning-simulations';
import { WavePlot, waveNumber as fmt, type Series } from './waves-plot';

type DimensionlessSeries = Series & { extent: [number, number] };
type Point = [number | null, number | null];
export default function ContinuumDimensionlessFigure({ result, conceptStep }: { result: ContinuumResult; conceptStep?: number }) {
  const { language } = useLanguage();
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const panels: React.ReactNode[] = [];
  const prefix = result.peclet ? 'continuum-peclet' : 'continuum-damkohler';
  const line = (name: string, points: Point[], color = 'blue', discrete = false): DimensionlessSeries => {
    const xs = points.map(([x]) => x).filter((x): x is number => x !== null && Number.isFinite(x));
    const segments: Vec2[][] = []; let part: Vec2[] = [];
    for (const [x,y] of points) {
      if (x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y)) { if (part.length) segments.push(part); part = []; }
      else if (discrete) segments.push([[x,y]]);
      else part.push([x,y]);
    }
    if (part.length) segments.push(part);
    return { name, color, segments, extent: [Math.min(0, ...xs), Math.max(0, ...xs)] };
  };
  const plot = (id: string, title: string, series: DimensionlessSeries[], x: string, unit: string, caption: string, vertical: number[] = []) => {
    const left = Math.min(0, ...series.map(s => s.extent[0])), right = Math.max(0, ...series.map(s => s.extent[1]));
    const magnitude = Math.max(0,...series.flatMap(s=>s.segments.flat().map(p=>Math.abs(p[1]))));
    const scale = magnitude > 1e5 || magnitude > 0 && magnitude < .001 ? magnitude : 1;
    return <WavePlot key={id} marker={`${prefix}-${id}`} title={title}
      series={series.map(s=>({...s,segments:s.segments.map(part=>part.map(([x,y])=>[x,y/scale] as Vec2))}))}
      xDomain={[left, right > left ? right : left + .01]} xLabel={x} yLabel={scale===1?unit:`y × ${fmt(scale)} · ${unit}`} minimumYPadding={magnitude===0?.01:1e-100}
      vertical={vertical} caption={caption+(scale===1?'':` ${t('실제 값 = 세로 좌표 ×','Physical value = ordinate ×')} ${fmt(scale)}.`)} />;
  };
  const unknown = t('미정: 기준 분모가0','Undefined: zero reference denominator');
  const absent = t('해당 기작 없음','Mechanism absent');
  const state = (q: TransportObservation | null) => q === null ? absent : q.sign===0 ? t('정확한0','Exact zero') : q.value===null ? t('비영·일반 값 불가','Nonzero; linear value unavailable') : fmt(q.value);
  type Row = [string, TransportObservation | null, string];
  const table = (id: string, title: string, rows: Row[]) => <figure key={id} data-continuum-table={`${prefix}-${id}`}>
    <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption>{title}</caption>
      <thead><tr>{[t('관측','Observation'),t('일반 값·상태','Linear value or state'),t('부호','Sign'),'ln |q / unit|',t('단위','Unit')].map(h=><th key={h} scope="col">{h}</th>)}</tr></thead>
      <tbody>{rows.map(([name,q,unit],i)=><tr key={i}><th scope="row">{name}</th><td>{state(q)}</td><td>{q===null?'—':q.sign}</td><td>{fmt(q?.logMagnitude)}</td><td>{unit}</td></tr>)}</tbody>
    </table></div></figure>;
  const note = (id: string, text: string) => <p key={id} className="ai-logic-statement" data-dimensionless-note={id}>{text}</p>;
  const similarityNote = t('b=1로 고정하고 L′=aL,U′=U/a,A′=a²A를 비교합니다. 입력 범위 밖의 비교를 자르지 않습니다.','Fix b=1 and compare L′=aL, U′=U/a, A′=a²A. Comparisons outside the supported range are not clamped.');
  const diagnostic = t('같은 해석적 가족의 대조이며 독립 정확도 인증이 아닙니다.','A comparison within the same analytic family, not independent accuracy certification.');
  if (result.peclet) {
    const r=result.peclet,q=r.query;
    const spatial=(name:string,get:(q:typeof r.query)=>number|null,color='blue')=>line(name,r.spatial.map(s=>[s.xi,get(s.value)]),color);
    panels.push(plot('clocks',t('확산과 이류의 시간 척도','Diffusion and advection clocks'),[
      line('tD',[[1,q.diffusionTime.logMagnitude]],'blue',true),line('tA',[[2,q.advectionTime?.logMagnitude??null]],'orange',true)],
      '1=tD; 2=tA','ln(t / s)',t('두 시간의 비가 |Pe|입니다. U0에는 유한한 이류 시간이 없습니다. 실제 값은 표에서 확인하세요.','Their ratio is |Pe|. Zero velocity has no finite advection clock. Read actual values in the table.')));
    panels.push(plot('profile',t('농도와 하류 경계층','Concentration and the downstream layer'),[
      spatial('c',q=>q.concentration.value),line(t('선형 대조','Linear control'),r.spatial.map(s=>[s.xi,r.parameters.leftConcentration+(r.parameters.rightConcentration-r.parameters.leftConcentration)*s.xi]),'orange')],
      'ξ = x/L','kg/m³',t('속도의 부호가 바뀌면 경계층의 위치가 바뀝니다. 두 정확한 경계값을 유지하며 층 주변 표본을 보완합니다.','Velocity direction changes the layer location. Both exact boundary values and extra samples near the layer are retained.'),[r.xi]));
    panels.push(plot('flux',t('이류·확산·전체 플럭스','Advective, diffusive and total flux'),[
      spatial('Uc',q=>q.advectiveFlux.value),spatial('−D cx',q=>q.diffusiveFlux.value,'orange'),spatial('j stable',q=>q.totalFlux.value,'green'),spatial('j rounded',q=>q.roundedTotalFlux,'neutral')],
      'ξ = x/L','kg/(m² s)',t('전체 플럭스는 일정하지만 개별 기여는 달라집니다. 작은 전체 값은 로그 표에서 확인하세요. ','Total flux is constant while individual contributions vary. Read tiny total values in the log table. ')+diagnostic));
    panels.push(plot('similarity',t('같은 Pe와 겹치는 정규화 기저','Matching Pe and overlapping basis profiles'),[
      spatial(t('원래 기저','Original basis'),q=>q.basisShape.value),line(t('닮음 기저','Similar basis'),r.spatial.map(s=>[s.xi,s.comparison?.basisShape.value??null]),'orange')],
      'ξ = x/L','1',similarityNote));
    panels.push(plot('physical',t('실제 길이에서의 농도','Concentration over physical length'),[
      line(t('원래 실험','Original experiment'),r.spatial.map(s=>[s.value.position.value,s.value.concentration.value])),
      line(t('닮은 실험','Similar experiment'),r.spatial.map(s=>[s.comparison?.position.value??null,s.comparison?.concentration.value??null]),'orange')],
      'x · m','kg/m³',t('정규화 곡선이 같아도 실제 길이·시간·수송량은 다릅니다.','A matching normalized curve does not imply equal length, time or transport.')));
    panels.push(plot('weights',t('두 경계의 양의 가중치','Positive weights of both boundaries'),[spatial('wL',q=>q.leftWeight.value),spatial('wR',q=>q.basisShape.value,'orange')],'ξ = x/L','1',t('선언한 기저입니다. 농도 대비0의 측정 정규화와 구분합니다.','These are declared basis functions, distinct from measured normalization at zero concentration contrast.')));
    panels.push(plot('gradient',t('실제 농도 기울기','Physical concentration gradient'),[spatial('dc/dx',q=>q.gradient.value)],'ξ = x/L','kg/m⁴',t('ξ에 대한 미분과 달리 실제 길이 L의 계수가 포함됩니다.','Unlike differentiation with respect to ξ, this includes the physical length factor.')));
    panels.push(plot('tail',t('농도와 작은 전체 플럭스의 로그','Logs of concentration and tiny total flux'),[
      spatial('ln c',q=>q.concentration.logMagnitude),spatial('ln |j|',q=>q.totalFlux.logMagnitude,'orange')],'ξ = x/L','ln |q / unit|',t('c는 kg/m³,j는 kg/(m²s)를 기준으로 나눈 로그입니다. 실제0에는 로그가 없습니다. 부호는 표에서 확인합니다.','Logs use c/(kg/m³) and |j|/(kg/(m² s)). Exact zero has no logarithm. Read signs in the table.')));
    panels.push(plot('small',t('작은 속도에서 직선 대비 보정','Correction from a line at small velocity'),[
      line('ln |θ−ξ|',r.corrections.map(s=>[s.logVelocity,s.value.stableShapeCorrection.logMagnitude]))],
      'log10(|U| / (m/s))','ln |θ−ξ|',t('조회 ξ에서의 안정 보정입니다. 곡선의 속도 범위는 현재 입력과 별도입니다. 양 끝의 보정은 정확히0입니다.','Stable correction at the selected ξ. The displayed velocity sweep is separate from the current input. Endpoint corrections are exactly zero.')));
    if (conceptStep===undefined) {
      panels.push(table('query',t('현재 이류·확산 관측','Current advection–diffusion observations'),[
        ['Pe',q.peclet,'1'],['x',q.position,'m'],['c',q.concentration,'kg/m³'],['dc/dx',q.gradient,'kg/m⁴'],['d²c/dx²',q.curvature,'kg/m⁵'],
        ['Uc',q.advectiveFlux,'kg/(m² s)'],['−D cx',q.diffusiveFlux,'kg/(m² s)'],['j stable',q.totalFlux,'kg/(m² s)'],['A j',q.totalTransport,'kg/s'],
        ['tD',q.diffusionTime,'s'],['tA',q.advectionTime,'s'],['D/|U|',q.boundaryLayerLength,'m'],['θ−ξ stable',q.stableShapeCorrection,'1'],['δc stable',q.stableConcentrationCorrection,'kg/m³']]));
      panels.push(note('pe-normalization',`${t('측정 정규화','Measured normalization')}: ${q.observedNormalizedShape===null?unknown:state(q.observedNormalizedShape)}. ${t('직접 반올림한 Uc−Dcx','Direct rounded Uc−Dcx')}: ${fmt(q.roundedTotalFlux)} kg/(m² s).`));
      if (r.comparison) panels.push(table('similarity',t('닮은 실험의 실제 관측','Physical observations in the similar experiment'),[
        ['Pe′',r.comparison.peclet,'1'],['x′',r.comparison.position,'m'],['tD′',r.comparison.diffusionTime,'s'],['j′',r.comparison.totalFlux,'kg/(m² s)'],['A′ j′',r.comparison.totalTransport,'kg/s']]));
      else panels.push(note('pe-range',r.comparisonStatus==='nonzero-coefficient-outside-arithmetic-range'?t('닮음 변환의 비영 속도를 충분한 상대 정확도로 표현할 수 없어 비교를 제공하지 않습니다. 원래 해는 유지합니다.','The transformed nonzero velocity cannot be represented with sufficient relative accuracy, so comparison is unavailable. The original solution remains valid.'):t('닮음 비교는 선언한 계수 범위 밖이라 제공하지 않습니다. 원래 실험의 해는 유지합니다.','The similarity comparison is outside the declared coefficient range and is unavailable. The original experiment remains valid.')));
    }
  }
  if (result.damkohler) {
    const r=result.damkohler,q=r.query;
    const spatial=(name:string,get:(q:typeof r.query)=>number|null,color='blue')=>line(name,r.spatial.map(s=>[s.xi,get(s.value)]),color);
    panels.push(plot('clocks',t('확산·반응 시간과 Da','Diffusion/reaction clocks and Da'),[
      line('tD',[[1,q.diffusionTime.logMagnitude]],'blue',true),line('tR',[[2,q.reactionTime?.logMagnitude??null]],'orange',true)],
      '1=tD; 2=tR','ln(t / s)',t('Da=tD/tR=φ²입니다. 반응0에는 유한한 반응 시계가 없습니다.','Da=tD/tR=φ². Zero reaction has no finite reaction clock.')));
    panels.push(plot('profile',t('반두께 판의 농도와 밀폐면','Half-slab concentration and sealed plane'),[
      spatial('c',q=>q.concentration.value),line('c0',r.spatial.map(s=>[s.xi,r.parameters.surfaceConcentration]),'orange')],
      'ξ = x/L','kg/m³',t('L은 반두께입니다. ξ=1의 무유속은 농도0이 아니라 기울기0입니다.','L is the half-thickness. Zero flux at ξ=1 means zero gradient, not zero concentration.'),[r.xi]));
    panels.push(plot('budget',t('표면에서 조회 위치까지의 소모','Consumption from surface to observation position'),[
      spatial(t('실제 소모 적분','Physical consumption integral'),q=>q.partialConsumption.value),
      spatial(t('반올림 유입−출구','Rounded inlet minus outlet'),q=>q.inletFlux.value===null||q.diffusiveFlux.value===null?null:r.parameters.area*(q.inletFlux.value-q.diffusiveFlux.value),'orange')],
      'ξ = x/L','kg/s',t('부분 체적의 소모와 A[j(0)−j(x)]를 비교합니다. 작은 창의 큰 항 뺄셈은 보정을 잃을 수 있습니다. ','Compare volume consumption with A[j(0)−j(x)]. Subtraction can lose a tiny-window correction. ')+diagnostic));
    panels.push(plot('effectiveness',t('반응 속도와 효과도','Reaction rate and effectiveness'),[
      line('η basis',r.response.map(s=>[s.logRate,s.value.analyticEffectiveness.value]))],
      'log10(k / (1/s))','1',t('선언한 기저 효과도입니다. k0 또는 c00의 실제 반응률 비는 미정입니다. 곡선은 별도 k범위를 사용합니다.','This is the declared basis effectiveness. The observed rate ratio is undefined at k=0 or c0=0. The curve uses a separate rate sweep.')));
    panels.push(<figure key="geometry" className="ai-foundation-figure" data-waves-panel={`${prefix}-geometry`}><h3>{t('표면과 대칭·밀폐면','Surface and symmetry/sealed plane')}</h3>
      <div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={t('반두께 판 경계조건 그림','Half-slab boundary-condition diagram')}><svg viewBox="0 0 350 240" role="img" aria-label={t('반두께 판 경계조건 그림','Half-slab boundary-condition diagram')} data-waves-plot={`${prefix}-geometry`}>
        <rect x="65" y="65" width="225" height="100" fill="var(--learning-blue)" fillOpacity=".12" stroke="var(--learning-blue)" strokeWidth="2"/>
        <path d="M290 50V180M30 115H63M49 105L63 115L49 125" fill="none" stroke="var(--learning-orange)" strokeWidth="3"/>
        <path d={`M${65+225*r.xi} 65V165`} stroke="var(--learning-green)" strokeDasharray="4 3"/>
        <g fill="currentColor" fontSize="12"><text x="65" y="35">c(0)=c0 · A={fmt(r.parameters.area)} m²</text><text x="65" y="195">L={fmt(r.parameters.length)} m · half slab</text><text x="290" y="215" textAnchor="end">dc/dx(L)=0</text></g>
      </svg></div><figcaption>{t('경계조건 개념도이며 길이·면적의 축척 그림이 아닙니다. 양면에 같은 조건을 준 전체 판의 저장·소모는 반쪽의2배입니다.','Boundary-condition schematic, not a length/area scale drawing. A symmetric full slab with the same conditions on both faces has twice the half-slab storage and consumption.')}</figcaption></figure>);
    panels.push(plot('flux',t('공간에 따라 감소하는 확산 유입','Diffusive inflow decreases through the slab'),[spatial('j=−D cx',q=>q.diffusiveFlux.value)],'ξ = x/L','kg/(m² s)',t('j_x=−kc입니다. 농도와 달리 밀폐면의 플럭스는 정확히0입니다.','j_x=−kc. Unlike concentration, the sealed-plane flux is exactly zero.')));
    panels.push(plot('source',t('국소 체적 소모율','Local volumetric consumption rate'),[spatial('kc',q=>q.localConsumption.value)],'ξ = x/L','kg/(m³ s)',t('적분할 때 실제 길이와 단면적을 함께 곱합니다.','Include physical length and cross-sectional area when integrating.')));
    panels.push(plot('inventory',t('부분 체적의 저장 질량','Inventory in the partial volume'),[spatial('M(0,x)',q=>q.partialStoredMass.value)],'ξ = x/L','kg',t('저장 질량과 소모율은 서로 단위가 다릅니다. 정상 상태도 비영 저장량을 가질 수 있습니다.','Inventory and consumption have different units. A steady state can retain nonzero inventory.')));
    panels.push(plot('tail',t('비영 밀폐 꼬리의 로그','Log of the nonzero sealed-plane tail'),[spatial('ln c',q=>q.concentration.logMagnitude)],'ξ = x/L','ln(c / (kg/m³))',t('큰 φ에서 c(L)≈2c0 exp(−φ)입니다. 반무한 지수의 계수로 바꾸거나 비영 꼬리를0으로 만들지 않습니다.','At large φ, c(L)≈2c0 exp(−φ). Retain the reflecting-wall factor and the nonzero tail.')));
    panels.push(plot('small',t('작은 반응에서도 남는 고갈','Depletion retained at a small reaction rate'),[
      line('ln(1−C)',r.depletion.map(s=>[s.logRate,s.value.stableDepletion.logMagnitude])),line('ln(1−η)',r.depletion.map(s=>[s.logRate,s.value.stableEffectivenessDeficit.logMagnitude]),'orange')],
      'log10(k / (1/s))','ln(deficit)',t('C나 η가1로 반올림되어도 안정 부족분은 남습니다. 표면의 농도 고갈은 정확히0입니다.','Stable deficits remain when C or η rounds to one. Surface depletion is exactly zero.')));
    panels.push(plot('similarity',t('같은 Da의 정규화 기저','Normalized basis at matching Da'),[
      spatial(t('원래 기저','Original basis'),q=>q.basisShape.value),line(t('닮음 기저','Similar basis'),r.spatial.map(s=>[s.xi,s.comparison?.basisShape.value??null]),'orange')],
      'ξ = x/L','1',t('L′=aL,D′=bD,k′=(b/a²)k,A′=a²A입니다. 시간·질량·소모는 각 차원 배율을 따릅니다.','L′=aL, D′=bD, k′=(b/a²)k, A′=a²A. Time, mass and consumption follow their dimensional scaling.')));
    if (conceptStep===undefined) {
      panels.push(table('query',t('현재 반응·확산과 반쪽 판 수지','Current reaction–diffusion and half-slab budget'),[
        ['Da',q.damkohler,'1'],['φ',q.thiele,'1'],['x',q.position,'m'],['c',q.concentration,'kg/m³'],['c(L)',r.sealed.concentration,'kg/m³'],
        ['dc/dx',q.gradient,'kg/m⁴'],['j(x)',q.diffusiveFlux,'kg/(m² s)'],['j(0)',q.inletFlux,'kg/(m² s)'],['R half slab',q.totalConsumption,'kg/s'],['M half slab',q.storedMass,'kg'],
        ['R(0,x)',q.partialConsumption,'kg/s'],['M(0,x)',q.partialStoredMass,'kg'],['η basis',q.analyticEffectiveness,'1'],['1−C stable',q.stableDepletion,'1'],['1−η stable',q.stableEffectivenessDeficit,'1'],
        ['tD',q.diffusionTime,'s'],['tR',q.reactionTime,'s'],['penetration length',q.penetrationLength,'m']]));
      panels.push(note('da-normalization',`${t('측정 c/c0','Measured c/c0')}: ${q.observedNormalizedShape===null?unknown:state(q.observedNormalizedShape)}. ${t('측정 효과도','Observed effectiveness')}: ${q.observedEffectiveness===null?unknown:state(q.observedEffectiveness)}.`));
      if (r.comparison) panels.push(table('similarity',t('닮은 판의 실제 관측','Physical observations of the similar slab'),[
        ['Da′',r.comparison.damkohler,'1'],['tD′',r.comparison.diffusionTime,'s'],['M′',r.comparison.storedMass,'kg'],['R′',r.comparison.totalConsumption,'kg/s']]));
      else panels.push(note('da-range',r.comparisonStatus==='nonzero-coefficient-outside-arithmetic-range'?t('변환한 비영 반응률을 충분한 상대 정확도로 표현할 수 없어 비교를 제공하지 않습니다. 원래 판은 유지합니다.','The transformed nonzero reaction rate cannot be represented with sufficient relative accuracy, so comparison is unavailable. The original slab remains valid.'):t('변환한 계수가 선언 범위 밖이라 닮음 비교를 제공하지 않습니다. 원래 판의 해는 유지합니다.','Transformed coefficients are outside the declared range, so similarity is unavailable. The original slab remains valid.')));
    }
  }
  return <div className="ai-matrix-panels">{conceptStep===undefined?panels:panels[conceptStep]}</div>;
}
