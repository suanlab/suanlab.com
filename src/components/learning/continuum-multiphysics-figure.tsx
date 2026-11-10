'use client';
import{useLanguage}from '@/components/language-provider';
import type{ContinuumResult}from '@/lib/learning-continuum';
import type{MultiphysicsObservation}from '@/lib/learning-continuum-multiphysics';
import type{Vec2}from '@/lib/learning-simulations';
import{WavePlot,waveNumber as fmt,type Series}from './waves-plot';
type MultiphysicsSeries = Series & { extent: [number, number] };
type Point=[number|null,number|null];
export default function ContinuumMultiphysicsFigure({result,conceptStep}:{result:ContinuumResult;conceptStep?:number}){
 const{language}=useLanguage(),t=(ko:string,en:string)=>language==='ko'?ko:en,panels:React.ReactNode[]=[];
 const prefix=result.thermoelastic?'continuum-thermoelastic':'continuum-thermoelectric';
 const line=(name:string,points:Point[],color='blue',discrete=false):MultiphysicsSeries=>{
  const xs=points.map(([x])=>x).filter((x):x is number=>x!==null&&Number.isFinite(x));
  const segments:Vec2[][]=[];let part:Vec2[]=[];
  for(const[x,y]of points){if(x===null||y===null||!Number.isFinite(x)||!Number.isFinite(y)){if(part.length)segments.push(part);part=[];}else if(discrete)segments.push([[x,y]]);else part.push([x,y]);}
  if(part.length)segments.push(part);return{name,color,segments,extent:[Math.min(0,...xs),Math.max(0,...xs)]};
 };
 const plot=(id:string,title:string,series:MultiphysicsSeries[],x:string,unit:string,caption:string,vertical:number[]=[])=>{
  const left=Math.min(0,...series.map(s=>s.extent[0])),right=Math.max(0,...series.map(s=>s.extent[1]));
  const magnitude=Math.max(0,...series.flatMap(s=>s.segments.flat().map(p=>Math.abs(p[1])))),scale=magnitude>1e5||magnitude>0&&magnitude<.001?magnitude:1;
  return <WavePlot key={id} marker={`${prefix}-${id}`} title={title} series={series.map(s=>({...s,segments:s.segments.map(part=>part.map(([x,y])=>[x,y/scale]as Vec2))}))} xDomain={[left,right>left?right:left+.01]} xLabel={x} yLabel={scale===1?unit:`y × ${fmt(scale)} · ${unit}`} minimumYPadding={magnitude===0?.01:1e-100} vertical={vertical} caption={caption+(scale===1?'':` ${t('실제 값 = 세로 좌표 ×','Physical value = ordinate ×')} ${fmt(scale)}.`)}/>;
 };
 type Row=[string,MultiphysicsObservation|null,string];
 const state=(q:MultiphysicsObservation|null)=>q===null?t('현재 조건에서 미정','Undefined at current conditions'):q.sign===0?t('정확한0','Exact zero'):q.value===null?t('비영·일반 값 불가','Nonzero; linear value unavailable'):fmt(q.value);
 const table=(id:string,title:string,rows:Row[])=><figure key={id} data-continuum-table={`${prefix}-${id}`}><div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption>{title}</caption><thead><tr>{[t('관측','Observation'),t('일반 값·상태','Linear value or state'),t('부호','Sign'),'ln |q / unit|',t('단위','Unit')].map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{rows.map(([name,q,unit],i)=><tr key={i}><th scope="row">{name}</th><td>{state(q)}</td><td>{q===null?'—':q.sign}</td><td>{fmt(q?.logMagnitude)}</td><td>{unit}</td></tr>)}</tbody></table></div></figure>;
 const note=(id:string,text:string)=><p key={id} className="ai-logic-statement" data-multiphysics-note={id}>{text}</p>;
 const diagnostic=t('같은 선언 가족의 진단이며 독립 정확도 인증이나 엄밀한 구적 오차 한계가 아닙니다.','A diagnostic of the declared family, not independent accuracy certification or a rigorous quadrature error bound.');
 if(result.thermoelastic){
  const r=result.thermoelastic,q=r.query,o=r.oneWay,w=r.window,ow=r.oneWayWindow;
  type State=typeof q;
  const time=(name:string,get:(q:State)=>number|null,color='blue',oneWay=false)=>line(name,r.temporal.map(s=>[s.time,get(oneWay?s.oneWay:s.value)]),color);
  const space=(name:string,get:(q:State)=>number|null,color='blue')=>line(name,r.spatial.map(s=>[s.x,get(s.value)]),color);
  panels.push(plot('reciprocity',t('열의 역작용이 바꾸는 온도 변화율','Reverse coupling changes the temperature rate'),[
   space(t('상호 결합','Reciprocal'),v=>v.temperatureRate.value),
   line(t('일방향 근사','One-way approximation'),r.spatial.map(s=>[s.x,s.oneWay.temperatureRate.value]),'orange')],
   'x · m','K/s',t('같은 초기 데이터에서 두 실제 모델을 풉니다. 역방향 −T0 β ε_t 항이 빠지면 온도 변화도 달라집니다.','Both actual models use the same initial data. Removing −T0 β ε_t changes the thermal evolution.'),[r.phase/r.parameters.wavenumber]));
  panels.push(plot('availability',t('역학·열 가용 에너지와 합','Mechanical, thermal and total availability'),[
   time(t('역학','Mechanical'),v=>v.mechanicalAvailability.value),time(t('열','Thermal'),v=>v.thermalAvailability.value,'orange'),time(t('합','Total'),v=>v.totalAvailability.value,'green')],
   't · s','J',t('전체 내부 에너지나 누적 열량이 아닌 기준 상태 근처의 이차 가용 에너지입니다. κ0에서도 두 부분은 교환할 수 있습니다.','Quadratic availability near the reference state, not total internal energy or accumulated heat. At κ=0 the two parts can still exchange.'),[r.time]));
  panels.push(plot('power',t('상호 교환과 전도 소산','Reciprocal exchange and conduction loss'),[
   time(t('역학 교환','Mechanical exchange'),v=>v.mechanicalExchangeRate.value),time(t('열 교환','Thermal exchange'),v=>v.thermalExchangeRate.value,'orange'),time(t('전도 소산','Conduction loss'),v=>v.conductionAvailabilityLoss.value,'green')],
   't · s','W',t('교환은 부호가 있고 두 장에서 상쇄됩니다. 전도 소산은 비음수입니다. 시간 적분J와 현재 전력W를 구분하세요.','Exchange is signed and cancels between fields. Conduction loss is nonnegative. Distinguish integrated joules from instantaneous watts.')));
  panels.push(plot('one-way',t('다른 모델의 실제 가용 에너지','Actual availability of two different models'),[
   time(t('상호 모델','Reciprocal model'),v=>v.totalAvailability.value),time(t('일방향 모델','One-way model'),v=>v.totalAvailability.value,'orange',true)],
   't · s','J',t('일방향은 다른 궤적입니다. 남는 교환은 근사나 외부 열 제어 포트를 선언하여 해석합니다. 상호 모델의 닫힌 수지를 그대로 사용하지 않습니다.','One-way coupling has a different trajectory. Interpret its retained exchange through a declared approximation or external heat controller; the closed reciprocal balance does not apply.')));
  panels.push(plot('displacement',t('실제 공간의 변위','Displacement over physical space'),[space('u',v=>v.displacement.value)],'x · m','m',t('진폭a와 한 점의 변위u=a sin(kx)는 다릅니다. 실제 셀 길이는2π/k입니다.','Amplitude a differs from point displacement u=a sin(kx). The actual cell length is2π/k.')));
  panels.push(plot('temperature',t('기준 온도에서의 편차','Deviation from the reference temperature'),[space('θ',v=>v.temperatureDeviation.value)],'x · m','K',t('T0=300K,ρ=1kg/m³,C=1J/(m³K)를 고정합니다. 음의 편차는 음의 절대온도가 아닙니다.','T0=300K,ρ=1kg/m³,C=1J/(m³ K) are fixed. A negative deviation is not a negative absolute temperature.')));
  panels.push(plot('stress',t('열팽창과 실제 응력','Thermal expansion and actual stress'),[space('σ',v=>v.stress.value)],'x · m','Pa',t('σ=E ε−β θ이며 두 장이 같은 실제 위치에서 결합합니다.','σ=E ε−β θ couples both fields at the same physical position.')));
  panels.push(plot('heat-flux',t('실제 Fourier 열 플럭스','Physical Fourier heat flux'),[space('q',v=>v.heatFlux.value)],'x · m','W/m²',t('열 플럭스와 전체 주기 셀의 가용 에너지 소산은 다른 단위의 관측입니다.','Heat flux and integrated availability-loss power have different units.')));
  panels.push(plot('amplitude',t('온도 진폭의 두 실제 이력','Two actual temperature-amplitude histories'),[
   time(t('상호','Reciprocal'),v=>v.temperatureAmplitude.value),time(t('일방향','One-way'),v=>v.temperatureAmplitude.value,'orange',true)],'t · s','K',t('초기 변위·속도·온도를 따로 지정합니다. 현재 온도만으로 모든 상태를 결정하지 않습니다.','Initial displacement, velocity and temperature are specified separately. Current temperature alone does not determine the whole state.')));
  panels.push(plot('tail',t('온도 진폭의 비영 꼬리','Nonzero temperature-amplitude tail'),[
   time(t('상호 로그','Reciprocal log'),v=>v.temperatureAmplitude.logMagnitude),time(t('일방향 로그','One-way log'),v=>v.temperatureAmplitude.logMagnitude,'orange',true)],'t · s','ln(|b| / K)',t('정확한0에는 로그가 없고 비영 표현 불가에는 부호·로그가 남습니다. 감쇠 꼬리와 작은 변화는 따로 보존합니다.','Exact zero has no log; unavailable nonzero values retain sign and log. Decaying tails and tiny changes are retained separately.')));
  panels.push(plot('small',t('작은 시간의 생성 온도와 소산','Generated temperature and loss at small times'),[
   line('ln |b|',r.small.map(s=>[s.logTime,s.value.temperatureAmplitude.logMagnitude])),line('ln loss',r.small.map(s=>[s.logTime,s.window.conductionAvailabilityLoss.logMagnitude]),'orange')],
   'log10(t / s)','ln |q / unit|',t('b는K,누적 소산은J로 나눈 로그입니다. a0=b0=0,v0≠0이면 b는1차,누적 소산은3차로 시작합니다. 다른 초기 상태에는 다른 선도 차수가 가능합니다.','Logs use b/K and loss/J. With a0=b0=0,v0≠0, b starts linearly and loss cubically; other initial states can have different leading orders.')));
  panels.push(plot('velocity-change',t('반올림된 속도와 별도의 작은 변화','Rounded velocity and a separate tiny change'),[
   line('ln |Δv|',r.small.map(s=>[s.logTime,s.value.velocityChange.logMagnitude]))],
   'log10(t / s)','ln(|Δv| / (m/s))',t('현재 속도가 초기와 같게 반올림되어도 안정적인 변화는 비영일 수 있습니다.','The stable change can be nonzero even when velocity rounds to its initial value.')));
  if(conceptStep===undefined){
   panels.push(table('query',t('현재 열탄성 관측과 전체 셀','Current thermoelastic observations and cell'),[
    ['a',q.displacementAmplitude,'m'],['v',q.velocityAmplitude,'m/s'],['b',q.temperatureAmplitude,'K'],['Δa stable',q.displacementChange,'m'],['Δv stable',q.velocityChange,'m/s'],['Δb stable',q.temperatureChange,'K'],
    ['u(x)',q.displacement,'m'],['ε(x)',q.strain,'1'],['T(x)',q.absoluteTemperature,'K'],['σ(x)',q.stress,'Pa'],['q(x)',q.heatFlux,'W/m²'],
    ['mechanical availability',q.mechanicalAvailability,'J'],['thermal availability',q.thermalAvailability,'J'],['total availability',q.totalAvailability,'J'],['mechanical exchange',q.mechanicalExchangeRate,'W'],['thermal exchange',q.thermalExchangeRate,'W'],['conduction loss',q.conductionAvailabilityLoss,'W'],['strain bound reciprocal',q.reciprocalStrainBound,'1']]));
   panels.push(note('cell',`${t('실제 주기 길이','Physical cell length')}: ${fmt(q.cellLength)} m; ${t('체적','volume')}: ${fmt(q.cellVolume)} m³. ${t('공간 위상','Spatial phase')}: ${fmt(r.phase)} rad. ρ=1 kg/m³; C=1 J/(m³ K); T0=300 K.`));
   if(w&&ow)panels.push(table('window',t('지정한 실제 시간 구간의 적분','Integrals over the selected time interval'),[
    ['reciprocal exchange',w.mechanicalExchange,'J'],['reciprocal conduction loss',w.conductionAvailabilityLoss,'J'],['reciprocal balance change',w.balanceDerivedAvailabilityChange,'J'],['one-way exchange',ow.mechanicalExchange,'J'],['one-way balance change',ow.balanceDerivedAvailabilityChange,'J']]));
   panels.push(note('window',`${t('적분 시작/끝','Integration start/end')}: ${fmt(r.start)} / ${fmt(r.end)} s. ${w?t('반올림된 끝점 차','Rounded endpoint difference')+': '+fmt(r.roundedWindowChange)+' J. '+diagnostic:t('시작이 끝보다 늦어 적분을 제공하지 않습니다. 실제 장은 유지합니다.','Start is after end, so the integral is unavailable. The actual field remains valid.')}`));
   panels.push(table('one-way',t('일방향 모델의 실제 상태와 수지','Actual one-way state and balance'),[['b one-way',o.temperatureAmplitude,'K'],['availability one-way',o.totalAvailability,'J'],['exchange retained',o.mechanicalExchangeRate,'W'],['reverse exchange omitted',o.thermalExchangeRate,'W'],['availability rate one-way',o.totalAvailabilityRate,'W']]));
  }
 }
 if(result.thermoelectric){
  const r=result.thermoelectric,q=r.query;
  const spatial=(name:string,get:(q:typeof r.query)=>number|null,color='blue')=>line(name,r.spatial.map(s=>[s.xi,get(s.value)]),color);
  panels.push(plot('temperature',t('경계 온도와 Joule 상승','Boundary temperatures and Joule rise'),[
   spatial('T',v=>v.temperature.value),line(t('경계 직선','Boundary line'),r.spatial.map(s=>[s.xi,r.parameters.leftTemperature+(r.parameters.rightTemperature-r.parameters.leftTemperature)*s.xi]),'orange')],
   'ξ = x/L','K',t('전류j를 지정하고 단자 전압을 계산합니다. 추가 포물선의 최대와 전체 온도의 최대를 구분합니다.','Current j is prescribed and terminal voltage is computed. Distinguish the maximum added bulge from the maximum total temperature.'),[r.xi]));
  panels.push(plot('heat',t('Fourier·Peltier·전체 열 플럭스','Fourier, Peltier and total heat flux'),[
   spatial('Fourier',v=>v.fourierHeatFlux.value),spatial('Peltier',v=>v.peltierHeatFlux.value,'orange'),spatial(t('합','Total'),v=>v.totalHeatFlux.value,'green')],
   'ξ = x/L','W/m²',t('Peltier 계수는Π=ST이며 절대온도를 사용합니다. 가역 수송을 추가Joule 손실로 세지 않습니다.','Peltier coefficient Π=ST uses absolute temperature. Reversible transport is not an extra Joule loss.')));
  panels.push(plot('local-work',t('국소 전기 일과 Joule 가열','Local electrical work and Joule heating'),[
   spatial('jE',v=>v.localElectricalWork.value),spatial('rj²',v=>v.localJouleHeating.value,'orange'),spatial('dq/dx',v=>v.heatFluxGradient.value,'green')],
   'ξ = x/L','W/m³',t('jE=rj²+SjT_x입니다. 전기 일이Joule만과 같지는 않습니다. ','jE=rj²+SjT_x. Electrical work need not equal Joule heating alone. ')+diagnostic));
  panels.push(plot('ports',t('경계 열과 전기 입력의 실제 포트','Actual boundary heat and electrical ports'),[
   line('W',[[1,q.leftHeatInput.value],[2,q.rightHeatOutput.value],[3,q.electricalPowerInput.value],[4,q.joulePower.value],[5,q.reversibleElectricalPower.value]],'blue',true)],
   '1=QL; 2=QR; 3=Pin; 4=J; 5=rev','W',t('좌우를 같은+x방향으로 정의했으므로 QR−QL=Pin입니다. 음의Pin은 발전이며 냉각COP조건과 구분합니다.','Both face fluxes use+x, so QR−QL=Pin. Negative Pin denotes generation, distinct from cooling-COP conditions.')));
  panels.push(plot('potential',t('기준 전위와 실제 전위 변화','Reference potential and actual potential change'),[
   spatial('V(x)',v=>v.potential.value),spatial('V(x)−V(0)',v=>v.stablePotentialChange.value,'orange')],'ξ = x/L','V',t('공통 전위 이동은 전기장·열 플럭스·전력을 바꾸지 않습니다. 전위와 전위차를 구별하세요.','A common potential shift changes neither field, heat flux nor power. Distinguish potential from potential difference.')));
  panels.push(plot('field',t('실제 전기장','Physical electric field'),[spatial('E=−V_x',v=>v.electricField.value)],'ξ = x/L','V/m',t('저항 전기장과 Seebeck 온도 구배 항을 함께 포함합니다.','Includes both resistive and Seebeck temperature-gradient contributions.')));
  panels.push(plot('partial',t('조회 위치까지의 실제 전기 일','Electrical work to the observation position'),[
   spatial(t('전체 전기 일','Total electrical work'),v=>v.partialElectricalPower.value),spatial('Joule',v=>v.partialJoulePower.value,'orange')],'ξ = x/L','W',t('체적 적분은 실제 길이와 면적을 포함합니다. 전체 전기 일과 불가역Joule 기여는 다를 수 있습니다.','Volume integrals include physical length and area. Total electrical work and irreversible Joule contribution can differ.')));
  panels.push(plot('rise',t('별도로 보존한 Joule 온도 상승','Separately retained Joule temperature rise'),[spatial('ΔT Joule',v=>v.stableJouleTemperatureRise.value)],'ξ = x/L','K',t('절대온도가 경계 직선으로 반올림되어도 양의 작은 상승이 남을 수 있습니다. 양 끝의 추가 상승은 정확히0입니다.','A small positive rise can remain when absolute temperature rounds to the boundary line. The added rise is exactly zero at both ends.')));
  panels.push(plot('response',t('전류에 따른 경계 열과 전기 일','Boundary heat and work versus current'),[
   line('QL',r.response.map(s=>[s.current,s.value.leftHeatInput.value])),line('Pin',r.response.map(s=>[s.current,s.value.electricalPowerInput.value]),'orange')],
   'j · A/m²','W',t('다른 계수는 현재 입력으로 유지하고 전류만 변화시킵니다. 냉각에는 QL>0과Pin>0이 함께 필요합니다.','Other coefficients stay at current values while current is swept. Cooling requires both QL>0 and Pin>0.'),[r.parameters.currentDensity]));
  panels.push(plot('small',t('작은 전류의 비영 이차 효과','Nonzero quadratic effects of small current'),[
   line('ln |Pin|',r.tiny.map(s=>[s.logCurrent,s.value.electricalPowerInput.logMagnitude])),line('ln Joule',r.tiny.map(s=>[s.logCurrent,s.value.joulePower.logMagnitude]),'orange')],
   'log10(|j| / (A/m²))','ln(|power| / W)',t('온도차0에서는 두 전력이j²에 비례합니다. 다른 조건의Pin에는 가역 항도 포함됩니다. 부호는 표에서 읽으세요.','At zero temperature difference both powers scale as j². Otherwise Pin includes a reversible term. Read signs in the table.')));
  if(conceptStep===undefined){
   panels.push(table('query',t('현재 열전 막대의 관측과 경계 수지','Current thermoelectric observations and port balance'),[
    ['x',q.position,'m'],['j',q.currentDensity,'A/m²'],['I',q.current,'A'],['R',q.resistance,'Ω'],['T',q.temperature,'K'],['Joule temperature rise',q.stableJouleTemperatureRise,'K'],['T_x',q.temperatureGradient,'K/m'],
    ['V',q.potential,'V'],['ΔV',q.terminalVoltageDrop,'V'],['stable potential change',q.stablePotentialChange,'V'],['E',q.electricField,'V/m'],['Fourier heat',q.fourierHeatFlux,'W/m²'],['Peltier heat',q.peltierHeatFlux,'W/m²'],['total heat',q.totalHeatFlux,'W/m²'],
    ['local electrical work',q.localElectricalWork,'W/m³'],['local Joule heating',q.localJouleHeating,'W/m³'],['QL',q.leftHeatInput,'W'],['QR',q.rightHeatOutput,'W'],['Pin',q.electricalPowerInput,'W'],['Joule power',q.joulePower,'W'],['reversible power',q.reversibleElectricalPower,'W'],['stable QR−QL',q.stableBoundaryHeatDifference,'W'],['partial electrical work',q.partialElectricalPower,'W'],['left cooling COP',q.leftCoolingCOP,'1']]));
   const regime=q.electricalRegime==='generation'?t('발전','Generation'):q.electricalRegime==='zero-terminal-work'?t('영 단자 전기 일','Zero terminal electrical work'):t('전기 입력','Electrical input');
   const cop=q.leftCoolingStatus==='defined'?t('선언한 포트의 냉각 비가 정의됩니다. 일반 값은 표현 범위 밖일 수 있습니다.','The cooling ratio is defined for these ports; its ordinary value may be outside the numeric range.'):q.leftCoolingStatus==='no-positive-left-heat-input'?t('양의 왼쪽 열 유입이 없어 이 냉각 비는 미정입니다.','No positive left heat input; this cooling ratio is undefined.'):t('양의 전기 입력이 없어 이 냉각 비는 미정입니다.','No positive electrical input; this cooling ratio is undefined.');
   panels.push(note('ports',`${regime}. ${cop} ${t('직접 반올림한QR−QL','Direct rounded QR−QL')}: ${fmt(q.roundedBoundaryHeatDifference)} W. ${t('정상 상수 물성 단일 막대이며 실제 장치의 모든 접촉·반환 경로를 포함하지 않습니다.','A steady constant-property single leg, without all contacts and return paths of a complete device.')}`));
  }
 }
 return <div className="ai-matrix-panels">{conceptStep===undefined?panels:panels[conceptStep]}</div>;
}
