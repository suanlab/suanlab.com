'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { temperatureHeatWorkLabs } from '@/data/interactive-book';
import { evaluateThermalContact, evaluateGasPaths, validThermalContactValues, validGasPathValues, type TemperatureHeatWorkKind } from '@/lib/learning-temperature-heat-work';
import type { BinaryObservation } from '@/lib/learning-binary-products';
import type { Vec2 } from '@/lib/learning-simulations';
import GuidedExperiments from './guided-experiments';
import NumberField from './thermodynamic-number-field';
import { useSavedState } from './use-saved-state';
import { WavePlot, waveNumber as fmt, type Series } from './waves-plot';

type Thermal = ReturnType<typeof evaluateThermalContact>;
type Gas = ReturnType<typeof evaluateGasPaths>;
type Row = [string, BinaryObservation | null, string];
const observed = (value: number): BinaryObservation => ({value,sign:Math.sign(value),logMagnitude:value===0?null:Math.log(Math.abs(value))});
function line(name: string, points: [number, number | null][], color: string): Series {
  const segments: Vec2[][]=[]; let segment:Vec2[]=[];
  for(const [x,y] of points){if(y===null){if(segment.length)segments.push(segment);segment=[];}else segment.push([x,y]);}
  if(segment.length)segments.push(segment);
  return {name,color,segments};
}
function Figures({thermal,gas,only}:{thermal?:Thermal;gas?:Gas;only?:number}) {
  const {language}=useLanguage();const t=(ko:string,en:string)=>language==='ko'?ko:en;
  const panels:React.ReactNode[]=[];
  if(thermal){const r=thermal,q=r.query,h=r.history;
    const add=(marker:string,title:string,series:Series[],unit:string,caption:string,dots:{point:Vec2;color:string}[]=[])=>panels.push(<WavePlot key={marker} marker={`thermal-work-${marker}`} title={title} series={series} dots={dots} yDomain={marker==='temperature'?[240,360]:marker==='production'?[0,Math.max(...h.map(s=>s.entropyProductionRate.value??0))*1.08||1]:undefined} xDomain={[0,8]} xLabel={t('시간 (s)','Time (s)')} yLabel={unit} caption={caption}/>);
    add('temperature',t('온도와 가중 평형','Temperatures and weighted equilibrium'),[line('T₁',h.map(s=>[s.time,s.temperature1]),'blue'),line('T₂',h.map(s=>[s.time,s.temperature2]),'orange'),line('Tₑ',[[0,q.equilibriumTemperature],[8,q.equilibriumTemperature]],'green')],'K',t('열용량이 큰 물체의 온도 변화가 작습니다.','The larger heat capacity undergoes a smaller temperature change.'),[{point:[q.time,q.temperature1],color:'blue'},{point:[q.time,q.temperature2],color:'orange'}]);
    add('energy',t('두 물체의 내부 에너지 변화','Internal-energy changes of both bodies'),[line('ΔU₁',h.map(s=>[s.time,s.energyChange1.value]),'blue'),line('ΔU₂ = Q₁→₂',h.map(s=>[s.time,s.energyChange2.value]),'orange')],'J',t('같은 내부 경계를 건너는 에너지의 부호가 반대입니다.','Energy crossing the same internal boundary has opposite signs for the two bodies.'));
    add('entropy',t('개별 엔트로피와 전체 변화','Individual and total entropy changes'),[line('ΔS₁',h.map(s=>[s.time,s.entropyChange1.value]),'blue'),line('ΔS₂',h.map(s=>[s.time,s.entropyChange2.value]),'orange'),line('ΔS',h.map(s=>[s.time,s.entropyChange.value]),'green')],'J/K',t('합친 변화는 양의 생성 적분으로 계산합니다. 작은 전체 변화는 아래 표에서도 확인하세요.','The total uses a positive production integral. Inspect the table for small total changes.'));
    add('rate',t('순간 열전달률','Instantaneous heat-transfer rate'),[line('Φ₁→₂',h.map(s=>[s.time,s.heatRate.value]),'blue')],'W',t('순간 전력이며 누적 에너지J와 다릅니다.','This is power, distinct from accumulated energy in joules.'));
    add('production',t('엔트로피 생성률','Entropy production rate'),[line('dS/dt',h.map(s=>[s.time,s.entropyProductionRate.value]),'green')],'W/K',t('외부 열 교환이 없어도 내부에서 엔트로피가 생성됩니다.','Entropy is produced internally without external heat exchange.'));
    add('difference',t('모형에 남은 온도차','Remaining model temperature difference'),[line('T₁−T₂',h.map(s=>[s.time,s.temperatureDifference.value]),'orange')],'K',t('표시 온도가 같아도 작은 차이는 남을 수 있습니다.','A small difference can remain when displayed temperatures agree.'));
  }
  if(gas){const r=gas,h=r.history,a=r.queryA,b=r.queryB;const volumes=[r.parameters.initialVolume,r.parameters.finalVolume];let lo=Math.min(...volumes),hi=Math.max(...volumes);if(lo===hi){lo*=.9;hi*=1.1;}
    panels.push(<WavePlot key="pv" marker="thermal-work-pv" title={t('실제 압력·부피 경로','Actual pressure-volume paths')} series={[line('A',h.map(s=>[s.a.volume,s.a.pressure]),'blue'),line('B',h.map(s=>[s.b.volume,s.b.pressure]),'orange')]} dots={[{point:[a.volume,a.pressure],color:'blue'},{point:[b.volume,b.pressure],color:'orange'}]} xDomain={[lo,hi]} yDomain={[0,Math.max(...h.flatMap(s=>[s.a.pressure,s.b.pressure]))*1.08]} xLabel="V (m³)" yLabel="p (Pa)" caption={t('점은 현재 위치입니다. 수직 구간의 체적 일은0입니다.','Dots mark current positions. Vertical segments perform no volume work.')}/>);
    for(const [key,ko,en,unit] of [['work','누적 일','Accumulated work','J'],['entropyChange','엔트로피 변화','Entropy change','J/K'],['heat','누적 열','Accumulated heat','J'],['energyChange','내부 에너지 변화','Internal-energy change','J']] as const){
      panels.push(<WavePlot key={key} marker={`thermal-work-${key}`} title={t(ko,en)} series={[line('A',h.map(s=>[s.progress,s.a[key]?.value??null]),'blue'),line('B',h.map(s=>[s.progress,s.b[key]?.value??null]),'orange')]} dots={[a,b].flatMap((s,i)=>s[key]?.value==null?[]:[{point:[s.progress,s[key]!.value!] as Vec2,color:i?'orange':'blue'}])} xDomain={[0,1]} xLabel={t('진행률 · 시간 아님','Progress, not time')} yLabel={unit} caption={t('같은 진행률의 중간 상태는 다를 수 있습니다. 미정·표현 불가 값은 잇지 않습니다.','Intermediate states can differ at equal progress. Unresolved or out-of-range values are not connected.')}/>);
    }
  }
  return <div className="ai-matrix-panels">{only===undefined?panels:panels[only]}</div>;
}
function Numbers({rows}:{rows:Row[]}){
  const {language}=useLanguage(),ko=language==='ko';return <div className="ai-readout" data-testid="lab-readout"><h3>{ko?'수치와 작은 변화':'Numbers and small changes'}</h3><p className="ai-small">{ko?'표시값은 반올림합니다. 부호와 해당 단위로 쓴 크기의 자연로그를 함께 확인하세요. —는 연결 없음·미정·표현 불가이며 실제0과 다릅니다.':'Displays are rounded. Compare sign and the natural log of magnitude in the stated unit. — marks disconnection, unresolved or out-of-range values, distinct from physical zero.'}</p><div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'열역학 관측 표':'Thermodynamic observations'}><table><thead><tr>{(ko?['관측','값','단위','부호','로그 크기']:['Observation','Value','Unit','Sign','Log magnitude']).map(s=><th key={s} scope="col">{s}</th>)}</tr></thead><tbody>{rows.map(([label,q,unit])=><tr key={label}><th scope="row">{label}</th><td>{fmt(q?.value)}</td><td>{unit}</td><td>{q===null?'—':q.sign}</td><td>{fmt(q?.logMagnitude)}</td></tr>)}</tbody></table></div></div>;
}
export function TemperatureHeatWorkConcept({kind,step}:{kind:TemperatureHeatWorkKind;step:number}) {
  const thermal=kind==='finite-body-thermal-contact-heat-entropy';const spec=temperatureHeatWorkLabs[kind];
  const preset=(thermal?[0,1,8,3]:[0,4,9,11])[step];
  const result=useMemo(()=>thermal?{thermal:evaluateThermalContact(spec.presets[preset].values)}:{gas:evaluateGasPaths(spec.presets[preset].values)},[thermal,spec,preset]);
  const selected=result.thermal?[observed(result.thermal.query.equilibriumTemperature),result.thermal.query.energyChange1,result.thermal.query.entropyChange,result.thermal.query.heatRate][step]:result.gas?[observed(result.gas.queryA.pressure),result.gas.queryA.work,result.gas.queryA.entropyChange,result.gas.queryA.heat][step]:null;
  return <><Figures {...result} only={[0,1,2,3][step]}/><p className="ai-small" data-thermal-concept-value="true">{thermal?['Tₑ','ΔU₁','ΔS','Φ'][step]:['p_A','W_A','ΔS_A','Q_A'][step]} = {fmt(selected?.value)} {thermal?['K','J','J/K','W'][step]:['Pa','J','J/K','J'][step]}</p></>;
}
export default function TemperatureHeatWorkLab({kind}:{kind:TemperatureHeatWorkKind}){
  const {language}=useLanguage(),ko=language==='ko',t=(a:string,b:string)=>ko?a:b;
  const thermal=kind==='finite-body-thermal-contact-heat-entropy',spec=temperatureHeatWorkLabs[kind];
  const initial=useMemo(()=>spec.controls.map(c=>c.initial),[spec]);
  const [values,set,ready,storage]=useSavedState(`suanlab-learning:${kind}:lab`,initial,thermal?validThermalContactValues:validGasPathValues);
  const [playing,setPlaying]=useState(false),[formEpoch,setFormEpoch]=useState(0),panel=useRef<HTMLElement>(null),active=useSceneActivity(panel),motion=useSiteMotion();
  const result=useMemo(()=>thermal?{thermal:evaluateThermalContact(values)}:{gas:evaluateGasPaths(values)},[thermal,values]);
  const clock=5,limit=thermal?8:1,increment=thermal ? .1 : .02;
  const ended=values[clock]>=limit;
  useEffect(()=>{if(!active||ended)setPlaying(false);},[active,ended]);
  useEffect(()=>{if(!playing||!active)return;const timer=setInterval(()=>set(v=>v.map((x,i)=>i===clock?Math.min(limit,Number((x+increment).toPrecision(14))):x)),200);return()=>clearInterval(timer);},[playing,active,set,clock,increment,limit]);
  const load=(v:number[])=>{setPlaying(false);set([...v]);setFormEpoch(n=>n+1);};
  const rows:Row[]=[];
  if(result.thermal){const q=result.thermal.query,w=result.thermal.window;
    rows.push(['T₁',observed(q.temperature1),'K'],['T₂',observed(q.temperature2),'K'],['Tₑ',observed(q.equilibriumTemperature),'K'],['T₁−T₂',q.temperatureDifference,'K'],['Φ₁→₂',q.heatRate,'W'],['Q₁→₂',q.heat,'J'],['ΔU₁',q.energyChange1,'J'],['ΔU₂',q.energyChange2,'J'],['ΔS₁',q.entropyChange1,'J/K'],['ΔS₂',q.entropyChange2,'J/K'],['ΔS',q.entropyChange,'J/K'],['dS/dt',q.entropyProductionRate,'W/K'],['τ',q.relaxationTime,'s'],[t('구간 전달량','Window transfer'),w?.heat??null,'J'],[t('구간 엔트로피 변화','Window entropy change'),w?.entropyChange??null,'J/K']);
  }
  if(result.gas){for(const [name,q]of[['A',result.gas.queryA],['B',result.gas.queryB]]as const)rows.push([`${name} · T`,observed(q.temperature),'K'],[`${name} · V`,observed(q.volume),'m³'],[`${name} · p`,observed(q.pressure),'Pa'],[`${name} · ΔT`,q.temperatureChange,'K'],[`${name} · ΔV`,q.volumeChange,'m³'],[`${name} · W`,q.work,'J'],[`${name} · Q`,q.heat,'J'],[`${name} · ΔU`,q.energyChange,'J'],[`${name} · ΔS`,q.entropyChange,'J/K']);rows.push([t('최종 A−B의 일','Final A−B work'),result.gas.comparison.workDifference,'J']);}
  return <section ref={panel} className="ai-lab" id="experiment" aria-labelledby="ai-lab-title"><div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{t('직접 바꿔보세요','Make it your experiment')}</h2></div><span className="ai-live-dot">{t('실시간 계산','Live calculation')}</span></div>
    <GuidedExperiments kind={kind} ready={ready} onLoad={g=>load(spec.presets[g.preset].values)}/>
    <fieldset className="ai-presets" disabled={!ready}><legend>{t('비교할 예제','Examples to compare')}</legend>{spec.presets.map((p,i)=><Button key={i} variant="outline" size="sm" type="button" onClick={()=>load(p.values)}>{p.name[language]}</Button>)}</fieldset>
    <fieldset className="ai-parameters" disabled={!ready} onFocusCapture={()=>setPlaying(false)}><legend>{t('모델 설정','Model settings')}</legend><div className="ai-two-fields" key={formEpoch}>{spec.controls.map((c,i)=><NumberField key={i} label={c.label[language]} value={values[i]} min={c.min} max={c.max} step={c.step} onChange={n=>{setPlaying(false);set(v=>v.map((x,j)=>i===j?n:x));}}/>)}</div></fieldset>
    <p className="ai-small">{t('숫자는 Enter 또는 입력란을 떠날 때 적용합니다. 잘못된 입력은 이전 설정을 유지하며 초기화하면 입력 초안도 지웁니다.', 'Apply a number with Enter or by leaving its field. Invalid drafts retain the previous setting; reset also clears drafts.')}</p>
    <p className="ai-logic-statement">{spec.note[language]}</p>
    {result.thermal?.window===null&&<p role="status" className="ai-logic-statement">{t('구간 끝이 시작보다 이릅니다. 두 값을 수정하면 구간 수지를 계산합니다.','Window end precedes start. Edit the bounds to calculate its budget.')}</p>}
    <div className="ai-lab-actions"><Button type="button" variant="outline" size="sm" disabled={!ready||ended||!active} onClick={()=>setPlaying(p=>!p)}>{playing?t('일시정지','Pause'):t('재생','Play')}</Button><Button type="button" variant="outline" size="sm" disabled={!ready||ended} onClick={()=>{setPlaying(false);set(v=>v.map((x,i)=>i===clock?Math.min(limit,x+increment):x));}}>{t('한 단계 진행','Step forward')}</Button><Button type="button" variant="outline" size="sm" disabled={!ready} onClick={()=>load(initial)}>{t('모든 실험 설정 초기화','Reset all experiment settings')}</Button></div>
    {!motion.canAnimate&&<p className="ai-small">{t('모션 설정에 따라 자동 재생을 멈췄습니다. 수동 단계는 사용할 수 있습니다.','Motion preferences pause automatic playback. Manual steps remain available.')}</p>}
    <Figures {...result}/><Numbers rows={rows}/><p className="ai-small">{storage?t('설정을 이 브라우저에 저장합니다.','Settings are saved in this browser.'):t('브라우저 저장을 사용할 수 없습니다.','Browser storage is unavailable.')}</p>
  </section>;
}
