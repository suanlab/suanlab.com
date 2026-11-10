'use client';
import React,{useId,useState} from 'react';
import {Button} from '@/components/ui/button';
import {realTimePhaseExampleLabels} from '@/data/real-time-phase-teaching';
import {realTimePhaseExamples,realTimeChainDefault,realTimePacketDefault,realTimeFoldDefault,type RealTimePhaseInput} from '@/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey} from '@/lib/learning-real-time-phase-protocol';
import {realTimeDraft,parseRealTimeDraft} from '@/lib/learning-real-time-phase-state';
import {useSavedRealTimePhase} from './use-saved-real-time-phase';
import {useRealTimePhaseCalculation} from './use-real-time-phase-calculation';
import Concept from './real-time-phase-concept';
import Lessons from './real-time-phase-lessons';
import Wave from './real-time-phase-wave';
import Matrix from './real-time-phase-matrix';
export default function RealTimePhaseLab({language,embedded=false}:{language:'ko'|'en';embedded?:boolean}){
  const id=useId(),t=(ko:string,en:string)=>language==='ko'?ko:en,saved=useSavedRealTimePhase(),{model,ready}=saved;
  const[running,setRunning]=useState(true),[invalid,setInvalid]=useState(false),calculation=useRealTimePhaseCalculation(model.input,ready&&running);
  const apply=(input:RealTimePhaseInput)=>{if(!ready)return;saved.update(previous=>({...previous,input,draft:realTimeDraft(input)}));setInvalid(false);setRunning(true);calculation.retry();};
  const inputKey=realTimePhaseInputKey(model.input),example=realTimePhaseExamples.find(e=>realTimePhaseInputKey(e.input)===inputKey),draft=model.draft;
  const defaults={chain:realTimeChainDefault,packet:realTimePacketDefault,fold:realTimeFoldDefault};
  const names:Record<string,string>={intervals:t('구간 수 N','Interval count N'),time:t('총시간 τ','Total time τ'),mass:t('질량 m','Mass m'),hbar:'ℏ',omegaSquared:t('진동자 계수 ω²','Oscillator coefficient ω²'),start:t('시작 위치 a','Start position a'),end:t('마지막 위치 b','End position b'),damping:t('Abel 조절 ε','Abel regulator ε'),probe:t('관측 첨자','Observation index'),position:t('초기 중심 q₀','Initial center q₀'),momentum:t('초기 운동량 p₀','Initial momentum p₀'),chirp:t('chirp χ','Chirp χ'),width:draft.fields.widthMode==='hbar-scaled-variance'?t('폭 계수 r=σ²/ℏ','Width coefficient r=σ²/ℏ'):t('초기 위치 분산 σ²','Initial position variance σ²'),control:t('물리 제어값 u','Physical control u')};
  return <section id={embedded?undefined:'experiment'} className="ai-qubit-lab information-lab scalar-lab" lang={language} aria-labelledby={`${id}-title`} data-real-time-phase={calculation.status} data-real-time-storage-ready={ready?'true':'false'}>
    <h2 id={`${id}-title`}>{t('실시간 위상·파동묶음·고전극한 실험','Explore real-time phase, packets and classical limits')}</h2>
    <p>{t('실수 작용의 복소 위상, 정규화된 파동묶음, canonical fold를 비교합니다. 적용한 모형은 계산·독립 검산 후 표시하며, 고정 설명 그림과 현재 입력의 결과를 구별합니다.','Compare complex phases of real actions, normalized packets and the canonical fold. Applied models appear after calculation and independent verification; distinguish fixed teaching illustrations from results for the current input.')}</p>
    {!embedded&&<Concept language={language} selectedStep={model.step} onStep={step=>{if(ready)saved.update(p=>({...p,step}));}}/>}
    {saved.storage!=='available'&&<p role="status" data-real-time-storage={saved.storage}>{saved.storage==='corrupt'?t('저장된 입력이나 학습 선택을 읽을 수 없어 기본값으로 복원했습니다.','Saved input or learning choices were invalid; defaults were restored.'):t('이 브라우저에 저장할 수 없습니다. 현재 화면에서 실험을 계속할 수 있습니다.','This browser cannot save progress. You can continue experimenting in this session.')}</p>}
    <fieldset disabled={!ready} style={{border:0,padding:0}}><legend className="sr-only">{t('입력과 학습 활동','Model inputs and learning activities')}</legend>
    <Lessons language={language} state={model.lesson} onState={lesson=>saved.update(p=>({...p,lesson}))} onApply={apply}/>
    <label className="qubit-field">{t('전체 예제','All examples')}<select aria-label={t('전체 예제','All examples')} value={example?.id??'custom'} onChange={e=>{const found=realTimePhaseExamples.find(x=>x.id===e.target.value);if(found)apply(found.input);}}><option value="custom" disabled>{t('직접 입력 또는 안내의 추가 예제','Custom input or additional guide example')}</option>{realTimePhaseExamples.map(e=><option key={e.id} value={e.id}>{realTimePhaseExampleLabels[e.id][language]}</option>)}</select></label>
    <form onSubmit={e=>{e.preventDefault();const input=parseRealTimeDraft(draft);if(input)apply(input);else setInvalid(true);}}>
      <h3>{t('직접 입력','Edit parameters')}</h3>
      <label className="qubit-field">{t('편집할 모형','Model being edited')}<select aria-label={t('편집할 모형','Model being edited')} value={draft.mode} onChange={e=>{const mode=e.target.value as RealTimePhaseInput['mode'];saved.update(p=>({...p,draft:realTimeDraft(defaults[mode])}));setInvalid(false);}}><option value="chain">{t('고정끝 사슬','Pinned-end chain')}</option><option value="packet">{t('정규화된 파동묶음','Normalized packet')}</option><option value="fold">{t('canonical fold','Canonical fold')}</option></select></label>
      <p>{t('N은2…32, 관측 첨자는0…N의 정수입니다. 정수·분수의 분자와 분모는 각각12자리 이내이며 소수·지수·공백 표기는 받지 않습니다. 시간·질량·ℏ·폭은0보다 크고16이하, ε는0…16, 나머지 유리수의 절댓값은16이하입니다.','N is an integer from2 to32; observation index is0…N. Integer/fraction numerators and denominators have at most12digits; decimals, exponents and whitespace are rejected. Time, mass, hbar and width are positive and at most16; ε is0…16 and other rational magnitudes at most16.')}</p>
      <div className="scalar-inputs">{Object.entries(draft.fields).map(([key,value])=>key==='widthMode'?<label className="qubit-field" key={key}>{t('폭의 의미','Width interpretation')}<select aria-label={t('폭의 의미','Width interpretation')} value={value} onChange={e=>{const next=e.target.value;saved.update(p=>({...p,draft:{...p.draft,fields:{...p.draft.fields,widthMode:next}}}));}}><option value="fixed-variance">{t('고정 σ²','Fixed σ²')}</option><option value="hbar-scaled-variance">σ²=ℏr</option></select></label>:<label className="qubit-field" key={key}>{names[key]}<input aria-label={names[key]} value={value} maxLength={32} inputMode={['intervals','probe'].includes(key)?'numeric':'text'} autoComplete="off" spellCheck={false} onChange={e=>{const next=e.target.value;saved.update(p=>({...p,draft:{...p.draft,fields:{...p.draft.fields,[key]:next}}}));}}/></label>)}</div>
      <p>{t('편집은 결과를 바꾸지 않습니다. 입력 적용을 누르기 전까지 결과는 마지막으로 적용한 모형의 값입니다. 편집 중인 초안도 이 브라우저에 저장합니다.','Editing does not change results. Until Apply input is pressed, results belong to the last applied model. The in-progress draft is also saved in this browser.')}</p>
      {invalid&&<p role="alert" data-real-time-invalid>{t('입력 형식·부호·범위를 확인하세요. 결과는 이전에 적용한 입력의 값입니다.','Check input syntax, signs and bounds. Results belong to the previously applied input.')}</p>}
      <div className="qubit-actions"><Button type="submit">{t('입력 적용','Apply input')}</Button><Button type="button" variant="outline" onClick={()=>apply(realTimeChainDefault)}>{t('첫 예제로 초기화','Reset to first example')}</Button></div>
    </form></fieldset>
    <details><summary>{t('현재 적용한 모형과 입력','Currently applied model and input')}</summary><pre data-real-time-applied style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{JSON.stringify(model.input,null,2)}</pre></details>
    <div className="qubit-actions">{calculation.status==='pending'&&<Button variant="outline" onClick={()=>{setRunning(false);calculation.cancel();}}>{t('계산 취소','Cancel calculation')}</Button>}{(calculation.error||!running)&&<Button onClick={()=>{setRunning(true);calculation.retry();}}>{t('다시 계산','Recalculate')}</Button>}</div>
    <p role={calculation.error?'alert':'status'} aria-live="polite">{calculation.error?t('계산을 마치지 못했습니다. 다시 계산하거나 예제를 선택하세요.','Calculation could not finish. Recalculate or select an example.'):!running?t('계산을 취소했습니다.','Calculation cancelled.'):!ready?t('저장한 입력과 학습 선택을 복원하고 있습니다.','Restoring saved input and learning choices.'):calculation.status==='ready'?t('현재 입력의 계산과 독립 검산을 완료했습니다.','Calculation and independent checks for the current input are complete.'):calculation.phase==='validating'?t('결과를 독립 검산하고 있습니다.','Independently checking the result.'):t('현재 입력을 계산하고 있습니다.','Calculating the current input.')}</p>
    {calculation.response&&<><Wave response={calculation.response} language={language} display={model.wave} onDisplay={wave=>saved.update(p=>({...p,wave}))}/><Matrix response={calculation.response} language={language} preferences={model.matrix} onPreferences={matrix=>saved.update(p=>({...p,matrix}))}/></>}
  </section>;
}
