'use client';
import { isProbabilityKind, evaluateProbability, type ProbabilityKind } from '@/lib/learning-probability';

import { isAbstractAlgebraKind, evaluateAbstractAlgebra, type AbstractAlgebraKind } from '@/lib/learning-abstract-algebra';
import { useId, useMemo } from 'react';
import { useLanguage } from '@/components/language-provider';
import { Button } from '@/components/ui/button';
import { probabilityLabs, abstractAlgebraLabs, tensorLabs, complexLabs, analysisLabs, matrixStructureLabs, discreteLabs, elementaryLabs, foundationLabs, learningUnits } from '@/data/interactive-book';
import { evaluateFoundation, type FoundationKind, type FoundationResult } from '@/lib/learning-foundations';
import { evaluateElementary, isElementaryKind, type ElementaryKind } from '@/lib/learning-elementary';
import { formatNumber } from '@/lib/learning-simulations';
import { useSavedState } from './use-saved-state';
import { NumberField } from './lab-controls';
import GuidedExperiments from './guided-experiments';
import { evaluateDiscrete, isDiscreteKind, type DiscreteKind } from '@/lib/learning-discrete';
import DiscreteFigure from './discrete-figure';
import { evaluateMatrixStructure, isMatrixStructureKind, type MatrixStructureKind } from '@/lib/learning-matrix-structures';
import { evaluateAnalysis, isAnalysisKind, type AnalysisKind } from '@/lib/learning-analysis';
import { evaluateComplex, isComplexKind, type ComplexKind } from '@/lib/learning-complex';
import { evaluateTensor, isTensorKind, type TensorKind } from '@/lib/learning-tensors';
import TensorFigure from './tensor-figure';
import ProbabilityFigure from './probability-figure';
type PlotKind=FoundationKind|ElementaryKind|DiscreteKind|MatrixStructureKind|AnalysisKind|ComplexKind|TensorKind|AbstractAlgebraKind|ProbabilityKind;
const specFor=(kind:PlotKind)=>isProbabilityKind(kind)?probabilityLabs[kind]:isAbstractAlgebraKind(kind)?abstractAlgebraLabs[kind]:isTensorKind(kind)?tensorLabs[kind]:isComplexKind(kind)?complexLabs[kind]:isAnalysisKind(kind)?analysisLabs[kind]:isMatrixStructureKind(kind)?matrixStructureLabs[kind]:isDiscreteKind(kind)?discreteLabs[kind]:isElementaryKind(kind)?elementaryLabs[kind]:foundationLabs[kind];
const modelFor=(kind:PlotKind,values:number[])=>isProbabilityKind(kind)?evaluateProbability(kind,values):isAbstractAlgebraKind(kind)?evaluateAbstractAlgebra(kind,values):isTensorKind(kind)?evaluateTensor(kind,values):isComplexKind(kind)?evaluateComplex(kind,values):isAnalysisKind(kind)?evaluateAnalysis(kind,values):isMatrixStructureKind(kind)?evaluateMatrixStructure(kind,values):isDiscreteKind(kind)?evaluateDiscrete(kind,values):isElementaryKind(kind)?evaluateElementary(kind,values):evaluateFoundation(kind,values);
const n=(value:number)=>Number.isFinite(value)?formatNumber(value):value===Infinity?'∞':value===-Infinity?'−∞':'—';
export function FoundationPlot({kind,result,label}:{kind:PlotKind;result:FoundationResult;label:string}){
 const {language}=useLanguage();const arrow=useId().replace(/:/g,'');const ko=language==='ko';const spec=specFor(kind);const points=result.curves.flat();
 const xs=points.map(v=>v[0]),ys=points.map(v=>v[1]);const geometry=['svd','real-complex-numbers','trigonometry-coordinates','vector-spaces-rank','determinants-traces','implicit-inverse-functions','vector-fields-divergence-curl','line-surface-volume-integrals','gauss-stokes-theorems','complex-polar-maps','cauchy-riemann-holomorphic','complex-contour-integrals','residues-winding','analytic-continuation-branches','complex-linear-algebra','wirtinger-gradients'].includes(kind);const radius=Math.max(.01,...xs.map(Math.abs),...ys.map(Math.abs));const minX=geometry?-radius:Math.min(0,...xs),maxX=geometry?radius:Math.max(.01,...xs),minY=geometry?-radius:Math.min(0,...ys),maxY=geometry?radius:Math.max(.01,...ys);const dx=maxX-minX,dy=maxY-minY;
 const xy=([x,y]:number[])=>[45+260*(x-minX)/dx,(geometry?300:220)-(geometry?260:170)*(y-minY)/dy];const path=(curve:number[][])=>curve.map((v,i)=>`${i?'L':'M'}${xy(v)}`).join('');
 if(isProbabilityKind(kind))return <ProbabilityFigure result={result} label={label} axis={spec.axis[language]}/>;
 if(isAbstractAlgebraKind(kind))return <DiscreteFigure result={result} label={label} axis={spec.axis[language]} multicolor/>;
 if(isTensorKind(kind))return <TensorFigure result={result} label={label} axis={spec.axis[language]} legend={spec.legend}/>;
 if(isDiscreteKind(kind))return <DiscreteFigure result={result} label={label} axis={spec.axis[language]}/>;
 const marker=(curve:number[][],i:number)=>{
  if(curve.length<2)return undefined;const first=curve[0],last=curve[curve.length-1];
  if(first[0]===last[0]&&first[1]===last[1])return undefined;
  return kind==='vector-fields-divergence-curl'?`url(#${arrow}-0)`:(kind==='line-surface-volume-integrals'||kind==='complex-contour-integrals')&&i>0?`url(#${arrow}-1)`:undefined;
 };
 const bars=kind==='probability-distributions'||kind==='expectation-variance';
 return <figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'실험 그림 · 작은 화면에서 가로로 이동':'Experiment plot · Scroll horizontally on narrow screens'}><svg data-equal-scale={geometry?'true':undefined} viewBox={geometry?'0 0 350 360':'0 0 350 280'} role="img" aria-label={label}>
 <defs>{['blue','orange'].map((color,i)=><marker key={color} id={`${arrow}-${i}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill={`var(--learning-${color})`}/></marker>)}</defs>
 <path d={`M35 ${xy([0,0])[1]}H315M${xy([0,0])[0]} 35V${geometry?310:230}`} stroke="var(--learning-neutral)" fill="none"/>
 {result.curves.map((curve,i)=><g key={i}>{bars&&i===0?curve.map(([x,y],j)=><path key={j} d={`M${xy([x,0])}L${xy([x,y])}`} stroke="var(--learning-blue)" strokeWidth={Math.min(18,130/curve.length)}/>):<path markerEnd={marker(curve,i)} d={path(curve)} fill="none" stroke={`var(--learning-${i&&kind!=='vector-fields-divergence-curl'?'orange':'blue'})`} strokeWidth="2.5" strokeDasharray={i&&kind!=='vector-fields-divergence-curl'?'5 3':undefined}/>}{curve.length<=6&&curve.map((v,j)=><circle key={j} cx={xy(v)[0]} cy={xy(v)[1]} r="3.5" fill={`var(--learning-${i&&kind!=='vector-fields-divergence-curl'?'orange':'blue'})`}/>)}</g>)}
 <g fill="var(--learning-neutral)" fontSize="12"><text x="45" y={geometry?325:245}>{n(minX)}</text><text x="305" y={geometry?325:245} textAnchor="end">{n(maxX)}</text><text x="45" y="28">{n(maxY)}</text><text x="45" y={geometry?345:265}>{geometry?(ko?'가로·세로의 단위 길이가 같습니다.':'Equal unit length on both axes.'):(ko?'축 범위는 현재 값에 맞춰 조절합니다.':'Axis ranges adapt to current values.')}</text></g>
 {result.curves.length===0&&<text x="175" y="140" textAnchor="middle" fill="var(--learning-neutral)" fontSize="13">{ko?'이 설정에서 그릴 값이 없습니다.':'No plotted value at this setting.'}</text>}
 {result.holes?.map((point,i)=><circle key={i} cx={xy(point)[0]} cy={xy(point)[1]} r="5" fill="hsl(var(--card))" stroke="var(--learning-orange)" strokeWidth="2" data-excluded-point="true"/>)}
 </svg></div><figcaption>{spec.axis[language]}</figcaption><div className="ai-legend">{result.curveLabels?result.curveLabels.map((name,i)=><span key={i} className={i?'ai-orange':'ai-blue'}>{name}</span>):spec.legend.slice(0,result.curves.length).map((name,i)=><span key={i} className={i?'ai-orange':'ai-blue'}>{name[language]}</span>)}</div>{kind==='kronecker-products'&&<DiscreteFigure result={{...result,curves:[]}} label={label} axis={ko?'표: A⊗B의 실제 원소':'Table: actual entries of A⊗B'}/>}</figure>;
}
export function FoundationConcept({kind,label,step}:{kind:PlotKind;label:string;step:number}){
 const spec=specFor(kind);const result=modelFor(kind,spec.controls.map(c=>c.initial));const entries=Object.entries(result.stats);const entry=entries[Math.min(step,entries.length-1)];
 return <div>{(isComplexKind(kind)||isProbabilityKind(kind)||isAbstractAlgebraKind(kind)||isTensorKind(kind))&&result.statement&&<p className="ai-logic-statement">{result.statement}</p>}<FoundationPlot kind={kind} result={result} label={label}/><p className="ai-concept-equation">{entry[0]} = {entry[1]===null?'—':Array.isArray(entry[1])?(entry[1].length?`(${entry[1].map(n).join(', ')})`:'∅'):n(entry[1])}</p></div>;
}
export default function FoundationsLab({kind}:{kind:PlotKind}){
 const {language}=useLanguage();const ko=language==='ko';const spec=specFor(kind);
 const initial=useMemo(()=>({values:spec.controls.map(c=>c.initial)}),[spec]);
 const valid=useMemo(()=>(v:unknown):v is {values:number[]}=>{if(!v||typeof v!=='object')return false;const arr=(v as {values:unknown}).values;return Array.isArray(arr)&&arr.length===spec.controls.length&&arr.every((x,i)=>Number.isFinite(x)&&x>=spec.controls[i].min&&x<=spec.controls[i].max&&(spec.controls[i].step==='any'||Number.isInteger(x)));},[spec]);
 const [s,set,ready,storage]=useSavedState(`suanlab-learning:${kind}:lab`,initial,valid);const result=modelFor(kind,s.values);
 const load=(index:number)=>set({values:[...spec.presets[index].values]});
 const warnings={'numeric-range':ko?'수학적 값이 현재 숫자 표현 범위를 벗어났습니다. 이를 무한값이나 정의역 오류로 해석하지 마세요.':'The mathematical value is outside the current numeric representation range. This is not mathematical infinity or a domain failure.','zero-divisor':ko?'0으로 나누는 값은 정의하지 않습니다. 분모·시간·역함수의 배율을 확인하세요.':'Division by zero is undefined. Check the denominator, time or inverse-map scale.','undefined-angle':ko?'영점에는 고유한 방향각이 없습니다. 크기 0과 정의되지 않는 각을 구분하세요.':'Zero has no unique direction angle. Distinguish zero magnitude from an undefined angle.','degenerate-equation':ko?'일차 계수가 0입니다. 0x=0이면 모든 실수, 0x=b≠0이면 해가 없습니다.':'The linear coefficient is zero. 0x=0 has all real solutions; 0x=b≠0 has none.','function-domain':ko?'선택 입력이나 밑이 이 함수의 정의 조건을 만족하지 않습니다. 주변 극한을 함수값으로 바꾸지 마세요.':'The selected input or base violates this function’s domain conditions. A nearby limit is not a value at the excluded point.','infinite-sum':ko?'이 설정에는 유한한 보통 의미의 무한 합이 없습니다. 부분합과 급수의 합을 구분하세요.':'There is no finite ordinary infinite sum at this setting. Distinguish a partial sum from a series sum.','asymptotic-bound':ko?'선택한 C로는 유한 시작점이 없습니다. 이 후보의 실패가 Big-O 분류 자체의 실패는 아닙니다.':'No finite threshold works for the chosen C. Failure of this candidate does not disprove the Big-O class.',support:ko?'모형이 관측 가능한 사건에 확률 0을 줍니다. 무한 로그 비용을 유한한 값으로 숨기지 않습니다.':'The model assigns zero probability to a possible observation. Infinite log cost is not replaced by a finite value.',rank:ko?'특이하거나 식별되지 않는 방향이 있습니다. 표시된 최소 노름 선택과 데이터의 유일한 결론을 구분하세요.':'A direction is singular or unidentified. Distinguish the displayed minimum-norm choice from a uniquely data-determined conclusion.',stationary:ko?'전이가 없어 모든 분포가 정상입니다. 유일한 정상값은 정의되지 않습니다.':'With no transitions every distribution is stationary. No unique stationary occupancy is defined.',stability:ko?'안정성 또는 수렴 조건의 경계·바깥입니다. 실제 연속 모형의 성질과 수치 경로를 구분하세요.':'This setting lies on or outside a stability or convergence condition. Distinguish the continuous model from its numerical trajectory.',uncertainty:ko?'가정에 따른 구간이 0을 포함합니다. 이는 효과 없음의 증명도, 평가 설계가 타당하다는 보장도 아닙니다.':'The assumption-dependent interval includes zero. This neither proves no effect nor validates the evaluation design.'};
 const title=learningUnits.find(u=>u.id===kind)!.title[language];
 return <section className="ai-lab" id="experiment" aria-labelledby="ai-lab-title"><div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{ko?'직접 바꿔보세요':'Make it your experiment'}</h2></div><span className="ai-live-dot">{ko?'실시간 계산':'Live calculation'}</span></div>
 <GuidedExperiments kind={kind} ready={ready} onLoad={g=>load(g.preset)}/><fieldset disabled={!ready} className="ai-presets"><legend>{ko?'비교할 예제':'Examples to compare'}</legend>{spec.presets.map((p,i)=><Button key={i} variant="outline" size="sm" onClick={()=>load(i)}>{p.name[language]}</Button>)}</fieldset>
 {result.statement&&<p className="ai-logic-statement" aria-live="polite">{result.statement}</p>}<FoundationPlot kind={kind} result={result} label={`${title}. ${spec.axis[language]}. ${spec.legend.map(t=>t[language]).join('; ')}`}/>
 <fieldset disabled={!ready} className="ai-parameters"><legend>{ko?'모델 설정':'Model settings'}</legend><div className="ai-three-fields">{spec.controls.map((control,i)=><NumberField key={i} label={control.label[language]} min={control.min} max={control.max} step={control.step} value={s.values[i]} onChange={x=>{if(control.step!=='any'&&!Number.isInteger(x))return;set(v=>({values:v.values.map((y,j)=>j===i?x:y)}));}}/>)}</div></fieldset>
 <div className="ai-readout" data-testid="lab-readout"><h3>{ko?'수치로 확인하기':'Check the numbers'}</h3><p className="ai-small">{ko?'표시값은 반올림합니다. —는 정의되지 않거나 표시할 유한값이 없는 경우입니다.':'Readouts are rounded. — means undefined or no finite display value.'}</p><dl>{Object.entries(result.stats).map(([key,value])=><div key={key}><dt>{key}</dt><dd>{value===null?'—':Array.isArray(value)?(value.length?`(${value.map(n).join(', ')})`:'∅'):n(value)}</dd></div>)}</dl></div>
 {result.warning&&<p className="ai-model-note" role="status">{warnings[result.warning]}</p>}<p className="ai-model-note">{spec.note[language]}</p>
 <Button className="ai-reset" size="sm" variant="outline" onClick={()=>set(initial)}>{ko?'모든 실험 설정 초기화':'Reset all experiment settings'}</Button><p className="ai-small">{storage?(ko?'실험 설정은 이 브라우저에 자동 저장됩니다.':'Experiment settings are saved in this browser.'):(ko?'브라우저 저장소를 사용할 수 없어 현재 방문에서만 유지됩니다.':'Browser storage is unavailable. Settings remain in memory for this visit.')}</p>
 </section>;
}
