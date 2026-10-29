'use client';
import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { Button } from '@/components/ui/button';
import { formatNumber as n, type Vec2 } from '@/lib/learning-simulations';
import { neuronState, neuronStep, parameterNames } from '@/lib/learning-calculus';
import { calculusValues, calculusPresets, defaultCalculus, validCalculus, type CalculusSettings, type CalculusKind } from './calculus-model';
import { useSavedState } from './use-saved-state';
import { NumberField, Tuple } from './lab-controls';
import GuidedExperiments from './guided-experiments';
import CalculusGraph from './calculus-graph';

function ChangeChart({kind,s,ko}:{kind:CalculusKind;s:CalculusSettings;ko:boolean}) {
  const v=calculusValues(s); const c=v.chain; const j=v.jacobian;
  const points = kind==='chain-rule' ? Array.from({length:61},(_,i)=>{const dx=-.6+i*.02;return [dx,(s.a*(s.x+dx)+s.b)**2-c.output] as Vec2;}) : [[0,0],j.jvp,j.actualChange] as Vec2[];
  const tangent = kind==='chain-rule' ? [[-.6,c.derivative*-.6],[.6,c.derivative*.6]] as Vec2[] : [[0,0],j.jvp] as Vec2[];
  const all=[...points,...tangent];const maxX=Math.max(.01,...all.map(p=>Math.abs(p[0])))*1.15;const maxY=Math.max(.01,...all.map(p=>Math.abs(p[1])))*1.15;
  const xy=(p:Vec2)=>[150+p[0]/maxX*125,120-p[1]/maxY*95];
  const path=(p:Vec2[])=>p.map((v,i)=>`${i?'L':'M'}${xy(v)}`).join('');
  return <figure className="ai-regression-chart"><svg viewBox="0 0 300 260" role="img" aria-label={ko ? (kind==='chain-rule'?'입력 변화에 대한 실제 출력 변화 곡선과 현재 점의 접선.':'같은 출력 좌표축에서 비교한 실제 변화와 Jacobian의 일차 예측.'):(kind==='chain-rule'?'Actual output change versus input displacement and the tangent at the current point.':'Actual output displacement and the Jacobian prediction on the same output coordinate axes.')}>
    <path d="M20 120H280M150 15V225" stroke="var(--learning-neutral)" fill="none"/>
    <path d={path(tangent)} stroke="var(--learning-orange)" strokeWidth="3" strokeDasharray="5 4" fill="none"/>
    <path d={path(kind==='chain-rule'?points:[[0,0],j.actualChange])} stroke="var(--learning-blue)" strokeWidth="2.5" fill="none"/>
    {[kind==='chain-rule'?[s.delta,c.actualChange] as Vec2:j.actualChange,kind==='chain-rule'?[s.delta,c.linearChange] as Vec2:j.jvp].map((p,i)=><circle key={i} cx={xy(p)[0]} cy={xy(p)[1]} r="4" fill={`var(--learning-${i?'orange':'blue'})`}/>)}
    <text x="277" y="139" textAnchor="end" fill="var(--learning-neutral)" fontSize="12">{n(maxX)}</text><text x="157" y="22" fill="var(--learning-neutral)" fontSize="12">{n(maxY)}</text>
    <text x="150" y="247" textAnchor="middle" fill="var(--learning-neutral)" fontSize="12">{ko?(kind==='chain-rule'?'가로: δ · 세로: Δy':'가로: ΔF₁ · 세로: ΔF₂'):(kind==='chain-rule'?'horizontal: δ · vertical: Δy':'horizontal: ΔF₁ · vertical: ΔF₂')}</text>
  </svg><figcaption>{ko?'파랑: 실제 변화 · 주황 점선: 일차 예측. 축은 현재 값에 맞춰 자동 조절되며 양의 표시 한계를 숫자로 표시합니다.':'Blue: actual change · Orange dashed: linear prediction. Axes auto-scale to current values; positive display limits are labeled numerically.'}</figcaption></figure>;
}
export default function CalculusLab({kind}:{kind:CalculusKind}) {
  const {language}=useLanguage(); const ko=language==='ko'; const [s,set,ready,storage]=useSavedState(`suanlab-learning:${kind}:lab`,defaultCalculus,validCalculus);
  const [updateNote,note]=useState(''); const v=calculusValues(s); const c=v.chain;const j=v.jacobian;const b=v.neuron;
  const change=(patch:Partial<CalculusSettings>)=>{set(current=>({...current,...patch}));note('');};
  const preset=(index:number)=>{set({...defaultCalculus,...calculusPresets[kind][index]});note('');};
  const names=kind==='chain-rule'?(ko?['기준 예제','기울기 0','음의 기울기','상수 함수']:['Reference','Zero slope','Negative slope','Constant function']):kind==='jacobians'?(ko?['기준 예제','작은 이동','특이 Jacobian','큰 이동']:['Reference','Small displacement','Singular Jacobian','Large displacement']):(ko?['기준 예제','닫힌 ReLU','큰 학습률','ReLU 꼭짓점','tanh']:['Reference','Inactive ReLU','Large learning rate','ReLU kink','tanh']);
  const field=(label:string,key:'x'|'a'|'b'|'delta'|'epsilon'|'target'|'rate',min:number,max:number)=><NumberField label={label} value={s[key]} min={min} max={max} step="any" onChange={value=>change({[key]:value})}/>;
  const pair=(key:'point'|'direction'|'seed'|'input',label:string,bound:number)=><fieldset className="ai-vector-fields"><legend>{label}</legend><div className="ai-three-fields">{s[key].map((value,i)=><NumberField key={i} label={`${label}${i+1}`} value={value} min={-bound} max={bound} step="any" onChange={value=>change({[key]:s[key].map((x,j)=>j===i?value:x) as Vec2})}/>)}</div></fieldset>;
  const row=(label:string,value:number|number[])=><div key={label}><dt>{label}</dt><dd>{Array.isArray(value)?<Tuple value={value}/>:n(value)}</dd></div>;
  function update() {
    const next=neuronStep(s.input,s.parameters,s.target,s.activation,s.rate);const after=neuronState(s.input,next,s.target,s.activation);
    const candidate={...s,parameters:next,updates:s.updates+1};
    if(!validCalculus(candidate)){note(ko?`적용하지 않았습니다: 매개변수 ±100 또는 50회 제한을 넘습니다. 후보 ${parameterNames.map(k=>`${k}=${n(next[k])}`).join(', ')}, 후보 손실 ${n(after.loss)}. 학습률을 줄이거나 초기화하세요.`:`Update not applied: candidate exceeds parameter ±100 or the 50-update limit. Candidate ${parameterNames.map(k=>`${k}=${n(next[k])}`).join(', ')}, candidate loss ${n(after.loss)}. Reduce the rate or reset.`);return;}
    set(candidate);note(ko?`이전 손실 ${n(b.loss)} → 새 손실 ${n(after.loss)}. 같은 이전 상태의 기울기로 동시 갱신했습니다.`:`Previous loss ${n(b.loss)} → new loss ${n(after.loss)}. All parameters used gradients from the same old state.`);
  }
  return <section className="ai-lab" id="experiment" aria-labelledby="ai-lab-title">
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{ko?'직접 바꿔보세요':'Make it your experiment'}</h2></div><span className="ai-live-dot">{ko?'실시간 계산':'Live calculation'}</span></div>
    <GuidedExperiments kind={kind} ready={ready} onLoad={g=>preset(g.preset)}/>
    <fieldset disabled={!ready} className="ai-presets"><legend>{ko?'비교할 예제':'Examples to compare'}</legend>{names.map((name,i)=><Button key={name} variant="outline" size="sm" onClick={()=>preset(i)}>{name}</Button>)}</fieldset>
    <fieldset disabled={!ready} className="ai-calculus-stages"><legend>{ko?'계산 흐름 단계':'Computation stages'}</legend>{(ko?['입력','순전파','역방향 민감도','전체 보기']:['Input','Forward pass','Reverse sensitivities','Full graph']).map((name,i)=><Button key={i} size="sm" variant="outline" aria-pressed={s.stage===i} onClick={()=>change({stage:i})}>{i+1}. {name}</Button>)}</fieldset>
    <CalculusGraph kind={kind} stage={s.stage} settings={s} label={ko?'현재 설정의 계산 그래프. 파랑은 값의 순방향 흐름, 주황은 출력에서 입력으로 전달하는 민감도이며 아래 수치 표에서 모든 값을 확인할 수 있습니다.':'Computation graph at current settings. Blue shows forward values; orange shows sensitivities propagated from output to input. All values are available in the numeric readout below.'}/>
    <p className="ai-small">{ko?'작은 화면에서는 그림 안을 가로로 이동할 수 있습니다. 위치와 선 길이는 값의 크기를 나타내지 않습니다. 단계 버튼은 설명을 나누어 보여주며 아래 계산은 항상 전체 모형을 사용합니다.':'On narrow screens, scroll inside the diagram horizontally. Node positions and edge lengths do not encode magnitude. Stage buttons reveal the explanation; calculations below always use the complete model.'}</p>
    <fieldset disabled={!ready} className="ai-parameters"><legend>{ko?'모델 설정':'Model settings'}</legend>
      {kind==='chain-rule'&&<><p>u=ax+b · y=u²</p><div className="ai-three-fields">{field('x','x',-3,3)}{field('a','a',-3,3)}{field('b','b',-3,3)}</div>{field('δ','delta',-.5,.5)}</>}
      {kind==='jacobians'&&<><p>F(x,y)=(xy,x²+y)</p>{pair('point',ko?'기준점 p':'Point p',3)}{pair('direction',ko?'이동 d':'Displacement d',.5)}{pair('seed',ko?'출력 가중치 q':'Output weights q',3)}</>}
      {kind==='backpropagation'&&<>{pair('input',ko?'입력 x':'Input x',3)}<div className="ai-three-fields">{parameterNames.map(name=><NumberField key={name} label={name} value={s.parameters[name]} min={-100} max={100} step="any" onChange={value=>change({parameters:{...s.parameters,[name]:value},updates:0})}/>)}</div>{field(ko?'목표 t':'Target t','target',-3,3)}<label className="ai-activation-label">{ko?'활성화 함수':'Activation'}<select value={s.activation} onChange={e=>change({activation:e.target.value as 'relu'|'tanh',updates:0})}><option value="relu">ReLU</option><option value="tanh">tanh</option></select></label>{field(ko?'학습률 η':'Learning rate η','rate',.001,2)}<Button size="sm" onClick={update}>{ko?'한 번 갱신':'Apply one update'}</Button><p className="ai-small">{ko?'동시 갱신 횟수':'Simultaneous updates'}: {s.updates} / 50</p></>}
      {field(ko?'중앙차분 ε':'Central-difference ε','epsilon',.000001,.1)}
    </fieldset>
    {updateNote&&<p className="ai-model-note" role="status">{updateNote}</p>}
    <div className="ai-readout" data-testid="lab-readout"><h3>{ko?'수치로 확인하기':'Check the numbers'}</h3><p className="ai-small">{ko?'표시는 반올림하며 계산은 반올림 전 값을 사용합니다.':'Readouts are rounded; calculations use unrounded values.'}</p><dl>
      {kind==='chain-rule'&&<>{row('u',c.u)}{row('y',c.output)}{row('du/dx',c.localInner)}{row('dy/du',c.localOuter)}{row('dy/dx',c.derivative)}{row(ko?'실제 변화 Δy':'Actual change Δy',c.actualChange)}{row(ko?'일차 예측':'Linear prediction',c.linearChange)}{row(ko?'나머지':'Remainder',c.remainder)}{row(ko?'중앙차분':'Central difference',c.numeric)}{row(ko?'절대 차이':'Absolute difference',c.checkError)}</>}
      {kind==='jacobians'&&<>{row('F(p)',j.output)}{row('J row 1',j.matrix[0])}{row('J row 2',j.matrix[1])}{row('det J',j.determinant)}{row('Jd',j.jvp)}{row('Jᵀq',j.vjp)}{row(ko?'실제 변화 ΔF':'Actual change ΔF',j.actualChange)}{row(ko?'나머지':'Remainder',j.remainder)}{row('qᵀJd',j.dualLeft)}{row('(Jᵀq)ᵀd',j.dualRight)}{row(ko?'중앙차분 Jd':'Central-difference Jd',j.numeric)}</>}
      {kind==='backpropagation'&&<>{row('z',b.z)}{row('h',b.hidden)}{row('ŷ',b.prediction)}{row('L',b.loss)}{row('φ′(z)',b.activationSlope)}{row('∂L/∂h',b.adjointHidden)}{row('∂L/∂z',b.adjointZ)}{parameterNames.map(name=>row(`∂L/∂${name}`,b.gradients[name]))}{row('∇xL',b.inputGradient)}</>}
    </dl></div>
    {kind!=='backpropagation'?<ChangeChart kind={kind} s={s} ko={ko}/>:<><div className="ai-calculus-table" tabIndex={0} role="region" aria-label={ko?'기울기 검산 표':'Gradient check table'}><table><caption>{ko?'역전파와 중앙차분 비교':'Backpropagation versus central differences'}</caption><thead><tr>{(ko?['변수','역전파','중앙차분','절대 차이','매끄러운 구간']:['Variable','Backprop','Central difference','Absolute difference','Smooth interval']).map(x=><th scope="col" key={x}>{x}</th>)}</tr></thead><tbody>{v.check.rows.map(r=><tr key={r.name}><th scope="row">{r.name}</th><td>{n(r.analytic)}</td><td>{n(r.numeric)}</td><td>{n(r.absoluteError)}</td><td>{r.smooth?(ko?'예':'Yes'):(ko?'아니요':'No')}</td></tr>)}</tbody></table></div><p className="ai-model-note" role="status">{b.kink?(ko?'z=0에서 ReLU의 고전적 도함수는 없습니다. 이 실험은 기울기 0을 계산 규약으로 선택합니다. 중앙차분과의 일치를 기대하지 마세요.':'At z=0, ReLU has no classical derivative. This lab explicitly chooses slope 0 as a computational convention. Central differences need not agree.'):!v.check.smooth?(ko?'차분 구간이 ReLU 경계를 지납니다. ε를 줄이거나 매끄러운 점에서 검산하세요.':'A finite-difference interval crosses the ReLU boundary. Reduce ε or check at a smooth point.'):(ko?'매끄러운 구간에서 검산합니다. ε가 너무 크면 절단 오차, 너무 작으면 반올림 오차가 커집니다.':'Checking on smooth intervals. Large ε increases truncation error; very small ε increases roundoff error.')}</p></>}
    <p className="ai-model-note">{ko?(kind==='chain-rule'?'이 모형의 나머지는 정확히 (aδ)²입니다. 기울기가 0이어도 이차 변화는 남을 수 있습니다.':kind==='jacobians'?'Jacobian의 행은 출력, 열은 입력입니다. q는 출력의 가중치이며 입력 이동이 아닙니다. Jd는 국소 예측이므로 유한 이동의 실제 변화와 구분하세요.':'하나의 은닉 뉴런과 하나의 표본으로 구성한 교육 모형입니다. 목표와 입력은 고정하고 다섯 매개변수만 갱신합니다. 올바른 기울기도 큰 보폭의 손실 감소를 보장하지 않습니다.'):(kind==='chain-rule'?'The remainder in this model is exactly (aδ)². Zero slope can still leave quadratic change.':kind==='jacobians'?'Jacobian rows index outputs; columns index inputs. q weights outputs and is not an input displacement. Jd is a local prediction, distinct from the actual finite change.':'An educational model with one hidden neuron and one sample. Updates change only five parameters; inputs and target stay fixed. A correct gradient does not guarantee loss reduction for a large step.')}</p>
    <Button variant="outline" size="sm" className="ai-reset" onClick={()=>preset(0)}>{ko?'모든 실험 설정 초기화':'Reset all experiment settings'}</Button><p className="ai-small">{storage?(ko?'실험 설정은 이 브라우저에 자동 저장됩니다.':'Experiment settings are saved in this browser.'):(ko?'브라우저 저장소를 사용할 수 없어 현재 방문에서만 유지됩니다.':'Browser storage is unavailable. Settings remain in memory for this visit.')}</p>
  </section>;
}
