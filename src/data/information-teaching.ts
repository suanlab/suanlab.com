import type {LearningUnit,LearningUnitId,LearningGuide} from './interactive-book';
import {informationTeachingIds,informationTeachingKinds,type InformationTeachingId} from '@/lib/learning-information-teaching';
import {discriminationSteps as steps0,discriminationExperiments as experiments0,discriminationQuestions as questions0} from './information-discrimination-teaching';
import {discriminationExamples as examples0} from '@/lib/learning-information-discrimination-input';
import {informationEntropySteps as steps1,informationEntropyExperiments as experiments1,informationEntropyQuestions as questions1} from './information-entropy-teaching';
import {informationEntropyExamples as examples1} from '@/lib/learning-information-entropy-input';
import {ensembleSteps as steps2,ensembleExperiments as experiments2,ensembleQuestions as questions2} from './information-ensemble-teaching';
import {ensembleExamples as examples2} from '@/lib/learning-information-ensemble-input';
import {relativeSteps as steps3,relativeExperiments as experiments3,relativeQuestions as questions3} from './information-relative-teaching';
import {relativeExamples as examples3} from '@/lib/learning-information-relative-input';
import {bb84Steps as steps4,bb84Experiments as experiments4,bb84Questions as questions4} from './information-bb84-teaching';
import {bb84Examples as examples4} from '@/lib/learning-information-bb84-input';
import {teleportationSteps as steps5,teleportationExperiments as experiments5,teleportationQuestions as questions5} from './information-teleportation-teaching';
import {teleportationExamples as examples5} from '@/lib/learning-information-teleportation-input';
import {errorCorrectionSteps as steps6,errorCorrectionExperiments as experiments6,errorCorrectionQuestions as questions6} from './information-error-correction-teaching';
import {errorCorrectionExamples as examples6} from '@/lib/learning-information-error-correction-input';
import {majorizationSteps as steps7,majorizationExperiments as experiments7,majorizationQuestions as questions7} from './information-majorization-teaching';
import {majorizationExamples as examples7} from '@/lib/learning-information-majorization-input';
import {capacitySteps as steps8,capacityExperiments as experiments8,capacityQuestions as questions8} from './information-capacity-teaching';
import {capacityExamples as examples8} from '@/lib/learning-information-capacity-input';
type DraftUnit=Omit<LearningUnit,'id'|'prerequisiteIds'>&{id:InformationTeachingId;prerequisiteIds:(LearningUnitId|InformationTeachingId)[]};
const text=(ko:string,en:string)=>({ko,en});
const configs=[
 [
  "이진 양자 상태 판별",
  "Binary Quantum State Discrimination",
  "사전확률·측정 효과·성공률을 구분해 한 번의 최적 판단을 이해합니다.",
  "Understand one-shot optimal decisions through priors, measurement effects and success probabilities.",
  [
   "qubit-density-spectrum-entropy-dephasing",
   "qubit-povm-instrument-backaction"
  ],
  "Psuccess=(1+‖pρ₀−(1−p)ρ₁‖₁)/2",
  "QUANTUM INFORMATION · STATE DISCRIMINATION",
  "비직교 순수 상태 쌍과 최적 측정축의 기준 예제입니다.",
  "Reference example of a nonorthogonal pure-state pair and its optimal measurement axis."
 ],
 [
  "양자 엔트로피·압축·조건부 정보",
  "Quantum Entropy, Compression and Conditional Information",
  "부분계와 결합 정보, 유한 블록의 보존 공간과 점근 압축률을 비교합니다.",
  "Compare marginal/joint information, finite-block retained subspaces and asymptotic compression rates.",
  [
   "bipartite-pure-schmidt-partial-trace",
   "qubit-density-spectrum-entropy-dephasing"
  ],
  "S(ρ)=−Trρlog₂ρ; S(B|A)=S(AB)−S(A)",
  "QUANTUM INFORMATION · ENTROPY AND COMPRESSION",
  "같은 부분계와 다른 결합 엔트로피를 갖는 세 기준 상태입니다.",
  "Three reference states share marginal entropies but differ in joint entropy."
 ],
 [
  "고전–양자 앙상블과 Holevo 정보",
  "Classical–Quantum Ensembles and Holevo Information",
  "레이블·측정 기록·접근 가능 정보의 상계를 서로 구분합니다.",
  "Distinguish labels, measurement records and the upper bound on accessible information.",
  [
   "binary-qubit-state-discrimination",
   "quantum-entropy-compression-conditional-information"
  ],
  "Iselected≤Iacc≤χ=S(ρ̄)−ΣpₓS(ρₓ)",
  "QUANTUM INFORMATION · ENSEMBLES AND HOLEVO",
  "준비 레이블·양자 상태·측정 기록의 흐름과 선택 정보/상계의 기준 예제입니다.",
  "Reference preparation–state–record flow distinguishes selected information from its bound."
 ],
 [
  "양자 상대 엔트로피와 정보처리",
  "Quantum Relative Entropy and Data Processing",
  "지지집합·스펙트럼 로그·같은 채널 전후의 비교를 정확히 읽습니다.",
  "Read supports, spectral logarithms and before/after comparisons under the same channel.",
  [
   "quantum-entropy-compression-conditional-information",
   "qubit-channel-kraus-choi-verification"
  ],
  "D(Φρ∥Φσ)≤D(ρ∥σ)",
  "QUANTUM INFORMATION · RELATIVE ENTROPY",
  "비가환 상태에 같은 Z 탈위상을 적용하는 기준 예제와 지지집합 조건입니다.",
  "Reference example applies the same Z dephasing to noncommuting states and distinguishes support conditions."
 ],
 [
  "복제 불가·BB84의 정보와 교란",
  "No-Cloning, BB84 Information and Disturbance",
  "CNOT 후보·내적 장애·기저 선별·도청 정보와 표본 오류를 비교합니다.",
  "Compare a CNOT candidate, overlap obstruction, basis sifting, Eve information and sample errors.",
  [
   "binary-qubit-state-discrimination",
   "finite-qubit-circuit-order-interference",
   "qubit-povm-instrument-backaction"
  ],
  "U(ψ⊗0)≠ψ⊗ψ; QBER=f/4",
  "QUANTUM INFORMATION · NO-CLONING AND BB84",
  "CNOT의 중첩 입력과 이상적 전면 가로채기 BB84의 기준 예제입니다.",
  "Reference examples show a CNOT superposition input and ideal full-intercept BB84."
 ],
 [
  "텔레포테이션·초밀집 부호화의 자원",
  "Teleportation, Dense Coding and Their Resources",
  "네 측정 가지·보정·공유 얽힘과 실제 통신을 분리해 추적합니다.",
  "Track four measurement branches, corrections, shared entanglement and actual communication.",
  [
   "finite-qubit-circuit-order-interference",
   "bipartite-pure-schmidt-partial-trace",
   "quantum-instruments-records-feedforward"
  ],
  "1 ebit+2 cbits → qubit state; 1 ebit+1 qubit →2 cbits",
  "QUANTUM INFORMATION · COMMUNICATION RESOURCES",
  "텔레포테이션의 세 wire와 초밀집 부호화의 두 wire 기준 회로입니다.",
  "Reference diagrams use three teleportation wires and two dense-coding wires."
 ],
 [
  "양자 오류 정정과 Knill–Laflamme 조건",
  "Quantum Error Correction and Knill–Laflamme Conditions",
  "syndrome·고정 복구·논리 채널·오류 가족 전체의 정정 조건을 비교합니다.",
  "Compare syndromes, fixed recovery, logical channels and correctability of entire error families.",
  [
   "finite-qubit-circuit-order-interference",
   "qubit-channel-kraus-choi-verification",
   "quantum-instruments-records-feedforward"
  ],
  "V†Eₐ†EᵦV=cₐᵦI; pL=3p²−2p³",
  "QUANTUM INFORMATION · ERROR CORRECTION",
  "반복 부호 단일 X·IID X 채널·오류 교차 항을 구분한 기준 예제입니다.",
  "Reference examples distinguish a single X, IID X channel and error cross terms."
 ],
 [
  "주요화와 순수 상태 얽힘 자원",
  "Majorization and Pure-State Entanglement Resources",
  "누적합의 부분 순서·최적 성공 확률·실제 Bell 희석 측정을 비교합니다.",
  "Compare prefix partial orders, optimum success probabilities and explicit Bell dilution.",
  [
   "bipartite-pure-schmidt-partial-trace",
   "quantum-entropy-compression-conditional-information"
  ],
  "ψ→φ ⇔ λψ≺λφ; pmax=min Eℓ(λψ)/Eℓ(λφ)",
  "QUANTUM INFORMATION · MAJORIZATION AND LOCC",
  "교차하는 세 확률 누적합과 a=4/5,b=3/5의 Bell 희석 기준 예제입니다.",
  "Reference examples show crossing three-weight prefixes and Bell dilution with a=4/5,b=3/5."
 ],
 [
  "양자 채널 용량과 부호화 한계",
  "Quantum Channel Capacities and Coding Limits",
  "통신 과제·자원·입력별 정보와 채널 가족의 점근 용량을 구분합니다.",
  "Distinguish communication tasks, resources, selected-input information and family-specific asymptotic capacities.",
  [
   "classical-quantum-ensembles-holevo-information",
   "quantum-relative-entropy-data-processing",
   "system-environment-stinespring-reference",
   "teleportation-superdense-coding-resources"
  ],
  "C=1; CE=2−h₂(p); Q=1−h₂(p)",
  "QUANTUM INFORMATION · CAPACITY AND CODING",
  "p=1/10과 끝점의 용량·선택 입력을 구분하는 탈위상 기준 예제입니다.",
  "Reference dephasing examples distinguish p=1/10 capacities, endpoints and selected inputs."
 ]
]as const;
const sources=[{steps:steps0,experiments:experiments0,questions:questions0,examples:examples0},{steps:steps1,experiments:experiments1,questions:questions1,examples:examples1},{steps:steps2,experiments:experiments2,questions:questions2,examples:examples2},{steps:steps3,experiments:experiments3,questions:questions3,examples:examples3},{steps:steps4,experiments:experiments4,questions:questions4,examples:examples4},{steps:steps5,experiments:experiments5,questions:questions5,examples:examples5},{steps:steps6,experiments:experiments6,questions:questions6,examples:examples6},{steps:steps7,experiments:experiments7,questions:questions7,examples:examples7},{steps:steps8,experiments:experiments8,questions:questions8,examples:examples8}];
/** Registered only by the isolated candidate until its gates pass. */
export const informationTeaching:{unit:DraftUnit;guide:LearningGuide}[]=informationTeachingIds.map((id,index)=>{
 const [ko,en,subtitleKo,subtitleEn,prerequisiteIds,formula,eyebrow,diagramKo,diagramEn]=configs[index],source=sources[index];
 const unit:DraftUnit={id,title:text(ko,en),subtitle:text(subtitleKo,subtitleEn),prerequisiteIds:[...prerequisiteIds],prerequisites:text('선수 단원의 상태·측정·채널 개념을 복습한 뒤 아래 예제에서 가정과 단위를 확인하세요.','Review states, measurements and channels in the prerequisite units, then check assumptions and units in the examples.'),eyebrow,formula,accent:`information-${informationTeachingKinds[id]}`,questions:source.questions.map((q,i)=>({id:`information-${informationTeachingKinds[id]}-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))};
 const guide:LearningGuide={intuition:unit.subtitle,steps:source.steps.map(step=>({title:step.title,explanation:step.body,formula:step.formula,diagram:text(diagramKo,diagramEn)})),experiments:source.experiments.map(g=>{
  const labels=g.examples.map(key=>source.examples.find(x=>x.id===key)!.label),preset=source.examples.findIndex(x=>x.id===g.examples[0]);
  return {title:g.title,predict:g.predict,action:text(`실험에서 ${labels.map(l=>l.ko).join(' → ')} 예제를 순서대로 고르고 결과·정확 값·정의 조건을 비교하세요.`,`In the experiment, choose ${labels.map(l=>l.en).join(' → ')} in order; compare results, exact values and defining conditions.`),explain:g.explanation,preset};
 })};return {unit,guide};
});
