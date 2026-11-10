/** Draft teaching examples; reader-safe and intentionally absent from the public registry. */
import {snapshotQuantumInformationInput,type QuantumInformationInput,type InformationState}from'./learning-quantum-information-input';
import type{QuantumInformationDraftId}from'./learning-quantum-information-topics';
export type QuantumInformationPreset=Readonly<{id:string;label:Readonly<{ko:string;en:string}>;input:QuantumInformationInput}>;
const s=(real:string,imaginary='0',rootReal='0')=>({real:[real,rootReal],imaginary:[imaginary,'0']}as const);
const zero:InformationState=[s('1'),s('0')],plus:InformationState=[s('1'),s('1')],complex:InformationState=[s('3/5'),s('0','4/5')];
const preset=(id:string,ko:string,en:string,input:QuantumInformationInput):QuantumInformationPreset=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput(input)});
const discrimination=(bloch0:readonly[string,string,string],bloch1:readonly[string,string,string],prior='1/2')=>({kind:'discrimination' as const,bloch0,bloch1,prior});
const entropy=(correlationWeight:string,coherence:string,retainedRank='3')=>({kind:'entropy' as const,correlationWeight,coherence,probabilityOne:'1/5',copies:2,retainedRank});
const ensemble=(measurement:'helstrom'|'z',prior='1/2',biased=false)=>({kind:'ensemble' as const,bloch0:biased?['0','0','1/5']as const:['0','0','1']as const,bloch1:biased?['0','0','-1/5']as const:['24/25','0','-7/25']as const,prior,measurement});
const attack=(attackFraction:string,eveZProbability='1/2',sampleSize=20)=>({kind:'bb84' as const,input:plus,comparison:zero,attackFraction,eveZProbability,sampleSize});
const teleport=(input:InformationState,a=1,b=1,order:'x-then-z'|'z-then-x'='x-then-z')=>({kind:'teleportation' as const,input,a,b,order});
const correction=(selectedError:string,input:InformationState=complex,errors:readonly string[]=['III','XII','IXI','IIX'])=>({kind:'error-correction' as const,input,selectedError,physicalXProbability:'1/10',errors});
const resources=(source:readonly string[],target:readonly string[])=>({kind:'majorization' as const,mode:'compare' as const,source,target});
const capacity=(probability:string,input:readonly[string,string,string]=['0','0','0'])=>({kind:'capacity' as const,probability,input});
const entries:Record<QuantumInformationDraftId,readonly QuantumInformationPreset[]>={
 'binary-qubit-state-discrimination':[
  preset('rational-pair','비직교 순수 상태 · 최적 측정','Nonorthogonal pure states · optimal measurement',discrimination(['0','0','1'],['24/25','0','-7/25'])),
  preset('biased-decision','편향된 사전확률 · 항상 같은 답','Biased prior · constant decision',discrimination(['0','0','1/5'],['0','0','-1/5'],'9/10')),
  preset('nonunique-threshold','영 고유값 · 비유일한 측정','Zero eigenvalue · nonunique measurement',discrimination(['0','0','1/5'],['0','0','-1/5'],'3/5')),
  preset('identical','같은 두 상태 · 정보 없음','Identical states · no distinguishing information',discrimination(['0','0','0'],['0','0','0'])),
  preset('tiny-gain','표시 반올림에 가려진 양의 이득','Positive gain hidden by display rounding',discrimination(['0','0','1'],['0','0','999999999999999999999999999999/1000000000000000000000000000000'])),
 ],
 'quantum-entropy-compression-conditional-information':[
  preset('bell','Bell 쌍 · 음의 조건부 엔트로피','Bell pair · negative conditional entropy',entropy('1','1')),
  preset('classical-correlation','고전 상관 · 같은 부분계','Classical correlation · same marginals',entropy('1','0')),
  preset('independent','독립 혼합 · 상관 없음','Independent mixture · no correlation',entropy('0','0')),
  preset('positive-conditional-entangled','양의 조건부 엔트로피와 얽힘','Entanglement with positive conditional entropy',entropy('1/2','1')),
  preset('empty-memory','rank0 · 보존 공간 없음','Rank0 · no retained subspace',entropy('1','1','0')),
 ],
 'classical-quantum-ensembles-holevo-information':[
  preset('helstrom-table','최소 오류 측정의 정보량','Information from the minimum-error measurement',ensemble('helstrom')),
  preset('z-table','같은 앙상블 · Z 측정','Same ensemble · Z measurement',ensemble('z')),
  preset('constant-guess','최소 오류 판정은 정보를 버릴 수 있다','A minimum-error decision can discard information',ensemble('helstrom','9/10',true)),
  preset('biased-z','같은 편향 앙상블 · 양의 측정 정보','Same biased ensemble · positive measured information',ensemble('z','9/10',true)),
 ],
 'quantum-relative-entropy-data-processing':[
  preset('noncommuting','비가환 혼합 상태 · 정보 손실','Noncommuting mixed states · information loss',{kind:'relative-entropy',rho:['0','0','3/5'],sigma:['4/5','0','0']}),
  preset('support-failure','겹쳐도 무한대 · 지지집합','Nonzero overlap yet infinite divergence · supports',{kind:'relative-entropy',rho:['0','0','1'],sigma:['1','0','0']}),
  preset('diagonal-equality','대각 상태 · 탈위상 등호','Diagonal states · equality under dephasing',{kind:'relative-entropy',rho:['0','0','3/5'],sigma:['0','0','0']}),
  preset('same-state','같은 상태 · 정확한 영값','Identical states · exact zero',{kind:'relative-entropy',rho:['1/5','2/5','1/5'],sigma:['1/5','2/5','1/5']}),
 ],
 'no-cloning-bb84-information-disturbance':[
  preset('full-intercept','전면 도청 · 선별 후 오류1/4','Full interception · sifted error1/4',attack('1')),
  preset('partial-intercept','공격 비율2/5 · 오류1/10','Attack fraction2/5 · error1/10',attack('2/5')),
  preset('biased-eve','Z만 고르는 Eve · 기저별 오류','Eve always chooses Z · basis-specific errors',attack('1','1')),
  preset('no-attack','공격 없음 · 비밀성 증명과의 구분','No attack · distinct from a secrecy proof',attack('0')),
  preset('no-sample','검사 표본0 · 오류를 발견하지 못함','Sample size0 · no observed error detection',attack('1','1/2',0)),
 ],
 'teleportation-superdense-coding-resources':[
  preset('complex-input','복소 입력 · 네 보정 가지','Complex input · four corrected branches',teleport(complex)),
  preset('basis-input','기저 입력 · 정규화와 가지 확률','Basis input · normalization and branch probability',teleport(zero,0,0)),
  preset('raw-scale','비정규화 입력 · 같은 밀도행렬','Unnormalized input · same density matrix',teleport([s('3'),s('0','4')],0,1)),
  preset('irrational-input','무리수 norm · 정확 진폭','Irrational norm · exact amplitudes',teleport([s('1','0','1'),s('0','1')],1,0)),
  preset('reverse-paulis','메시지11 · 순서에 따른 전역 부호','Message11 · global sign from Pauli order',teleport(complex,1,1,'z-then-x')),
 ],
 'quantum-error-correction-syndrome-knill-laflamme':[
  preset('single-x','단일 X · 복소 coherence 복구','Single X · restore complex coherence',correction('XII')),
  preset('phase-failure','Z 오류 · syndrome이 놓치는 위상','Z error · phase missed by the syndrome',correction('ZII',complex,['III','ZII'])),
  preset('double-x','두 X · 논리 반전으로 남는 오류','Two X errors · residual logical flip',correction('XXI')),
  preset('misleading-plus','|+〉 충실도1 · 채널 손상','Fidelity1 for |+> · damaged channel',correction('III',plus)),
  preset('degenerate-family','퇴화 오류 가족 · 다른 복구 필요','Degenerate error family · different recovery needed',correction('ZII',complex,['ZII','IZI','IIZ'])),
 ],
 'pure-state-entanglement-majorization-resources':[
  preset('incomparable','누적합 교차 · 양방향 확률 전환','Crossing prefixes · probabilistic conversion both ways',resources(['1/2','1/4','1/4'],['2/5','2/5','1/5'])),
  preset('reverse-incomparable','엔트로피 감소만으로 충분하지 않다','Entropy decrease alone is insufficient',resources(['2/5','2/5','1/5'],['1/2','1/4','1/4'])),
  preset('bell-product','Bell 쌍에서 곱 상태로','Bell pair to product state',resources(['1/2','1/2'],['1'])),
  preset('rank-obstruction','곱 상태에서 얽힘 · rank 장애','Product state to entanglement · rank obstruction',resources(['1'],['1/2','1/2'])),
  preset('actual-dilution','실제 국소 측정 · Bell 희석','Explicit local measurement · Bell dilution',{kind:'majorization',mode:'bell-dilution',a:'4/5',b:'3/5'}),
 ],
 'quantum-channel-capacities-coding-limits':[
  preset('identity','항등 · C1/CE2/Q1','Identity · C1/CE2/Q1',capacity('0')),
  preset('phase-noise','위상 오류1/10 · 서로 다른 용량','Phase error1/10 · distinct capacities',capacity('1/10')),
  preset('pure-input','순수 입력의 coherent information0','Pure-input coherent information0',capacity('1/10',['24/25','0','-7/25'])),
  preset('complete-dephasing','완전 탈위상 · 고전 bit는 보존','Complete dephasing · classical bit survives',capacity('1/2')),
  preset('known-z','알려진 Z · 양자 용량 회복','Known Z · quantum capacity restored',capacity('1')),
  preset('tiny-capacity','표시 반올림과 작은 양의 용량','Display rounding and small positive capacity',capacity('1000000000000000000000000000001/2000000000000000000000000000000')),
 ],
};
export const quantumInformationPresets:Readonly<typeof entries>=Object.freeze(Object.fromEntries(Object.entries(entries).map(([id,presets])=>[id,Object.freeze(presets)])))as Readonly<typeof entries>;
