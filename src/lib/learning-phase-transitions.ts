/** Lightweight teaching data. Exporting it does not register a book reader. */
import {parseNumberDraft} from './learning-number-input';
import {phaseKinds as kinds,isPhaseInput,snapshotPhaseInput,type PhaseInput,type PhaseKind} from './learning-phase-transitions-protocol';
import type {CurieWeissParameters} from './learning-curie-weiss';
import type {CurieWeissVariationalParameters} from './learning-curie-weiss-variational';
import type {LandauSixthOrderParameters} from './learning-landau-sixth-order';
export {phaseKinds} from './learning-phase-transitions-protocol';
export type {PhaseKind,PhaseInput};
export interface PhaseText {ko:string;en:string}
export interface PhasePreset {id:string;label:PhaseText;input:PhaseInput}
export interface PhaseGuide {title:PhaseText;predict:PhaseText;action:PhaseText;explain:PhaseText;presetIndices:readonly number[]}
const t=(ko:string,en:string):PhaseText=>({ko,en});
export const isPhaseKind=(v:string):v is PhaseKind=>(kinds as readonly string[]).includes(v);
export const phaseTitles:Record<PhaseKind,PhaseText>={
 'finite-curie-weiss-magnetization-symmetry':t('유한 Curie–Weiss 분포와 대칭','Finite Curie–Weiss distributions and symmetry'),
 'curie-weiss-variational-stability-limits':t('변분 극한·안정성·대칭 깨짐','Variational limits, stability and symmetry breaking'),
 'landau-sixth-order-coexistence-stability':t('육차 Landau 공존과 국소 안정성','Sixth-order Landau coexistence and local stability'),
};
const finite=(p:Partial<CurieWeissParameters>={}):PhaseInput=>({kind:kinds[0],parameters:{spinCount:2,couplingK:0,fieldB:0,...p}});
const variational=(p:Partial<CurieWeissVariationalParameters>={}):PhaseInput=>({kind:kinds[1],parameters:{couplingK:.5,fieldB:0,...p}});
const landau=(p:Partial<LandauSixthOrderParameters>={}):PhaseInput=>({kind:kinds[2],parameters:{quadraticA:.1,quarticB:1,sexticC:1,fieldH:0,...p}});
const preset=(id:string,ko:string,en:string,input:PhaseInput):PhasePreset=>({id,label:t(ko,en),input});
export const phasePresets:Record<PhaseKind,readonly PhasePreset[]>={
 'finite-curie-weiss-magnetization-symmetry':[
  preset('finite-two-independent','두 스핀·다중도1:2:1','Two spins, multiplicities 1:2:1',finite()),
  preset('finite-actual-log-two','실제 log(2)·표시상 동률','Actual log(2), displayed tie',finite({couplingK:Math.log(2)})),
  preset('finite-three-independent','세 스핀·평균0과 비영 상태','Three spins, zero mean and nonzero states',finite({spinCount:3})),
  preset('finite-symmetric-sixteen','N=16·대칭인 두 봉우리','N=16, symmetric two peaks',finite({spinCount:16,couplingK:2})),
  preset('finite-symmetric-thirty-two','N=32·부호 평균0','N=32, zero signed mean',finite({spinCount:32,couplingK:2})),
  preset('finite-symmetric-sixty-four','N=64·유한 평형 반응','N=64, finite equilibrium response',finite({spinCount:64,couplingK:2})),
  preset('finite-positive-field','양의 장·집단 확률 이동','Positive field, shifted population weights',finite({spinCount:32,couplingK:2,fieldB:.2})),
  preset('finite-negative-field','음의 장·반사된 분포','Negative field, reflected distribution',finite({spinCount:32,couplingK:2,fieldB:-.2})),
  preset('finite-subnormal-field','최소 비영 장·부호 보존','Smallest nonzero field, retained sign',finite({spinCount:32,couplingK:2,fieldB:Number.MIN_VALUE})),
 ],
 'curie-weiss-variational-stability-limits':[
  preset('variational-subcritical','K=0.5·하나의 최소','K=0.5, one minimum',variational()),
  preset('variational-critical','K=1·사차 임계 최소','K=1, quartic critical minimum',variational({couplingK:1})),
  preset('variational-coexistence','K=2·두 전역 최소','K=2, two global minima',variational({couplingK:2})),
  preset('variational-below-spinodal','접힘 바로 아래·세 근','Just below the fold, three roots',variational({couplingK:9/8,fieldB:.028426409720027343})),
  preset('variational-rounded-spinodal','반올림된 임계 장·한 근','Rounded threshold, one root',variational({couplingK:9/8,fieldB:.028426409720027346})),
  preset('variational-above-spinodal','다음 실수·한 근','Next binary64 field, one root',variational({couplingK:9/8,fieldB:.02842640972002735})),
  preset('variational-positive-field','양의 장·준안정 최소','Positive field, metastable minimum',variational({couplingK:2,fieldB:.2})),
  preset('variational-negative-field','음의 장·전역 선택 반사','Negative field, reflected global selection',variational({couplingK:2,fieldB:-.2})),
  preset('variational-subnormal-field','같아 보이는 에너지·비영 분할','Equal-looking energies, nonzero splitting',variational({couplingK:2,fieldB:Number.MIN_VALUE})),
 ],
 'landau-sixth-order-coexistence-stability':[
  preset('landau-disordered','a=0.1,b=1·중앙 최소','a=0.1,b=1, central minimum',landau()),
  preset('landau-quartic-critical','a=0,b=1·삼중 정지근','a=0,b=1, triple stationary root',landau({quadraticA:0})),
  preset('landau-continuous-ordered','a=-0.1,b=1·정렬 쌍','a=-0.1,b=1, ordered pair',landau({quadraticA:-.1})),
  preset('landau-tricritical','a=b=0·오중 정지근','a=b=0, quintuple stationary root',landau({quadraticA:0,quarticB:0})),
  preset('landau-ordered-fold','a=1/4,b=-1·이중근','a=1/4,b=-1, double roots',landau({quadraticA:1/4,quarticB:-1})),
  preset('landau-three-coexistence','a=3/16,b=-1·세 최소 공존','a=3/16,b=-1, three coexisting minima',landau({quadraticA:3/16,quarticB:-1})),
  preset('landau-flat-maximum','a=0,b=-1·평평한 극대','a=0,b=-1, flat maximum',landau({quadraticA:0,quarticB:-1})),
  preset('landau-tilted-fold','H=-1/8·기울어진 접힘','H=-1/8, tilted fold',landau({quadraticA:-5/16,quarticB:0,fieldH:-1/8})),
  preset('landau-tilted-wells','작은 장·여러 국소 최소','Small field, multiple local minima',landau({quadraticA:.2,quarticB:-1,fieldH:.005})),
  preset('landau-tilted-quartic','m=1/2·기울어진 사차 최소','m=1/2, tilted quartic minimum',landau({quadraticA:15/16,quarticB:-2.5,sexticC:3,fieldH:.25})),
  preset('landau-subnormal-field','공존 계수·최소 비영 장','Coexistence coefficients, smallest field',landau({quadraticA:3/16,quarticB:-1,fieldH:Number.MIN_VALUE})),
  preset('landau-unresolved-tie','비영 장 동률·상한의 두 후보','Nonzero-field tie, two candidates at the cap',{...landau({quadraticA:59/64,quarticB:-13/4,sexticC:3,fieldH:1/8}),options:{maximumPrecision:128}}),
 ],
};
export function phasePresetInput(kind:PhaseKind,index:number):PhaseInput {
 const p=phasePresets[kind]?.[index];if(!Number.isInteger(index)||!p||!isPhaseInput(p.input))throw new RangeError('Unknown phase-transition preset');return snapshotPhaseInput(p.input);
}
export const phaseGuides:Record<PhaseKind,readonly PhaseGuide[]>={
 'finite-curie-weiss-magnetization-symmetry':[
  {title:t('배치 수에서 확률로','From configuration counts to probabilities'),predict:t('자화 레이블 세 개가 같은 확률일까요?','Are three magnetization labels equally probable?'),action:t('독립 두 스핀, 실제 log(2), 독립 세 스핀을 비교하세요.','Compare two independent spins, actual log(2) and three independent spins.'),explain:t('다중도는 미시 배치를 셉니다. 실제 log(2)의 표시 확률이 같아도 최빈값 동률의 증명은 아닙니다. 홀수 N에는 자화0 상태가 없어도 평균은0입니다.','Multiplicity counts microscopic configurations. Equal displayed probabilities at actual log(2) do not prove a modal tie. Odd N has no zero-magnetization configuration, yet its mean can be zero.'),presetIndices:[0,1,2]},
  {title:t('평균0과 두 봉우리','Zero mean and two peaks'),predict:t('N을 늘리면 영장 평균이 한 봉우리로 이동할까요?','Does the zero-field mean move to one peak as N grows?'),action:t('K=2,b=0에서 N=16,32,64의 분포와 세 평균·반응을 비교하세요.','At K=2,b=0 compare the distributions, three moments and response for N=16,32,64.'),explain:t('스핀 반전 대칭으로 부호 평균은 계속0입니다. 절댓값 평균·이차 모멘트는 양수이며 N Var(m)은 두 집단의 재분배를 포함합니다. 유한 이봉 분포는 열역학 특이점의 증명이 아닙니다.','Spin-flip symmetry keeps the signed mean zero. Mean magnitude and second moment are positive; N Var(m) includes population reweighting. A finite two-peak distribution does not prove a thermodynamic singularity.'),presetIndices:[3,4,5]},
  {title:t('보이지 않는 장의 선택','Selection by a barely visible field'),predict:t('아주 작은 양의 장은 정확한 영장과 같을까요?','Is a tiny positive field the same as exact zero field?'),action:t('양·음의 장과 최소 비영 장을 비교하고 평균의 부호·표현 상태를 읽으세요.','Compare positive, negative and smallest nonzero fields; read the mean sign and representation status.'),explain:t('부호 평균은 장의 부호를 따르고 음·양 장은 분포를 반사합니다. 표시 범위 밖 비영과 정밀도 상한 미해결, 정확한0을 구별해야 합니다.','The signed mean follows the field and field reversal reflects the law. Distinguish outside-display nonzero values, unresolved precision limits and exact zero.'),presetIndices:[6,7,8]},
 ],
 'curie-weiss-variational-stability-limits':[
  {title:t('곡률이 사라질 때','When curvature vanishes'),predict:t('곡률0이면 최소가 아닌가요?','Does zero curvature rule out a minimum?'),action:t('영장에서 K=0.5,1,2의 곡선과 모든 근을 비교하세요.','Compare curves and all roots at zero field for K=0.5,1,2.'),explain:t('K=1의 양의 사차항은 임계 최소를 만듭니다. K>1에서는 두 전역 최소와 중앙 극대를 구분합니다. 가지 미분은 유한계 혼합의 감수율과 다릅니다.','The positive quartic term at K=1 establishes a critical minimum. For K>1 distinguish two global minima from the central maximum. A branch derivative differs from finite-mixture susceptibility.'),presetIndices:[0,1,2]},
  {title:t('가까운 근을 합치지 않기','Keep nearby roots distinct'),predict:t('반올림한 임계 장을 정확한 접점이라 불러도 될까요?','Can a rounded threshold be called an exact tangency?'),action:t('K=9/8에서 인접 binary64 장 세 개를 선택하고 국소 확대와 근 표를 비교하세요.','At K=9/8 select three adjacent binary64 fields and compare the local zoom with the root table.'),explain:t('첫 입력은 세 근, 나머지 둘은 한 근입니다. 접힘 직전의 두 근은 자유에너지 표시가 같아도 서로 다른 근입니다. 비교가 해결되지 않으면 상한 미해결로 남깁니다.','The first input has three roots and the other two have one. Just before the fold, equal displayed energies do not merge distinct roots. An undecidable capped comparison remains unresolved.'),presetIndices:[3,4,5]},
  {title:t('국소 안정성과 전역 선택','Local stability and global selection'),predict:t('두 최소의 소수 표시가 같으면 공존인가요?','Do matching displayed energies establish coexistence?'),action:t('양·음의 장과 최소 비영 장에서 역할·장벽·최소 에너지 차이를 비교하세요.','Compare roles, barriers and the minimum-energy gap under positive, negative and smallest fields.'),explain:t('장의 부호와 대칭 정리가 전역 최소를 선택합니다. 반대쪽은 준안정 최소이며 중앙은 극대입니다. 작은 에너지 차이의 부호는 표시 숫자의 뺄셈으로 결정하지 않습니다.','Field sign and symmetry select the global minimum. The opposite minimum is metastable and the center is a maximum. Rounded-number subtraction cannot decide a tiny energy gap.'),presetIndices:[6,7,8]},
 ],
 'landau-sixth-order-coexistence-stability':[
  {title:t('사차와 육차의 균형','Quartic and sextic balances'),predict:t('곡률0인 두 최소가 같은 임계 반응을 가질까요?','Do two zero-curvature minima have the same critical response?'),action:t('양의 사차 계수에서 a를 바꾸고 사차항0의 삼중임계 준비와 비교하세요.','Vary a with positive quartic coefficient, then compare the zero-quartic tricritical preparation.'),explain:t('연속 정렬 진폭은 b>0에서 제곱근, b=0에서 네제곱근으로 시작합니다. 원점의 삼중·오중 정지근은 모두 최소일 수 있지만 고차항과 장 반응이 다릅니다.','The continuous ordered amplitude begins with a square root for b>0 and a fourth root for b=0. Triple and quintuple stationary roots can both be minima, with different leading terms and field response.'),presetIndices:[0,1,2,3]},
  {title:t('공존과 접힘을 분리하기','Separate coexistence from folds'),predict:t('평형 전이에서 모든 최소의 곡률이0이 될까요?','Must every minimum have zero curvature at equilibrium coexistence?'),action:t('a=1/4,3/16,0과 비영 장의 접힘을 비교하세요.','Compare a=1/4,3/16,0 and the nonzero-field fold.'),explain:t('공존에서는 세 최소의 깊이가 같고 곡률은 양수입니다. 접힘은 국소 가지가 합쳐지는 이중근이며 정지 변곡점입니다. 중앙의 평평한 극대는 또 다른 경계입니다.','At coexistence three minima have equal depth and positive curvature. A fold merges local branches into a double root and stationary inflection. The central flat maximum is another boundary.'),presetIndices:[4,5,6,7]},
  {title:t('기울어진 지형과 남겨야 할 후보','Tilted landscapes and honest candidates'),predict:t('양의 장이면 양의 국소 최소 하나만 남을까요?','Does a positive field leave just one positive local minimum?'),action:t('작은 장, 기울어진 사차 최소, 최소 비영 장, 상한의 두 후보를 비교하세요.','Compare small-field wells, a tilted quartic minimum, the smallest field and two capped candidates.'),explain:t('여러 양의 최소도 경쟁할 수 있어 실제 에너지 비교가 필요합니다. 정확한 삼중근과 작은 비영 분할을 보존하고, 무리수 동률을 구간으로 인증하지 못하면 후보 둘을 남깁니다.','Several positive minima can compete, requiring energy comparison. Preserve exact triple roots and tiny nonzero splittings; when intervals cannot certify an algebraic tie, retain both candidates.'),presetIndices:[8,9,10,11]},
 ],
};
/** Decimal text is converted to binary64; never turn nonzero underflow into zero. */
export function parsePhaseNumber(text:string,integer=false):{ok:true;value:number}|{ok:false}{const s=text.trim();if(s.length>256||! /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s))return {ok:false};if(integer){const value=parseNumberDraft(s,-Number.MAX_VALUE,Number.MAX_VALUE,true);return value===null?{ok:false}:{ok:true,value};}const value=Number(s);return !Number.isFinite(value)||value===0&&/[1-9]/.test(s.split(/[eE]/)[0])?{ok:false}:{ok:true,value:value===0?0:value};}
