/** Bilingual teaching preparations; this file does not register readers. */
import {parseNumberDraft} from './learning-number-input';
import {correlationKinds as kinds,isCorrelationInput,snapshotCorrelationInput,type CorrelationInput,type CorrelationKind} from './learning-correlations-protocol';
export {correlationKinds} from './learning-correlations-protocol';
export type {CorrelationKind,CorrelationInput};
export interface CorrelationText {ko:string;en:string}
export interface CorrelationPreset {id:string;label:CorrelationText;input:CorrelationInput}
export interface CorrelationGuide {title:CorrelationText;predict:CorrelationText;action:CorrelationText;explain:CorrelationText;presetIndices:readonly number[]}
export const correlationText=(ko:string,en:string):CorrelationText=>({ko,en});
const t=correlationText;
export const isCorrelationKind=(s:string):s is CorrelationKind=>(kinds as readonly string[]).includes(s);
export const correlationTitles:Record<CorrelationKind,CorrelationText>={
 'binary-connected-correlations-field-hessian':t('연결 공분산과 장의 Hessian','Connected covariance and the field Hessian'),
 'periodic-ising-ring-correlation-length':t('주기 Ising 고리의 공간 상관','Spatial correlations on a periodic Ising ring'),
 'finite-correlated-average-ar1-uncertainty':t('상관된 AR(1) 평균의 유한 오차','Finite uncertainty of correlated AR(1) averages'),
};
const pair=(K=0,h1=0,h2=0,a1=1,a2=1):CorrelationInput=>({kind:kinds[0],parameters:{couplingK:K,fieldOne:h1,fieldTwo:h2,probeOne:a1,probeTwo:a2}});
const ring=(N:number,K:number):CorrelationInput=>({kind:kinds[1],parameters:{spinCount:N,couplingK:K}});
const ar=(n:number,rho:number):CorrelationInput=>({kind:kinds[2],parameters:{sampleCount:n,correlationRho:rho}});
const p=(id:string,ko:string,en:string,input:CorrelationInput):CorrelationPreset=>({id,label:t(ko,en),input});
// Pin the actual reference input: Math.log may differ by one ULP across runtimes.
const halfLog3=0.5493061443340549;
export const correlationPresets:Record<CorrelationKind,readonly CorrelationPreset[]>={
 'binary-connected-correlations-field-hessian':[
 p('pair-uniform','독립·편향 없음','Independent, unbiased',pair()),
 p('pair-aligned','양의 결합·합 관측','Positive coupling, sum probe',pair(halfLog3)),
 p('pair-opposing','음의 결합·합 관측','Negative coupling, sum probe',pair(-halfLog3)),
 p('pair-difference','음의 결합·차 관측','Negative coupling, difference probe',pair(-halfLog3,0,0,1,-1)),
 p('pair-biased','독립·반대 편향','Independent, opposing biases',pair(0,halfLog3,-halfLog3)),
 p('pair-general','서로 다른 장·가중 관측','Unequal fields, weighted probe',pair(.7,-.4,.9,2,-1)),
 p('pair-scaled','관측·source 두 배','Double observable and source',pair(.7,-.4,.9,4,-2)),
 p('pair-zero','정확한 영 관측량','Exact zero observable',pair(.7,-.4,.9,0,0)),
 p('pair-tiny','최소 비영 결합','Smallest nonzero coupling',pair(Number.MIN_VALUE,.8,-.4)),
 p('pair-capped','같은 결합·128비트 상한','Same coupling, 128-bit ceiling',{...pair(Number.MIN_VALUE,.8,-.4),options:{maximumPrecision:128}}),
 p('pair-tiny-probe','표시 범위 아래 분산','Variance below linear display',pair(.7,-.4,.9,Number.MIN_VALUE,-Number.MIN_VALUE)),
 ],
 'periodic-ising-ring-correlation-length':[
 p('ring-triangle-plus','양의 삼각형','Positive triangle',ring(3,halfLog3)),
 p('ring-triangle-minus','반강자성 삼각형','Antiferromagnetic triangle',ring(3,-halfLog3)),
 p('ring-square-plus','양의 사각형','Positive square',ring(4,halfLog3)),
 p('ring-square-minus','반강자성 사각형','Antiferromagnetic square',ring(4,-halfLog3)),
 p('ring-odd-cold','홀수127·강한 반대 결합','Odd 127, strong opposing coupling',ring(127,-8)),
 p('ring-even-cold','짝수128·강한 반대 결합','Even 128, strong opposing coupling',ring(128,-8)),
 p('ring-small','N=8·고리 보정','N=8, wrapping correction',ring(8,.4)),
 p('ring-large','N=128·같은 결합','N=128, same coupling',ring(128,.4)),
 p('ring-tiny','먼 거리의 최소 비영 상관','Tiny nonzero distant correlations',ring(128,Number.MIN_VALUE)),
 p('ring-independent','결합0·독립 위치','Zero coupling, independent sites',ring(8,0)),
 ],
 'finite-correlated-average-ar1-uncertainty':[
 p('ar-independent','네 독립 관측','Four independent observations',ar(4,0)),
 p('ar-positive','네 관측·ρ=1/2','Four observations, rho=1/2',ar(4,.5)),
 p('ar-negative','네 관측·ρ=-1/2','Four observations, rho=-1/2',ar(4,-.5)),
 p('ar-short','두 관측·점근식의 차이','Two observations, asymptotic mismatch',ar(2,.5)),
 p('ar-long','64관측·같은 계수','64 observations, same coefficient',ar(64,.5)),
 p('ar-near-constant','512관측·1의 이웃 계수','512 observations, neighbor of one',ar(512,.9999999999999999)),
 p('ar-constant','상수 끝점','Constant endpoint',ar(512,1)),
 p('ar-alternating-even','교대 끝점·짝수','Alternating endpoint, even count',ar(4,-1)),
 p('ar-alternating-odd','교대 끝점·홀수','Alternating endpoint, odd count',ar(5,-1)),
 p('ar-near-alternating','512관측·-1의 이웃 계수','512 observations, neighbor of minus one',ar(512,-.9999999999999999)),
 p('ar-tiny-negative','최소 음의 계수·먼 지연','Smallest negative coefficient, distant lags',ar(512,-Number.MIN_VALUE)),
 ],
};
export function correlationPresetInput(kind:CorrelationKind,index:number):CorrelationInput {
 const item=correlationPresets[kind]?.[index];if(!Number.isInteger(index)||!item||!isCorrelationInput(item.input))throw new RangeError('Unknown correlation preset');return snapshotCorrelationInput(item.input);
}
const guide=(title:CorrelationText,predict:CorrelationText,action:CorrelationText,explain:CorrelationText,presetIndices:number[]):CorrelationGuide=>({title,predict,action,explain,presetIndices});
export const correlationGuides:Record<CorrelationKind,readonly CorrelationGuide[]>={
 'binary-connected-correlations-field-hessian':[
 guide(t('합과 차이의 방향','Directions of sums and differences'),t('음의 교차 공분산이면 합의 분산이 음수일까요?','Does negative cross covariance make the sum variance negative?'),t('독립·양의 결합·음의 결합의 합과 차이를 비교하세요.','Compare sum and difference probes for independent, aligned and opposing preparations.'),t('교차항이 합과 차이에 반대 부호로 들어갑니다. C는 양의 정부호이며 비대각이 음수일 수 있습니다.','Cross terms enter sums and differences with opposite signs. C is positive definite while an off-diagonal entry may be negative.'),[0,1,2,3]),
 guide(t('편향과 연결 상관','Bias and connected correlation'),t('곱의 평균이 음수이면 의존성일까요?','Does a negative product mean establish dependence?'),t('편향 없는 독립 준비와 반대 편향의 독립 준비를 비교하세요.','Compare unbiased independence with independently opposing biases.'),t('독립이면 곱 평균은 평균의 곱입니다. 연결 값은 이를 빼므로 편향과 무관하게0입니다.','Under independence the product mean equals the product of means. Subtracting it gives zero connected covariance despite the biases.'),[0,4]),
 guide(t('source 척도와 수치 표현','Source scale and numerical representation'),t('가중치를 두 배로 하면 분산도 두 배일까요?','Does doubling weights double variance?'),t('가중 관측·두 배 척도·영 관측·작은 결합과 상한을 비교하세요.','Compare weighted, doubled and zero probes, then tiny coupling and its precision cap.'),t('관측과 대응 source를 함께 두 배로 하면 분산은 네 배입니다. 정확한0, 비영 표시 범위 밖, 상한 미해결을 구별합니다.','Doubling both observable and matching source multiplies variance by four. Distinguish exact zero, outside-display nonzero and unresolved precision caps.'),[5,6,7,8,9,10]),
 ],
 'periodic-ising-ring-correlation-length':[
 guide(t('양쪽 경로를 세기','Count both paths'),t('삼각형의 C(1)은 항상 tanhK일까요?','Is triangle C(1) always tanhK?'),t('양·음 삼각형과 사각형에서 두 경로와 상관을 비교하세요.','Compare both paths and correlations in positive and negative triangles and squares.'),t('짧은 경로·돌아오는 경로와 고리 정규화가 모두 들어갑니다. 무한 사슬 참조는 유한 고리의 정답이 아닙니다.','Both paths and loop normalization contribute. The infinite-chain reference is not the finite-ring answer.'),[0,1,2,3]),
 guide(t('좌절과 균일 반응','Frustration and uniform response'),t('강한 반강자성 홀수 고리는 완전히 교대할 수 있을까요?','Can an odd antiferromagnetic ring alternate perfectly?'),t('홀수127과 짝수128의 반응·길이·이웃 상관을 비교하세요.','Compare response, length and nearest correlation for odd 127 and even 128.'),t('홀수 고리에는 불만족 결합이 필요합니다. 긴 교대 포락선과 작은 균일 반응이 공존할 수 있습니다.','An odd ring needs an unsatisfied bond. A long alternating envelope can coexist with a small uniform response.'),[1,4,5]),
 guide(t('크기·거리·영 결합','Size, distance and zero coupling'),t('거리 축을 한 바퀴 늘려도 지수감쇠할까요?','Does exponential decay continue around a full circumference?'),t('같은 K의 N=8,128, 최소 결합과 영 결합을 비교하세요.','Compare N=8 and 128 at the same K, then tiny and zero coupling.'),t('유한 고리는 주기적이며 자기 상관은1입니다. 먼 비영 상관은 표시 범위 아래일 수 있고 K=0만 구조적 독립입니다.','The finite ring is periodic with self correlation one. Distant nonzero correlation may underflow the display; K=0 establishes structural independence.'),[6,7,8,9]),
 ],
 'finite-correlated-average-ar1-uncertainty':[
 guide(t('유한 쌍의 부호','Signs of finite covariance pairs'),t('같은 관측 수면 평균 오차도 같을까요?','Does equal observation count imply equal mean uncertainty?'),t('n=4에서 ρ=0,1/2,-1/2의 행렬·분산·표준오차를 비교하세요.','At n=4 compare matrices, variance and standard error for rho=0,1/2,-1/2.'),t('공분산은 지연별2(n-k)번 들어갑니다. 분산과 표준오차의 단위를 구별하며 음의 항도 합의 비음성을 깨지 않습니다.','Each lag contributes 2(n-k) covariance terms. Distinguish variance from standard error; negative terms do not violate nonnegative total variance.'),[0,1,2]),
 guide(t('유한 오차와 점근식','Finite uncertainty and asymptotics'),t('τ/n은 짧은 표본에도 정확할까요?','Is tau/n exact for short samples?'),t('같은 ρ=1/2에서 n=2,4,64를 비교하고 1의 이웃 계수도 보세요.','Compare n=2,4,64 at rho=1/2, then inspect the neighbor of one.'),t('τ/n은 고정 내부 모수의 점근식입니다. 짧거나 거의 비혼합인 준비에서는 정확한 유한 답과 크게 다릅니다.','Tau/n is asymptotic at a fixed interior parameter. Short or nearly nonmixing preparations can differ greatly from the exact finite answer.'),[3,1,4,5]),
 guide(t('끝점과 관측량별 등가 수','Endpoints and observable-specific counts'),t('짝수 평균이 상쇄되면 모든 정보가 무한할까요?','Does exact cancellation imply infinite information for every observable?'),t('상수·교대 짝홀·-1의 이웃과 작은 음의 계수를 비교하세요.','Compare constant and alternating endpoints, their neighbor and tiny negative dependence.'),t('끝점은 정상이지만 비혼합입니다. 등가 수는 선형 평균의 분산 비교이며, 제곱 평균이나 독립 데이터 개수가 아닙니다.','Endpoints are stationary but nonmixing. The equivalent count compares linear-mean variance, not squared means or independent data counts.'),[6,7,8,9,10]),
 ],
};
/** Apply decimal text as binary64, rejecting nonzero underflow. */
export function parseCorrelationNumber(text:string,integer=false):{ok:true;value:number}|{ok:false}{const s=text.trim();if(s.length>256||! /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s))return {ok:false};if(integer){const value=parseNumberDraft(s,-Number.MAX_VALUE,Number.MAX_VALUE,true);return value===null?{ok:false}:{ok:true,value};}const value=Number(s);return !Number.isFinite(value)||value===0&&/[1-9]/.test(s.split(/[eE]/)[0])?{ok:false}:{ok:true,value:value===0?0:value};}
