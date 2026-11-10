/** Only this worker-side dispatcher imports the exact numerical models. */
import {stationaryThreeStateCycleModel} from './learning-stationary-cycle';
import {metropolisHastingsProposalModel} from './learning-hastings-proposal';
import {reversibleTwoStateMixingModel} from './learning-two-state-mixing';
import {detailedBalanceRecord,isDetailedBalanceInput,isDetailedBalanceRequestEnvelope,snapshotDetailedBalanceInput,detailedBalanceInputKey,detailedBalanceKinds,type DetailedBalanceResponse} from './learning-detailed-balance-protocol';
export function evaluateDetailedBalanceRequest(value:unknown):DetailedBalanceResponse {
 const requestId=detailedBalanceRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isDetailedBalanceRequestEnvelope(value))return {version:1,requestId,ok:false,error:'invalid-request'};
 if(!isDetailedBalanceInput(value.input))return {version:1,requestId,ok:false,error:'invalid-input'};
 try{
  const input=snapshotDetailedBalanceInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:detailedBalanceInputKey(input)};
  if(input.kind===detailedBalanceKinds[0])return {...base,kind:input.kind,input,result:stationaryThreeStateCycleModel(input.parameters)};
  if(input.kind===detailedBalanceKinds[1])return {...base,kind:input.kind,input,result:metropolisHastingsProposalModel(input.parameters)};
  return {...base,kind:input.kind,input,result:reversibleTwoStateMixingModel(input.parameters)};
 }catch{return {version:1,requestId,ok:false,error:'calculation-failed'};}
}
