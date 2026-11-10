import {evaluateRegularPerturbationRequest} from './learning-regular-perturbation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateRegularPerturbationRequest(event.data,message=>scope.postMessage(message)));
