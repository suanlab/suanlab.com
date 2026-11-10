import {evaluateGaugeOrbitRequest} from './learning-gauge-orbits-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateGaugeOrbitRequest(event.data,message=>scope.postMessage(message)));
