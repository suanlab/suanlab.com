import { bindWorkerMessageError, clearWorkerMessageError } from './learning-worker-message-error';
import {readRealTimePhaseInput,type RealTimePhaseInput} from './learning-real-time-phase-input';
import {realTimePhaseData,realTimePhaseInputKey,realTimePhaseUnverifiedHeader,type RealTimePhaseValidatedSuccess} from './learning-real-time-phase-protocol';
import {snapshotQubitWireInTasks,QubitWireLimit,type QubitCopyOptions} from './learning-qubit-wire';
export type RealTimePhaseWorkerPort=Pick<Worker,'postMessage'|'terminate'|'onmessage'|'onerror'|'onmessageerror'>&{
  addEventListener?(type:'messageerror',listener:EventListener):void;
  removeEventListener?(type:'messageerror',listener:EventListener):void;
};
export type RealTimePhaseControllerError='invalid-input'|'unavailable'|'worker-failed'|'invalid-response'|'resource-limit'|'timeout';
export type RealTimePhaseCalculationState=
  |{status:'idle';requestId:null;input:null;inputKey:null}
  |{status:'pending';phase:'starting'|'calculating'|'validating';requestId:number;input:RealTimePhaseInput;inputKey:string}
  |{status:'ready';requestId:number;input:RealTimePhaseInput;inputKey:string;response:RealTimePhaseValidatedSuccess}
  |{status:'error';requestId:number;input:RealTimePhaseInput|null;inputKey:string|null;error:RealTimePhaseControllerError};
export const idleRealTimePhaseState=():RealTimePhaseCalculationState=>({status:'idle',requestId:null,input:null,inputKey:null});
type Token={id:number;input:RealTimePhaseInput;key:string;started:boolean;accepting:boolean};
function envelope(value:unknown) {
  try {
    if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype)return null;
    const keys=Reflect.ownKeys(value);return keys.length<=12&&keys.every(k=>typeof k==='string')?realTimePhaseData(value,keys as string[]):null;
  }catch{return null;}
}
/** Main-thread orchestration only. Each run owns two separate Worker ports;
 * the mathematical gate never enters this module orits reader bundle.
 * Both large copies yield andcheck request ownership before every chunk.
 */
export class RealTimePhaseCalculationController {
  private calculation:RealTimePhaseWorkerPort|null=null;
  private validation:RealTimePhaseWorkerPort|null=null;
  private active:Token|null=null;
  private sequence=0;
  private disposed=false;
  private deadline:ReturnType<typeof setTimeout>|null=null;
  private state:RealTimePhaseCalculationState=idleRealTimePhaseState();
  constructor(private readonly factories:{calculate:()=>RealTimePhaseWorkerPort;validate:()=>RealTimePhaseWorkerPort},
    private readonly notify:(state:RealTimePhaseCalculationState)=>void,
    private readonly copyOptions:Omit<QubitCopyOptions,'active'>={},private readonly timeoutMilliseconds=120000) {
    if(!Number.isInteger(timeoutMilliseconds)||timeoutMilliseconds<10||timeoutMilliseconds>120000)throw new RangeError('Worker deadline10..120000ms required');
  }
  getSnapshot(){return this.state;}
  private publish(state:RealTimePhaseCalculationState){this.state=Object.freeze(state);this.notify(this.state);}
  private current(t:Token){return !this.disposed&&this.active===t;}
  private release(port:RealTimePhaseWorkerPort|null) {
    if(!port)return;port.onmessage=null;port.onerror=null;clearWorkerMessageError(port);
    try{port.terminate();}catch{/* Still release the other port anddeadline. */}
  }
  private stop() {
    if(this.deadline!==null)clearTimeout(this.deadline);this.deadline=null;
    const a=this.calculation,b=this.validation;this.calculation=null;this.validation=null;this.active=null;
    this.release(a);this.release(b);
  }
  private fail(t:Token,error:RealTimePhaseControllerError) {
    if(!this.current(t))return;this.stop();this.publish({status:'error',requestId:t.id,input:t.input,inputKey:t.key,error});
  }
  run(value:unknown):number|null {
    if(this.disposed)return null;
    if(!Number.isSafeInteger(this.sequence+1)){this.stop();this.publish(idleRealTimePhaseState());return null;}
    this.stop();const id=++this.sequence,input=readRealTimePhaseInput(value);
    if(!input){this.publish({status:'error',requestId:id,input:null,inputKey:null,error:'invalid-input'});return id;}
    const t:Token={id,input,key:realTimePhaseInputKey(input),started:false,accepting:false};this.active=t;
    this.publish({status:'pending',phase:'starting',requestId:id,input,inputKey:t.key});if(!this.current(t))return id;
    this.deadline=setTimeout(()=>this.fail(t,'timeout'),this.timeoutMilliseconds);
    let port:RealTimePhaseWorkerPort;
    try{port=this.factories.calculate();}catch{this.fail(t,'unavailable');return id;}
    if(!this.current(t)){this.release(port);return id;}this.calculation=port;
    port.onmessage=event=>{
      if(!this.current(t)||this.calculation!==port)return;
      const message=envelope(event.data);if(!message){this.fail(t,'invalid-response');return;}
      if(typeof message.requestId==='number'&&Number.isSafeInteger(message.requestId)&&message.requestId>0&&message.requestId!==id)return;
      if(message.version!==1||message.requestId!==id){this.fail(t,'invalid-response');return;}
      if(realTimePhaseData(message,['version','requestId','phase','kind'])&&message.phase==='started'&&message.kind==='real-time-phase') {
        if(!t.started){t.started=true;this.publish({status:'pending',phase:'calculating',requestId:id,input,inputKey:t.key});}return;
      }
      if(!realTimePhaseUnverifiedHeader(message,input,id)) {
        const failure=realTimePhaseData(message,['version','requestId','ok','error']);
        this.fail(t,failure?.ok===false?failure.error==='resource-limit'?'resource-limit':failure.error==='invalid-input'?'invalid-input':
          ['calculation-failed','invalid-request'].includes(failure.error as string)?'worker-failed':'invalid-response':'invalid-response');return;
      }
      this.calculation=null;this.release(port);this.publish({status:'pending',phase:'validating',requestId:id,input,inputKey:t.key});
      if(this.current(t))void this.validate(t,message);
    };
    port.onerror=event=>{if(this.current(t)&&this.calculation===port){event.preventDefault();this.fail(t,'worker-failed');}};
    bindWorkerMessageError(port,()=>{if(this.current(t)&&this.calculation===port)this.fail(t,'invalid-response');});
    try{port.postMessage({version:1,requestId:id,input});}catch{this.fail(t,'worker-failed');}
    return id;
  }
  private async validate(t:Token,response:unknown) {
    try {
      const owned=await snapshotQubitWireInTasks(response,{...this.copyOptions,active:()=>this.current(t)});if(!this.current(t))return;
      let port:RealTimePhaseWorkerPort;
      try{port=this.factories.validate();}catch{this.fail(t,'unavailable');return;}
      if(!this.current(t)){this.release(port);return;}this.validation=port;
      port.onmessage=event=>{
        if(!this.current(t)||this.validation!==port||t.accepting)return;
        const receipt=envelope(event.data);if(!receipt){this.fail(t,'invalid-response');return;}
        if(typeof receipt.requestId==='number'&&Number.isSafeInteger(receipt.requestId)&&receipt.requestId>0&&receipt.requestId!==t.id)return;
        if(realTimePhaseData(receipt,['version','requestId','phase','kind'])&&receipt.version===1&&receipt.requestId===t.id&&receipt.phase==='verification-started'&&receipt.kind==='real-time-phase')return;
        const rejected=realTimePhaseData(receipt,['version','requestId','phase','error']);
        if(rejected&&receipt.version===1&&receipt.requestId===t.id&&receipt.phase==='rejected') {
          this.fail(t,receipt.error==='resource-limit'?'resource-limit':receipt.error==='invalid-input'?'invalid-input':receipt.error==='invalid-request'?'worker-failed':'invalid-response');return;
        }
        if(!realTimePhaseData(receipt,['version','requestId','phase','kind','inputKey','fullResponseValidated','response'])||
          receipt.version!==1||receipt.requestId!==t.id||receipt.phase!=='validated'||receipt.kind!=='real-time-phase'||
          receipt.inputKey!==t.key||receipt.fullResponseValidated!==true||!realTimePhaseUnverifiedHeader(receipt.response,t.input,t.id)){this.fail(t,'invalid-response');return;}
        t.accepting=true;this.validation=null;this.release(port);void this.accept(t,receipt.response);
      };
      port.onerror=event=>{if(this.current(t)&&this.validation===port){event.preventDefault();this.fail(t,'worker-failed');}};
      bindWorkerMessageError(port,()=>{if(this.current(t)&&this.validation===port)this.fail(t,'invalid-response');});
      try{port.postMessage({version:1,requestId:t.id,input:t.input,response:owned});}catch{this.fail(t,'worker-failed');}
    }catch(error){if(this.current(t))this.fail(t,error instanceof QubitWireLimit?'resource-limit':'invalid-response');}
  }
  private async accept(t:Token,response:unknown) {
    try {
      // Structured clone removes Object.freeze. Re-own/freeze after the
      // trusted validation port,without importing any math onthe main thread.
      const owned=await snapshotQubitWireInTasks(response,{...this.copyOptions,active:()=>this.current(t)});if(!this.current(t))return;
      if(!realTimePhaseUnverifiedHeader(owned,t.input,t.id)){this.fail(t,'invalid-response');return;}
      this.stop();this.publish({status:'ready',requestId:t.id,input:t.input,inputKey:t.key,response:owned as RealTimePhaseValidatedSuccess});
    }catch(error){if(this.current(t))this.fail(t,error instanceof QubitWireLimit?'resource-limit':'invalid-response');}
  }
  cancel(){if(!this.disposed){this.stop();this.publish(idleRealTimePhaseState());}}
  dispose(){if(!this.disposed){this.disposed=true;this.stop();}}
}
