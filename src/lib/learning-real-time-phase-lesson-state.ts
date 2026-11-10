/** Reader-safe learning state; limits are supplied by the teaching data layer. */
export interface RealTimeLessonState {readonly guide:number;readonly explanationOpen:boolean;readonly predictions:readonly string[];readonly answers:readonly number[];}
export interface RealTimeLessonLimits {guides:number;options:readonly number[];}
export function emptyRealTimeLesson(limits:RealTimeLessonLimits):RealTimeLessonState {
  return Object.freeze({guide:0,explanationOpen:false,predictions:Object.freeze(Array(limits.guides).fill('') as string[]),answers:Object.freeze(Array(limits.options.length).fill(-1) as number[])});
}
function array(value:unknown,length:number):unknown[]|null{
  if(!Array.isArray(value))return null;
  const d=Object.getOwnPropertyDescriptors(value) as unknown as Record<string,PropertyDescriptor>;
  if(Reflect.ownKeys(d).length!==length+1||d.length?.value!==length)return null;
  const result:unknown[]=[];for(let i=0;i<length;i++){const item=d[String(i)];if(!item||!Object.hasOwn(item,'value'))return null;result.push(item.value);}return result;
}
export function readRealTimeLesson(value:unknown,limits:RealTimeLessonLimits):RealTimeLessonState|null{
  try{
    if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype)return null;
    const d=Object.getOwnPropertyDescriptors(value),keys=['guide','explanationOpen','predictions','answers'];
    if(Reflect.ownKeys(d).length!==keys.length||keys.some(k=>!d[k]||!Object.hasOwn(d[k],'value')))return null;
    const guide=d.guide.value,open=d.explanationOpen.value,predictions=array(d.predictions.value,limits.guides),answers=array(d.answers.value,limits.options.length);
    if(typeof guide!=='number'||!Number.isInteger(guide)||guide<0||guide>=limits.guides||typeof open!=='boolean'||!predictions||!answers)return null;
    if(predictions.some(v=>typeof v!=='string'||v.length>2000)||answers.some((v,i)=>typeof v!=='number'||!Number.isInteger(v)||v< -1||v>=limits.options[i]))return null;
    return Object.freeze({guide,explanationOpen:open,predictions:Object.freeze(predictions as string[]),answers:Object.freeze(answers as number[])});
  }catch{return null;}
}
