/** Include physical zero and reserve a proportional margin. Existing oscillator
 * diagrams retain their default margin; small SI quantities can request less.
 */
export function waveVerticalDomain(values:number[],minimumPadding=.01){
 if(!Number.isFinite(minimumPadding)||minimumPadding<=0)throw new RangeError('Positive finite axis padding required.');
 if(values.some(v=>!Number.isFinite(v)))throw new RangeError('Finite vertical coordinates required.');
 const low=Math.min(0,...values),high=Math.max(0,...values),padding=Math.max(minimumPadding,(high-low)*.08);
 return {minimum:low-padding,maximum:high+padding};
}
