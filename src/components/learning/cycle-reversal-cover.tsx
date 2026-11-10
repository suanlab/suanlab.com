/** Decorative cover; no numerical evaluation or worker during SSR. */
export default function CycleReversalCover({kind}:{kind:string}) {
  return <svg data-reversal-cover={kind} viewBox="0 0 480 320" aria-hidden="true" style={{width:'100%',height:'100%'}}><g fill="none" stroke="currentColor" strokeWidth={3}><path d="M240 70L95 240L385 240Z" opacity={.5}/><path d="M208 111L135 199M152 183L135 199L139 174M155 240H326M309 229L326 240L309 251M360 210L268 101M271 126L268 101L292 112"/><circle cx={240} cy={70} r={33}/><circle cx={95} cy={240} r={33}/><circle cx={385} cy={240} r={33}/><path d="M230 149L205 174L230 199M250 149L275 174L250 199" opacity={.6}/></g></svg>;
}
