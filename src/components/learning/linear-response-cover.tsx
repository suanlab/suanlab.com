/** Decorative native SVG; all numerical work stays in the worker. */
export default function LinearResponseCover({ kind }: { kind: string }) {
  return <svg data-lr-cover={kind} viewBox="0 0 480 320" aria-hidden="true" style={{ width: '100%', height: '100%' }}><g fill="none" stroke="currentColor" strokeWidth={3}><circle cx={95} cy={90} r={30} /><circle cx={240} cy={90} r={30} /><circle cx={385} cy={90} r={30} /><path d="M128 80H204M193 71L205 80L193 89M352 103H276M287 94L275 103L287 112" /><path d="M85 130C125 185 355 185 395 130" opacity={.4} /><path d="M50 195V275H435M65 210C135 250 220 269 425 270M65 210C150 220 215 250 425 265" /><path d="M65 268C155 250 200 235 260 260S350 272 425 272" strokeDasharray="7 6" /><path d="M225 25H255M240 15V45" /></g></svg>;
}
