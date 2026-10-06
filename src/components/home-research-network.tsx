'use client';

import { useId } from 'react';

const nodes = [
  { id: 'llm', x: 58, y: 54, label: 'LLM', color: '#67e8f9' },
  { id: 'rag', x: 302, y: 54, label: 'RAG', color: '#a78bfa' },
  { id: 'agents', x: 58, y: 206, label: 'Agents', color: '#34d399' },
  { id: 'multimodal', x: 302, y: 206, label: 'Multimodal', color: '#fbbf24' },
];
const edges = [
  { from: 'llm', to: 'suanlab', d: 'M58 54L180 130' },
  { from: 'rag', to: 'suanlab', d: 'M302 54L180 130' },
  { from: 'agents', to: 'suanlab', d: 'M58 206L180 130' },
  { from: 'multimodal', to: 'suanlab', d: 'M302 206L180 130' },
  { from: 'llm', to: 'rag', d: 'M58 54Q180 8 302 54' },
  { from: 'agents', to: 'multimodal', d: 'M58 206Q180 264 302 206' },
  { from: 'llm', to: 'agents', d: 'M58 54Q8 130 58 206' },
  { from: 'rag', to: 'multimodal', d: 'M302 54Q352 130 302 206' },
  { from: 'llm', to: 'multimodal', d: 'M58 54Q180 20 302 206' },
  { from: 'rag', to: 'agents', d: 'M302 54Q180 244 58 206' },
];

export function HomeResearchNetwork() {
  const gradient = `research-network-${useId().replace(/:/g, '')}`;
  return (
    <div aria-hidden="true" className="identity-network research-network relative mx-auto mb-6 max-w-xs">
      <svg viewBox="0 0 360 260" className="w-full" fill="none" focusable="false">
        <defs>
          <linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1"><stop stopColor="#67e8f9" /><stop offset=".5" stopColor="#a78bfa" /><stop offset="1" stopColor="#34d399" /></linearGradient>
        </defs>
        {edges.map((edge, i) => <g key={`${edge.from}-${edge.to}`} data-from={edge.from} data-to={edge.to} className="research-connection">
          <path d={edge.d} stroke={nodes[i % nodes.length].color} strokeWidth="1" strokeOpacity=".25" />
          <path className="research-link-signal" d={edge.d} pathLength="100" stroke={nodes[i % nodes.length].color} strokeWidth="2.2" strokeLinecap="round" strokeDasharray="7 93" style={{ animationDelay: `${-i * .43}s`, animationDuration: `${2.8 + i % 4 * .5}s`, animationDirection: i % 2 ? 'reverse' : 'normal' }} />
          <path className="research-link-signal" d={edge.d} pathLength="100" stroke={nodes[(i + 1) % nodes.length].color} strokeWidth="1.5" strokeOpacity=".6" strokeLinecap="round" strokeDasharray="4 96" style={{ animationDelay: `${-i * .61 - 1.2}s`, animationDuration: `${3.5 + i % 4 * .5}s`, animationDirection: i % 2 ? 'normal' : 'reverse' }} />
        </g>)}
        <circle cx="180" cy="130" r="63" stroke={`url(#${gradient})`} strokeOpacity=".2" strokeDasharray="2 8" />
        <g className="research-orbit"><circle cx="180" cy="130" r="58" stroke={`url(#${gradient})`} strokeOpacity=".5" strokeDasharray="18 112" /><circle cx="180" cy="72" r="3" fill="#a5f3fc" /></g>
        <g className="research-orbit research-orbit-reverse"><circle cx="180" cy="130" r="67" stroke="#a78bfa" strokeOpacity=".2" strokeDasharray="3 24" /><circle cx="247" cy="130" r="2" fill="#c4b5fd" /></g>
        {nodes.map((node, i) => <g key={node.id} data-node={node.id}>
          <circle className="research-node-aura" cx={node.x} cy={node.y} r="25" stroke={node.color} strokeOpacity=".5" style={{ animationDelay: `${-i * .8}s` }} />
          <circle cx={node.x} cy={node.y} r="18" fill="#0f172a" stroke={node.color} strokeOpacity=".75" />
          <circle className="research-node-core" cx={node.x} cy={node.y} r="5" fill={node.color} style={{ animationDelay: `${-i * .65}s` }} />
          <text className="research-node-label" x={node.x} y={node.y + 38} textAnchor="middle" fill="#e2e8f0" fontSize="12">{node.label}</text>
        </g>)}
        <circle className="research-hub-aura" cx="180" cy="130" r="47" stroke={`url(#${gradient})`} strokeOpacity=".6" />
        <circle cx="180" cy="130" r="41" fill="#0f172a" stroke={`url(#${gradient})`} strokeWidth="1.5" />
        <text x="180" y="134" textAnchor="middle" fill="#cffafe" fontSize="13" fontWeight="600" letterSpacing="1">SUANLAB</text>
      </svg>
    </div>
  );
}
