'use client';

import Link from 'next/link';
import { ChevronRight, Home } from 'lucide-react';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  backgroundImage?: string;
  breadcrumbs?: { label: string; href?: string }[];
}

export default function PageHeader({
  title,
  subtitle,
  backgroundImage = '/assets/images/slider/blue.jpg',
  breadcrumbs = [],
}: PageHeaderProps) {
  return (
    <section
      className="relative py-16 md:py-20 lg:py-24 text-white overflow-hidden"
      style={{
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}
    >
      {/* Dark Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950/95 via-slate-900/90 to-blue-950/80" />

      {/* Animated Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Floating Particles */}
        <div className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-particle-1 opacity-60" style={{ left: '10%', top: '20%' }} />
        <div className="absolute w-1.5 h-1.5 bg-blue-400 rounded-full animate-particle-2 opacity-50" style={{ left: '20%', top: '60%' }} />
        <div className="absolute w-1 h-1 bg-purple-400 rounded-full animate-particle-3 opacity-70" style={{ left: '80%', top: '30%' }} />
        <div className="absolute w-2 h-2 bg-cyan-300 rounded-full animate-particle-4 opacity-40" style={{ left: '70%', top: '70%' }} />
        <div className="absolute w-1 h-1 bg-indigo-400 rounded-full animate-particle-1 opacity-60" style={{ left: '50%', top: '15%' }} />
        <div className="absolute w-1.5 h-1.5 bg-teal-400 rounded-full animate-particle-2 opacity-50" style={{ left: '85%', top: '50%' }} />
        <div className="absolute w-1 h-1 bg-blue-300 rounded-full animate-particle-3 opacity-60" style={{ left: '15%', top: '80%' }} />
        <div className="absolute w-1 h-1 bg-violet-400 rounded-full animate-particle-4 opacity-50" style={{ left: '40%', top: '40%' }} />
      </div>

      {/* Neural Network / Data Flow Background */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Animated Data Streams - More visible */}
        <div className="absolute top-0 left-[15%] w-0.5 h-full overflow-hidden">
          <div className="w-full h-20 bg-gradient-to-b from-transparent via-cyan-400 to-transparent animate-stream-down" />
        </div>
        <div className="absolute top-0 left-[35%] w-0.5 h-full overflow-hidden">
          <div className="w-full h-32 bg-gradient-to-b from-transparent via-blue-400 to-transparent animate-stream-down animation-delay-2000" />
        </div>
        <div className="absolute top-0 left-[55%] w-0.5 h-full overflow-hidden">
          <div className="w-full h-24 bg-gradient-to-b from-transparent via-purple-400 to-transparent animate-stream-down animation-delay-4000" />
        </div>
        <div className="absolute top-0 left-[75%] w-0.5 h-full overflow-hidden">
          <div className="w-full h-16 bg-gradient-to-b from-transparent via-indigo-400 to-transparent animate-stream-down" style={{ animationDelay: '1s' }} />
        </div>
        <div className="absolute top-0 left-[90%] w-0.5 h-full overflow-hidden">
          <div className="w-full h-28 bg-gradient-to-b from-transparent via-teal-400 to-transparent animate-stream-down" style={{ animationDelay: '3s' }} />
        </div>

        {/* Deep Neural Network Structure */}
        <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="xMidYMid meet">
          <defs>
            {/* Gradients for connections */}
            <linearGradient id="conn-input-hidden1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(34,211,238)" stopOpacity="0.6" />
              <stop offset="100%" stopColor="rgb(59,130,246)" stopOpacity="0.3" />
            </linearGradient>
            <linearGradient id="conn-hidden1-hidden2" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(59,130,246)" stopOpacity="0.5" />
              <stop offset="100%" stopColor="rgb(139,92,246)" stopOpacity="0.3" />
            </linearGradient>
            <linearGradient id="conn-hidden2-output" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(139,92,246)" stopOpacity="0.5" />
              <stop offset="100%" stopColor="rgb(236,72,153)" stopOpacity="0.4" />
            </linearGradient>
            {/* Signal flow gradients */}
            <linearGradient id="signal1">
              <stop offset="0%" stopColor="rgb(34,211,238)" stopOpacity="0" />
              <stop offset="50%" stopColor="rgb(34,211,238)" stopOpacity="1" />
              <stop offset="100%" stopColor="rgb(59,130,246)" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="signal2">
              <stop offset="0%" stopColor="rgb(59,130,246)" stopOpacity="0" />
              <stop offset="50%" stopColor="rgb(139,92,246)" stopOpacity="1" />
              <stop offset="100%" stopColor="rgb(139,92,246)" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="signal3">
              <stop offset="0%" stopColor="rgb(139,92,246)" stopOpacity="0" />
              <stop offset="50%" stopColor="rgb(236,72,153)" stopOpacity="1" />
              <stop offset="100%" stopColor="rgb(236,72,153)" stopOpacity="0" />
            </linearGradient>
            {/* Node glow filters */}
            <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-violet" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-pink" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>

          {/* Layer Labels */}
          <text x="8%" y="12%" fill="rgba(34,211,238,0.4)" fontSize="10" fontFamily="monospace" className="hidden md:block">Input</text>
          <text x="28%" y="12%" fill="rgba(59,130,246,0.4)" fontSize="10" fontFamily="monospace" className="hidden md:block">Hidden 1</text>
          <text x="48%" y="12%" fill="rgba(139,92,246,0.4)" fontSize="10" fontFamily="monospace" className="hidden md:block">Hidden 2</text>
          <text x="68%" y="12%" fill="rgba(168,85,247,0.4)" fontSize="10" fontFamily="monospace" className="hidden md:block">Hidden 3</text>
          <text x="88%" y="12%" fill="rgba(236,72,153,0.4)" fontSize="10" fontFamily="monospace" className="hidden md:block">Output</text>

          {/* ===== CONNECTION LINES ===== */}
          {/* Input to Hidden1 connections */}
          <g className="opacity-30">
            {/* From input node 1 */}
            <line x1="10%" y1="25%" x2="30%" y2="20%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="25%" x2="30%" y2="35%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="25%" x2="30%" y2="50%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="25%" x2="30%" y2="65%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="25%" x2="30%" y2="80%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            {/* From input node 2 */}
            <line x1="10%" y1="40%" x2="30%" y2="20%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="40%" x2="30%" y2="35%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="40%" x2="30%" y2="50%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="40%" x2="30%" y2="65%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="40%" x2="30%" y2="80%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            {/* From input node 3 */}
            <line x1="10%" y1="55%" x2="30%" y2="20%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="55%" x2="30%" y2="35%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="55%" x2="30%" y2="50%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="55%" x2="30%" y2="65%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="55%" x2="30%" y2="80%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            {/* From input node 4 */}
            <line x1="10%" y1="70%" x2="30%" y2="20%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="70%" x2="30%" y2="35%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="70%" x2="30%" y2="50%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="70%" x2="30%" y2="65%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="70%" x2="30%" y2="80%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            {/* From input node 5 */}
            <line x1="10%" y1="85%" x2="30%" y2="20%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="85%" x2="30%" y2="35%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="85%" x2="30%" y2="50%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="85%" x2="30%" y2="65%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
            <line x1="10%" y1="85%" x2="30%" y2="80%" stroke="url(#conn-input-hidden1)" strokeWidth="1" />
          </g>

          {/* Hidden1 to Hidden2 connections */}
          <g className="opacity-25">
            <line x1="30%" y1="20%" x2="50%" y2="30%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="20%" x2="50%" y2="50%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="20%" x2="50%" y2="70%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="35%" x2="50%" y2="30%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="35%" x2="50%" y2="50%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="35%" x2="50%" y2="70%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="50%" x2="50%" y2="30%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="50%" x2="50%" y2="50%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="50%" x2="50%" y2="70%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="65%" x2="50%" y2="30%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="65%" x2="50%" y2="50%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="65%" x2="50%" y2="70%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="80%" x2="50%" y2="30%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="80%" x2="50%" y2="50%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
            <line x1="30%" y1="80%" x2="50%" y2="70%" stroke="url(#conn-hidden1-hidden2)" strokeWidth="1" />
          </g>

          {/* Hidden2 to Hidden3 connections */}
          <g className="opacity-25">
            <line x1="50%" y1="30%" x2="70%" y2="35%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="30%" x2="70%" y2="50%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="30%" x2="70%" y2="65%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="50%" x2="70%" y2="35%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="50%" x2="70%" y2="50%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="50%" x2="70%" y2="65%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="70%" x2="70%" y2="35%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="70%" x2="70%" y2="50%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="50%" y1="70%" x2="70%" y2="65%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
          </g>

          {/* Hidden3 to Output connections */}
          <g className="opacity-30">
            <line x1="70%" y1="35%" x2="90%" y2="40%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="70%" y1="35%" x2="90%" y2="60%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="70%" y1="50%" x2="90%" y2="40%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="70%" y1="50%" x2="90%" y2="60%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="70%" y1="65%" x2="90%" y2="40%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
            <line x1="70%" y1="65%" x2="90%" y2="60%" stroke="url(#conn-hidden2-output)" strokeWidth="1" />
          </g>

          {/* ===== ANIMATED SIGNAL FLOWS ===== */}
          {/* Signal flowing through network */}
          <line x1="10%" y1="40%" x2="30%" y2="35%" stroke="url(#signal1)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" />
          <line x1="10%" y1="55%" x2="30%" y2="50%" stroke="url(#signal1)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '0.3s' }} />
          <line x1="10%" y1="70%" x2="30%" y2="65%" stroke="url(#signal1)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '0.6s' }} />

          <line x1="30%" y1="35%" x2="50%" y2="30%" stroke="url(#signal2)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '0.5s' }} />
          <line x1="30%" y1="50%" x2="50%" y2="50%" stroke="url(#signal2)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '0.8s' }} />
          <line x1="30%" y1="65%" x2="50%" y2="70%" stroke="url(#signal2)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '1.1s' }} />

          <line x1="50%" y1="30%" x2="70%" y2="35%" stroke="url(#signal2)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '1s' }} />
          <line x1="50%" y1="50%" x2="70%" y2="50%" stroke="url(#signal2)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '1.3s' }} />
          <line x1="50%" y1="70%" x2="70%" y2="65%" stroke="url(#signal2)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '1.6s' }} />

          <line x1="70%" y1="35%" x2="90%" y2="40%" stroke="url(#signal3)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '1.5s' }} />
          <line x1="70%" y1="50%" x2="90%" y2="50%" stroke="url(#signal3)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '1.8s' }} />
          <line x1="70%" y1="65%" x2="90%" y2="60%" stroke="url(#signal3)" strokeWidth="2" strokeDasharray="8 80" className="animate-signal-flow" style={{ animationDelay: '2.1s' }} />

          {/* ===== NEURAL NETWORK NODES ===== */}
          {/* Input Layer (5 nodes) - Cyan */}
          <g filter="url(#glow-cyan)">
            <circle cx="10%" cy="25%" r="6" fill="rgb(34,211,238)" className="animate-node-pulse" style={{ animationDelay: '0s' }} />
            <circle cx="10%" cy="40%" r="6" fill="rgb(34,211,238)" className="animate-node-pulse" style={{ animationDelay: '0.2s' }} />
            <circle cx="10%" cy="55%" r="6" fill="rgb(34,211,238)" className="animate-node-pulse" style={{ animationDelay: '0.4s' }} />
            <circle cx="10%" cy="70%" r="6" fill="rgb(34,211,238)" className="animate-node-pulse" style={{ animationDelay: '0.6s' }} />
            <circle cx="10%" cy="85%" r="6" fill="rgb(34,211,238)" className="animate-node-pulse" style={{ animationDelay: '0.8s' }} />
          </g>

          {/* Hidden Layer 1 (5 nodes) - Blue */}
          <g filter="url(#glow-blue)">
            <circle cx="30%" cy="20%" r="5" fill="rgb(59,130,246)" className="animate-node-pulse" style={{ animationDelay: '0.3s' }} />
            <circle cx="30%" cy="35%" r="5" fill="rgb(59,130,246)" className="animate-node-pulse" style={{ animationDelay: '0.5s' }} />
            <circle cx="30%" cy="50%" r="5" fill="rgb(59,130,246)" className="animate-node-pulse" style={{ animationDelay: '0.7s' }} />
            <circle cx="30%" cy="65%" r="5" fill="rgb(59,130,246)" className="animate-node-pulse" style={{ animationDelay: '0.9s' }} />
            <circle cx="30%" cy="80%" r="5" fill="rgb(59,130,246)" className="animate-node-pulse" style={{ animationDelay: '1.1s' }} />
          </g>

          {/* Hidden Layer 2 (3 nodes) - Violet */}
          <g filter="url(#glow-violet)">
            <circle cx="50%" cy="30%" r="5" fill="rgb(139,92,246)" className="animate-node-pulse" style={{ animationDelay: '0.6s' }} />
            <circle cx="50%" cy="50%" r="5" fill="rgb(139,92,246)" className="animate-node-pulse" style={{ animationDelay: '0.9s' }} />
            <circle cx="50%" cy="70%" r="5" fill="rgb(139,92,246)" className="animate-node-pulse" style={{ animationDelay: '1.2s' }} />
          </g>

          {/* Hidden Layer 3 (3 nodes) - Purple */}
          <g filter="url(#glow-violet)">
            <circle cx="70%" cy="35%" r="5" fill="rgb(168,85,247)" className="animate-node-pulse" style={{ animationDelay: '0.9s' }} />
            <circle cx="70%" cy="50%" r="5" fill="rgb(168,85,247)" className="animate-node-pulse" style={{ animationDelay: '1.2s' }} />
            <circle cx="70%" cy="65%" r="5" fill="rgb(168,85,247)" className="animate-node-pulse" style={{ animationDelay: '1.5s' }} />
          </g>

          {/* Output Layer (2 nodes) - Pink */}
          <g filter="url(#glow-pink)">
            <circle cx="90%" cy="40%" r="6" fill="rgb(236,72,153)" className="animate-node-pulse" style={{ animationDelay: '1.2s' }} />
            <circle cx="90%" cy="60%" r="6" fill="rgb(236,72,153)" className="animate-node-pulse" style={{ animationDelay: '1.5s' }} />
          </g>

          {/* Ripple effects on active nodes */}
          <circle cx="10%" cy="40%" r="6" fill="none" stroke="rgb(34,211,238)" strokeWidth="1" className="animate-ripple" style={{ animationDelay: '0s' }} />
          <circle cx="30%" cy="50%" r="5" fill="none" stroke="rgb(59,130,246)" strokeWidth="1" className="animate-ripple" style={{ animationDelay: '0.5s' }} />
          <circle cx="50%" cy="50%" r="5" fill="none" stroke="rgb(139,92,246)" strokeWidth="1" className="animate-ripple" style={{ animationDelay: '1s' }} />
          <circle cx="70%" cy="50%" r="5" fill="none" stroke="rgb(168,85,247)" strokeWidth="1" className="animate-ripple" style={{ animationDelay: '1.5s' }} />
          <circle cx="90%" cy="50%" r="6" fill="none" stroke="rgb(236,72,153)" strokeWidth="1" className="animate-ripple" style={{ animationDelay: '2s' }} />
        </svg>

      </div>

      {/* Circuit Grid Pattern */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0v60M0 30h60M15 15h30v30H15z' fill='none' stroke='%2306b6d4' stroke-width='0.5'/%3E%3Ccircle cx='30' cy='30' r='3' fill='none' stroke='%2306b6d4' stroke-width='0.5'/%3E%3C/svg%3E")`,
          backgroundSize: '60px 60px',
        }}
      />

      {/* Pulsing Core - AI Brain Effect */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] opacity-25 pointer-events-none">
        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-500/50 to-blue-500/50 blur-3xl animate-core-pulse" />
        <div className="absolute inset-12 rounded-full bg-gradient-to-r from-blue-500/60 to-purple-500/60 blur-2xl animate-core-pulse animation-delay-2000" />
        <div className="absolute inset-24 rounded-full bg-gradient-to-r from-purple-500/70 to-pink-500/70 blur-xl animate-core-pulse animation-delay-4000" />
      </div>

      {/* Tech Circuit Lines */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-500/60 to-transparent animate-pulse" />
      <div className="absolute bottom-0.5 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-blue-500/40 to-transparent" />

      {/* Content */}
      <div className="container relative z-10">
        <div className="max-w-3xl">
          {/* Breadcrumbs - Tech Style */}
          {breadcrumbs.length > 0 && (
            <nav className="mb-6 inline-flex items-center space-x-2 text-sm bg-slate-900/60 backdrop-blur-md rounded-lg px-4 py-2 border border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.1)]">
              <Link
                href="/"
                className="flex items-center text-cyan-400/80 hover:text-cyan-300 transition-colors"
              >
                <Home className="h-4 w-4" />
              </Link>
              {breadcrumbs.map((crumb, index) => (
                <span key={index} className="flex items-center">
                  <ChevronRight className="h-4 w-4 text-slate-600" />
                  {crumb.href ? (
                    <Link
                      href={crumb.href}
                      className="ml-2 text-slate-400 hover:text-cyan-300 transition-colors"
                    >
                      {crumb.label}
                    </Link>
                  ) : (
                    <span className="ml-2 text-cyan-300 font-medium">{crumb.label}</span>
                  )}
                </span>
              ))}
            </nav>
          )}

          {/* Title with Tech Gradient */}
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl lg:text-6xl">
            <span className="inline-block bg-gradient-to-r from-white via-cyan-100 to-blue-200 bg-clip-text text-transparent drop-shadow-[0_0_30px_rgba(6,182,212,0.3)]">
              {title}
            </span>
          </h1>

          {/* Subtitle */}
          {subtitle && (
            <p className="mt-4 text-lg md:text-xl text-slate-400 max-w-2xl animate-fade-in-up">
              {subtitle}
            </p>
          )}

          {/* Tech Accent Line - Animated */}
          <div className="mt-6 flex items-center gap-2">
            <div className="h-0.5 w-2 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(34,211,238,0.6)] animate-pulse" />
            <div className="h-0.5 w-8 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full shadow-[0_0_10px_rgba(6,182,212,0.5)] animate-pulse animation-delay-2000" />
            <div className="h-0.5 w-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full shadow-[0_0_12px_rgba(96,165,250,0.4)] animate-pulse animation-delay-4000" />
            <div className="h-0.5 w-4 bg-purple-400/50 rounded-full animate-pulse" />
          </div>
        </div>
      </div>
    </section>
  );
}
