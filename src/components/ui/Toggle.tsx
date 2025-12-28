'use client';

import { useState } from 'react';

interface ToggleProps {
  label: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export default function Toggle({ label, children, defaultOpen = false }: ToggleProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`toggle ${isOpen ? 'active' : ''}`}>
      <label onClick={() => setIsOpen(!isOpen)}>{label}</label>
      <div
        className="toggle-content"
        style={{ display: isOpen ? 'block' : 'none' }}
      >
        {children}
      </div>
    </div>
  );
}
