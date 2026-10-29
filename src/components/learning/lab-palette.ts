import type { LabColor } from './lab-model';
export interface LabPalette extends Record<LabColor, string> { grid: string; label: string; }

/** Resolve inherited site tokens to CSS colors supported by the WebGL renderer. */
export function readLabPalette(element: HTMLElement): LabPalette {
  const style = getComputedStyle(element);
  const context = document.createElement('canvas').getContext('2d');
  const color = (value: string) => {
    if (!context) return style.color;
    context.fillStyle = value;
    return context.fillStyle;
  };
  return {
    blue: color(style.getPropertyValue('--learning-blue').trim()),
    orange: color(style.getPropertyValue('--learning-orange').trim()),
    green: color(style.getPropertyValue('--learning-green').trim()),
    neutral: color(style.getPropertyValue('--learning-neutral').trim()),
    grid: color(style.getPropertyValue('--learning-grid').trim()),
    label: color(style.color),
  };
}
