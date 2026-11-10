import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { QuarticAsymptoticView } from '@/lib/learning-quartic-asymptotic-view';
import { quarticSurfaceDepth } from '@/lib/learning-quartic-asymptotic-view';
import type { LabPalette } from './lab-palette';

/** Event-driven integrand surface: displacement, coupling and kernel height. */
export function createQuarticIntegrandPlot(host: HTMLElement, initial: QuarticAsymptoticView, palette: LabPalette, label: string, onFailure: () => void) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75)); renderer.setClearColor(0, 0);
  const canvas = renderer.domElement; canvas.tabIndex = 0; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', label);
  canvas.style.cssText = 'display:block;width:100%;height:100%'; host.append(canvas);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(42, 1, .1, 100), controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false; controls.enablePan = false; controls.enableZoom = false; canvas.style.touchAction = 'pan-y';
  let content = new THREE.Group(), disposed = false; scene.add(content);
  const render = () => { if (!disposed) renderer.render(scene, camera); };
  const fit = () => { const half = Math.min(camera.fov * Math.PI / 360, Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect)); camera.position.normalize().multiplyScalar(4.8 / Math.sin(half)); controls.update(); };
  const release = (group: THREE.Group) => group.traverse(node => {
    const mesh = node as THREE.Mesh; mesh.geometry?.dispose();
    (Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : []).forEach(material => material.dispose());
  });
  function update(model: QuarticAsymptoticView, p: LabPalette) {
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const point = (x: number, value: number, fraction: number) => new THREE.Vector3(3 * x / model.width, 3 * value - 1.5, quarticSurfaceDepth(model.coupling, fraction));
    const line = (points: THREE.Vector3[], color: string) => content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color })));
    for (let j = 0; j < model.surface.length; j++) line(model.surface[j].map(p => point(p.x, p.value, j / 20)), j === 20 ? p.orange : j === 0 ? p.blue : p.neutral);
    for (let i = 0; i < 41; i += 2) line(model.surface.map((row, j) => point(row[i].x, row[i].value, j / 20)), p.green);
    const startDepth = quarticSurfaceDepth(model.coupling, 0);
    line([new THREE.Vector3(-3, -1.5, startDepth), new THREE.Vector3(3, -1.5, startDepth)], p.blue);
    if (model.coupling > 0) line([new THREE.Vector3(-3, -1.5, -2), new THREE.Vector3(-3, -1.5, 2)], p.green);
    line([new THREE.Vector3(-3, -1.5, startDepth), new THREE.Vector3(-3, 1.5, startDepth)], p.neutral);
    render();
  }
  function view(mode: 'perspective' | 'front') { camera.position.set(...(mode === 'front' ? [0, 2, 12] : [7, 6, 9]) as [number, number, number]); controls.target.set(0, 0, 0); fit(); render(); }
  const key = (event: KeyboardEvent) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return;
    event.preventDefault(); if (event.key === 'Home') { view('perspective'); return; }
    const spherical = new THREE.Spherical().setFromVector3(camera.position);
    spherical.theta += (event.key === 'ArrowLeft' ? 1 : event.key === 'ArrowRight' ? -1 : 0) * Math.PI / 18;
    spherical.phi = Math.max(.1, Math.min(Math.PI - .1, spherical.phi + (event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0) * Math.PI / 18));
    camera.position.setFromSpherical(spherical); controls.update(); render();
  };
  const resize = new ResizeObserver(() => { const { width, height } = host.getBoundingClientRect(); if (disposed || !width || !height) return; camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height, false); fit(); render(); });
  resize.observe(host); controls.addEventListener('change', render); canvas.addEventListener('keydown', key);
  const lost = (event: Event) => { event.preventDefault(); onFailure(); }; canvas.addEventListener('webglcontextlost', lost);
  view('perspective'); update(initial, palette);
  return { update, view, dispose() { if (disposed) return; disposed = true; resize.disconnect(); canvas.removeEventListener('keydown', key); canvas.removeEventListener('webglcontextlost', lost); controls.removeEventListener('change', render); controls.dispose(); release(content); renderer.dispose(); renderer.forceContextLoss(); canvas.remove(); } };
}
export type QuarticIntegrandRenderer = ReturnType<typeof createQuarticIntegrandPlot>;
