import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { RegularPerturbationView } from '@/lib/learning-regular-perturbation-view';
import type { LabPalette } from './lab-palette';

/** Event-driven energy surface: displacement, coupling and actual energy. */
export function createPerturbationEnergyPlot(host: HTMLElement, initial: RegularPerturbationView, palette: LabPalette, label: string, onFailure: () => void) {
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
  function update(model: RegularPerturbationView, p: LabPalette) {
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const span = model.energyMax - model.energyMin || 1;
    const point = (x: number, energy: number, fraction: number) => new THREE.Vector3(3 * x / model.width, 3 * (energy - model.energyMin) / span - 1.5, 4 * fraction - 2);
    const line = (points: THREE.Vector3[], color: string) => content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color })));
    for (let j = 0; j < model.surface.length; j++) line(model.surface[j].map(p => point(p.x, p.energy, j / 20)), j === 20 ? p.orange : j === 0 ? p.blue : p.neutral);
    for (let i = 0; i < 41; i += 2) line(model.surface.map((row, j) => point(row[i].x, row[i].energy, j / 20)), p.green);
    line([new THREE.Vector3(-3, -1.5, -2), new THREE.Vector3(3, -1.5, -2)], p.blue);
    line([new THREE.Vector3(-3, -1.5, -2), new THREE.Vector3(-3, -1.5, 2)], p.green);
    line([new THREE.Vector3(-3, -1.5, -2), new THREE.Vector3(-3, 1.5, -2)], p.neutral);
    const root = new THREE.Mesh(new THREE.SphereGeometry(.09, 12, 8), new THREE.MeshBasicMaterial({ color: p.green }));
    root.position.copy(point(model.root, model.rootEnergy, 1)); content.add(root);
    if (model.partialInside) {
      const marker = new THREE.Mesh(new THREE.SphereGeometry(.09, 12, 8), new THREE.MeshBasicMaterial({ color: p.orange }));
      marker.position.copy(point(model.partial!, model.energy(model.partial!), 1)); content.add(marker);
    }
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
export type PerturbationEnergyRenderer = ReturnType<typeof createPerturbationEnergyPlot>;
