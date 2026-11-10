import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { BlockOrderSpatialModel } from '@/lib/learning-block-order-view';
import type { LabPalette } from './lab-palette';

/** Physical site positions; heights are fixed to spin units, with no time integration. */
export function createBlockOrderSpatialPlot(host: HTMLElement,initial: BlockOrderSpatialModel,palette: LabPalette,label: string,onFailure: () => void) {
  const renderer = new THREE.WebGLRenderer({ antialias: true,alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1,1.75)); renderer.setClearColor(0,0);
  const canvas = renderer.domElement; canvas.tabIndex = 0; canvas.setAttribute('role','img'); canvas.setAttribute('aria-label',label);
  canvas.style.cssText = 'width:100%;height:100%;display:block;touch-action:pan-y'; host.append(canvas);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(42,1,.1,80), controls = new OrbitControls(camera,canvas);
  controls.enableDamping = false; controls.enablePan = false; controls.enableZoom = false;
  let content = new THREE.Group(), disposed = false; scene.add(content);
  let radius = initial.points.some(point => point.z !== 0) ? 5.5 : 4.1;
  const fit = () => {
    const vertical = camera.fov*Math.PI/360, angle = Math.min(vertical,Math.atan(Math.tan(vertical)*camera.aspect));
    const direction = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(direction.multiplyScalar(radius/Math.sin(angle)*1.07).add(controls.target));
  };
  const render = () => { if (!disposed) renderer.render(scene,camera); };
  const release = (group: THREE.Group) => group.traverse(node => { const mesh = node as THREE.Mesh; mesh.geometry?.dispose(); (Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : []).forEach(m => m.dispose()); });
  function update(model: BlockOrderSpatialModel,p: LabPalette) {
    radius = model.points.some(point => point.z !== 0) ? 5.5 : 4.1;
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const line = (points: THREE.Vector3[],color: string) => content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({ color })));
    line([new THREE.Vector3(-3.6,0,0),new THREE.Vector3(3.6,0,0)],p.orange);
    if (model.points.some(point => point.z !== 0)) line([new THREE.Vector3(0,0,-3.6),new THREE.Vector3(0,0,3.6)],p.neutral);
    line([new THREE.Vector3(-3.6,-1.6,0),new THREE.Vector3(-3.6,1.6,0)],p.blue);
    const groups = new Set(model.points.map(point => point.block));
    for (const block of groups) {
      const points = model.points.filter(point => point.block === block), xs = points.map(point => point.x), zs = points.map(point => point.z), pad = Math.min(.25,model.spacing*.35);
      const x0 = Math.min(...xs)-pad, x1 = Math.max(...xs)+pad, z0 = Math.min(...zs)-pad, z1 = Math.max(...zs)+pad;
      line([[x0,z0],[x1,z0],[x1,z1],[x0,z1],[x0,z0]].map(([x,z]) => new THREE.Vector3(x,-.03,z)),p.grid);
    }
    for (const [i,j] of model.edges) {
      const a = model.points[i], b = model.points[j], start = new THREE.Vector3(a.x,0,a.z), end = new THREE.Vector3(b.x,0,b.z);
      if (j < i) {
        const bend = a.z === b.z ? new THREE.Vector3((a.x+b.x)/2,0,a.z-.6) : new THREE.Vector3(a.x-.6,0,(a.z+b.z)/2);
        line(new THREE.QuadraticBezierCurve3(start,bend,end).getPoints(24),p.neutral);
      } else line([start,end],p.neutral);
    }
    for (const point of model.points) {
      if (point.value === null) continue;
      const value = point.value, color = value < 0 ? p.orange : value > 0 ? p.blue : p.neutral;
      if (value !== 0) {
        const bar = new THREE.Mesh(new THREE.CylinderGeometry(.045,.045,Math.abs(value)*1.6,10),new THREE.MeshBasicMaterial({ color }));
        bar.position.set(point.x,value*.8,point.z); content.add(bar);
      }
      const dot = new THREE.Mesh(new THREE.SphereGeometry(.09,12,8),new THREE.MeshBasicMaterial({ color })); dot.position.set(point.x,value*1.6,point.z); content.add(dot);
    }
    fit(); controls.update(); render();
  }
  function view(mode: 'perspective' | 'front') { camera.position.set(...(mode === 'front' ? [0,0,11] : [6,5,9]) as [number,number,number]); controls.target.set(0,0,0); fit(); controls.update(); render(); }
  const key = (event: KeyboardEvent) => {
    if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(event.key)) return;
    event.preventDefault(); if (event.key === 'Home') { view('perspective'); return; }
    const s = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    s.theta += (event.key === 'ArrowLeft' ? 1 : event.key === 'ArrowRight' ? -1 : 0)*Math.PI/18;
    s.phi = Math.max(.1,Math.min(Math.PI-.1,s.phi+(event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0)*Math.PI/18));
    camera.position.setFromSpherical(s).add(controls.target); controls.update(); render();
  };
  canvas.addEventListener('keydown',key); controls.addEventListener('change',render); view('perspective');
  const resize = new ResizeObserver(() => { const { width,height } = host.getBoundingClientRect(); if (disposed || !width || !height) return; camera.aspect = width/height; camera.updateProjectionMatrix(); fit(); controls.update(); renderer.setSize(width,height,false); render(); }); resize.observe(host);
  const lost = (event: Event) => { event.preventDefault(); onFailure(); }; canvas.addEventListener('webglcontextlost',lost); update(initial,palette);
  return { update,view,dispose() { if (disposed) return; disposed = true; resize.disconnect(); canvas.removeEventListener('keydown',key); canvas.removeEventListener('webglcontextlost',lost); controls.removeEventListener('change',render); controls.dispose(); release(content); renderer.dispose(); renderer.forceContextLoss(); canvas.remove(); } };
}
export type BlockOrderSpatialRenderer = ReturnType<typeof createBlockOrderSpatialPlot>;
