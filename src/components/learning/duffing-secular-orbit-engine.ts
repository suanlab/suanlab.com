import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { DuffingOrbitView } from '@/lib/learning-duffing-secular-orbit-view';
import type { LabPalette } from './lab-palette';

/** Event-driven exact-reference trajectory in position, velocity and local time. */
export function createDuffingOrbitPlot(host: HTMLElement, initial: DuffingOrbitView, palette: LabPalette, label: string, onFailure: () => void) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75)); renderer.setClearColor(0, 0);
  const canvas = renderer.domElement; canvas.tabIndex = 0; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', label);
  canvas.style.cssText = 'display:block;width:100%;height:100%'; host.append(canvas);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(42, 1, .1, 100), controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false; controls.enablePan = false; controls.enableZoom = false; canvas.style.touchAction = 'pan-y';
  let content = new THREE.Group(), disposed = false; scene.add(content);
  const render = () => { if (!disposed) renderer.render(scene, camera); };
  let fitting=false;
  const fit=()=>{
    if(fitting)return;fitting=true;
    const half=Math.min(camera.fov*Math.PI/360,Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect));camera.position.normalize().multiplyScalar(4.8/Math.sin(half));
    const box=new THREE.Box3().setFromObject(content);
    if(!box.isEmpty())for(let i=0;i<5;i++){
      camera.updateMatrixWorld();let extent=0;
      for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){const p=new THREE.Vector3(x,y,z).project(camera);extent=Math.max(extent,Math.abs(p.x),Math.abs(p.y));}
      if(!Number.isFinite(extent)||extent===0)break;
      camera.position.multiplyScalar(Math.max(.5,Math.min(2,extent/.82)));
      if(camera.position.length()<4.4)camera.position.setLength(4.4);
    }
    controls.update();fitting=false;
  };
  const release = (group: THREE.Group) => group.traverse(node => {
    const mesh = node as THREE.Mesh; mesh.geometry?.dispose();
    (Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : []).forEach(material => material.dispose());
  });
  function update(model: DuffingOrbitView, p: LabPalette) {
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const line=(points:THREE.Vector3[],color:string)=>content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color})));
    line(model.points.map(v=>new THREE.Vector3(v.x,v.y,v.z)),p.blue);
    const chosen=model.points[model.selected],marker=new THREE.Mesh(new THREE.SphereGeometry(.09,12,8),new THREE.MeshBasicMaterial({color:p.orange}));
    marker.position.set(chosen.x,chosen.y,chosen.z);content.add(marker);
    line([new THREE.Vector3(-2,0,0),new THREE.Vector3(2,0,0)],p.orange);
    line([new THREE.Vector3(0,-2,0),new THREE.Vector3(0,2,0)],p.green);
    line([new THREE.Vector3(0,0,-3),new THREE.Vector3(0,0,3)],p.neutral);
    fit(); render();
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
  const changed=()=>{if(!fitting){fit();render();}};
  resize.observe(host); controls.addEventListener('change', changed); canvas.addEventListener('keydown', key);
  const lost = (event: Event) => { event.preventDefault(); onFailure(); }; canvas.addEventListener('webglcontextlost', lost);
  view('perspective'); update(initial, palette);
  return { update, view, dispose() { if (disposed) return; disposed = true; resize.disconnect(); canvas.removeEventListener('keydown', key); canvas.removeEventListener('webglcontextlost', lost); controls.removeEventListener('change', changed); controls.dispose(); release(content); renderer.dispose(); renderer.forceContextLoss(); canvas.remove(); } };
}
export type DuffingOrbitRenderer = ReturnType<typeof createDuffingOrbitPlot>;
