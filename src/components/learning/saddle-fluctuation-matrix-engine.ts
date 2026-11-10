import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { SaddleFluctuationMatrixView } from '@/lib/learning-saddle-fluctuation-view';
import type { LabPalette } from './lab-palette';

/** Event-driven discrete matrix bars; horizontal axes are time pair indices, not physical two-dimensional space. */
export function createSaddleFluctuationMatrixPlot(host: HTMLElement, initial: SaddleFluctuationMatrixView, palette: LabPalette, label: string, onFailure: () => void) {
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
  function update(model: SaddleFluctuationMatrixView, p: LabPalette) {
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const line=(points:THREE.Vector3[],color:string)=>content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color})));
    const spacing=6/Math.max(1,model.n-1),width=Math.min(.48,spacing*.6);
    for(const v of model.points){
      if(v.y===0){const dot=new THREE.Mesh(new THREE.SphereGeometry(.025,8,6),new THREE.MeshBasicMaterial({color:p.neutral}));dot.position.set(v.x,0,v.z);content.add(dot);}
      else{const bar=new THREE.Mesh(new THREE.BoxGeometry(width,Math.abs(v.y),width),new THREE.MeshBasicMaterial({color:v.y>0?p.blue:p.orange}));bar.position.set(v.x,v.y/2,v.z);content.add(bar);}
      if(v.selected){const marker=new THREE.Mesh(new THREE.SphereGeometry(.11,12,8),new THREE.MeshBasicMaterial({color:p.green}));marker.position.set(v.x,v.y,v.z);content.add(marker);}
    }
    line([new THREE.Vector3(-3,0,-3),new THREE.Vector3(3,0,-3)],p.orange);
    line([new THREE.Vector3(-3,0,-3),new THREE.Vector3(-3,0,3)],p.neutral);
    line([new THREE.Vector3(-3,-2.5,-3),new THREE.Vector3(-3,2.5,-3)],p.green);
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
export type SaddleFluctuationMatrixRenderer = ReturnType<typeof createSaddleFluctuationMatrixPlot>;
