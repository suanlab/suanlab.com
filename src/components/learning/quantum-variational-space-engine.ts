import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { RitzGeometry, RitzVector } from '@/lib/learning-quantum-variational-geometry';
import type { LabPalette } from './lab-palette';

export type RitzCameraView = 'perspective' | 'front' | 'top';
/** Fit the full reference axes and their labels, including portrait canvases. */
export function ritzSpaceCameraDistance(radius: number, aspect: number, verticalFov = 42) {
  const halfAngle = Math.atan(Math.tan(verticalFov * Math.PI / 360) * Math.min(1, aspect));
  return Math.max(radius * 3.8, Math.max(radius, 1.75) * 1.08 / Math.sin(halfAngle));
}
/** Camera interaction only. Geometry contains already-calculated observations;
 * this renderer performs no quantum evolution and has no animation loop. */
export function createRitzSpaceRenderer(host: HTMLElement, initial: RitzGeometry, palette: LabPalette, onFailure: () => void) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setClearColor(0, 0);
  const canvas = renderer.domElement;
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.width = '100%'; canvas.style.height = '100%'; canvas.style.display = 'block';
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false;
  controls.enablePan = false;
  controls.enableZoom = false;
  canvas.style.touchAction = 'pan-y';
  let content = new THREE.Group(), disposed = false, radius = 1.4;
  const position = (v: number[]) => new THREE.Vector3(v[0], v[2], -v[1]);
  const render = () => { if (!disposed) renderer.render(scene, camera); };
  function fitCamera() {
    const distance = ritzSpaceCameraDistance(radius, camera.aspect, camera.fov);
    camera.position.normalize().multiplyScalar(distance);
    camera.far = Math.max(100, distance + 4 * Math.max(radius, 1.75));
    camera.updateProjectionMatrix();
  }
  function release(group: THREE.Group) {
    group.traverse(node => {
      const mesh = node as THREE.Mesh;
      mesh.geometry?.dispose();
      const materials = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
      for (const material of materials) {
        (material as THREE.SpriteMaterial).map?.dispose();
        material.dispose();
      }
    });
  }
  function view(mode: RitzCameraView) {
    const direction = mode === 'front' ? [0, 0, 1] : mode === 'top' ? [0, 1, 0.001] : [2.7, 2.1, 3.2];
    camera.position.copy(new THREE.Vector3(...direction as RitzVector).normalize().multiplyScalar(radius * 3.8));
    fitCamera();
    controls.target.set(0, 0, 0);
    controls.update(); render();
  }
  function arrow(vector: number[], color: string) {
    const end = position(vector), length = end.length();
    if (length > 0) content.add(new THREE.ArrowHelper(end.clone().normalize(), new THREE.Vector3(), length, color, Math.min(0.12, length * 0.2), Math.min(0.06, length * 0.1)));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), new THREE.MeshBasicMaterial({ color }));
    dot.position.copy(end); content.add(dot);
  }
  function update(geometry: RitzGeometry, p: LabPalette) {
    if (disposed) return;
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const nextRadius = Math.max(1.4, Math.hypot(...geometry.residual) * 1.12);
    radius = nextRadius;
    fitCamera(); controls.update();
    const reference = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color: p.neutral, wireframe: true, transparent: true, opacity: 0.15 }));
    content.add(reference);
    for (let i = 0; i < 3; i++) {
      const v = [0, 0, 0]; v[i] = 1.3;
      const end = position(v);
      content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([end.clone().negate(), end]), new THREE.LineBasicMaterial({ color: p.neutral })));
      const label = document.createElement('canvas'); label.width = 128; label.height = 64;
      const ctx = label.getContext('2d');
      if (ctx) {
        ctx.font = 'bold 36px sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = p.label; ctx.fillText(`e${i}`, 64, 44);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(label), depthTest: false }));
        sprite.position.copy(end.multiplyScalar(1.12)); sprite.scale.set(0.48, 0.24, 1); content.add(sprite);
      }
    }
    if (geometry.columns === 1) {
      const end = position(geometry.axes[0]);
      content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([end.clone().negate(), end]), new THREE.LineBasicMaterial({ color: p.green })));
    } else if (geometry.columns === 2) {
      const [u, v] = geometry.axes.map(position), points = [new THREE.Vector3()];
      for (let i = 0; i <= 64; i++) points.push(u.clone().multiplyScalar(Math.cos(i * Math.PI / 32)).addScaledVector(v, Math.sin(i * Math.PI / 32)));
      const coordinates: number[] = [];
      for (let i = 1; i < points.length - 1; i++) for (const point of [points[0], points[i], points[i + 1]]) coordinates.push(point.x, point.y, point.z);
      const disk = new THREE.BufferGeometry(); disk.setAttribute('position', new THREE.Float32BufferAttribute(coordinates, 3));
      content.add(new THREE.Mesh(disk, new THREE.MeshBasicMaterial({ color: p.green, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false })));
      content.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points.slice(1, -1)), new THREE.LineBasicMaterial({ color: p.green })));
    } else {
      content.add(new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color: p.green, wireframe: true, transparent: true, opacity: 0.4 })));
    }
    arrow(geometry.state, p.blue); arrow(geometry.residual, p.orange); render();
  }
  const resize = new ResizeObserver(() => {
    if (disposed) return;
    const { width, height } = host.getBoundingClientRect();
    if (!width || !height) return;
    camera.aspect = width / height; camera.updateProjectionMatrix(); fitCamera(); controls.update(); renderer.setSize(width, height, false); render();
  });
  const lose = (event: Event) => { event.preventDefault(); if (!disposed) onFailure(); };
  function dispose() {
    if (disposed) return;
    disposed = true; resize.disconnect(); controls.removeEventListener('change', render); controls.dispose();
    canvas.removeEventListener('webglcontextlost', lose); release(content); scene.clear(); renderer.dispose(); renderer.forceContextLoss(); canvas.remove();
  }
  try {
    host.appendChild(canvas); scene.add(content); view('perspective');
    controls.addEventListener('change', render); canvas.addEventListener('webglcontextlost', lose); resize.observe(host);
    update(initial, palette);
  } catch (error) { dispose(); throw error; }
  return { update, view, dispose };
}
export type RitzSpaceRenderer = ReturnType<typeof createRitzSpaceRenderer>;
