// TwinTeleop digital-twin viewer v0.
//
// Loads viewer/sample_timeseries.json (produced by
// tools/export_timeseries.py) and renders a 2-link planar arm whose joint
// angles follow the series. This is a v0: two links, fixed link lengths,
// no InstancedMesh batching and no 60 FPS tuning yet — those stay on the
// roadmap until they are measured, not claimed.
import * as THREE from "three";

const LINK1 = 2.0;
const LINK2 = 1.5;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / (window.innerHeight - 90), 0.1, 100);
camera.position.set(0, 3.5, 6);
camera.lookAt(0, 1.2, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight - 90);
document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(3, 6, 4);
scene.add(light);

const mat = new THREE.MeshStandardMaterial({ color: 0x58a6ff });
const base = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.45, 0.3, 24), mat);
base.position.y = 0.15;
scene.add(base);

function linkMesh(len) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.BoxGeometry(0.22, len, 0.22), mat);
  m.position.y = len / 2;
  g.add(m);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), mat);
  tip.position.y = len;
  g.add(tip);
  return g;
}
const shoulder = new THREE.Group();
shoulder.position.y = 0.3;
const link1 = linkMesh(LINK1);
const elbow = new THREE.Group();
elbow.position.y = LINK1;
const link2 = linkMesh(LINK2);
elbow.add(link2);
link1.add(elbow);
shoulder.add(link1);
scene.add(shoulder);

const grid = new THREE.GridHelper(10, 20, 0x30363d, 0x21262d);
scene.add(grid);

const playBtn = document.getElementById("play");
const scrub = document.getElementById("scrub");
const readout = document.getElementById("readout");
let series = null;
let idx = 0;
let playing = true;
let last = performance.now();

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});
scrub.addEventListener("input", () => {
  idx = Math.floor((scrub.value / 100) * (series.joints[0].t_s.length - 1));
  playing = false;
  playBtn.textContent = "Play";
});

function frame() {
  requestAnimationFrame(frame);
  if (!series) return;
  const n = series.joints[0].t_s.length;
  if (playing) {
    const now = performance.now();
    const dt = (now - last) / 1000;
    last = now;
    // Advance one sample per rendered frame slice is wrong for wall-clock
    // fidelity; instead step by elapsed time mapped to the series clock.
    const tEnd = series.joints[0].t_s[n - 1];
    const tNow = (series.joints[0].t_s[idx] + dt) % (tEnd + 1e-9);
    idx = series.joints[0].t_s.findIndex((t) => t >= tNow);
    if (idx < 0) idx = n - 1;
  } else {
    last = performance.now();
  }
  const j0 = series.joints[0].position[idx];
  const j1 = series.joints.length > 1 ? series.joints[1].position[idx] : 0;
  // Planar arm in the XY plane: shoulder rotates about Z, elbow relative.
  shoulder.rotation.z = -j0;
  elbow.rotation.z = -j1;
  scrub.value = Math.round((idx / (n - 1)) * 100);
  readout.textContent =
    `t=${series.joints[0].t_s[idx].toFixed(2)}s  j0=${j0.toFixed(3)} rad  ` +
    `j1=${j1.toFixed(3)} rad  source=${series.source}`;
  renderer.render(scene, camera);
}

fetch("./sample_timeseries.json")
  .then((r) => r.json())
  .then((data) => {
    series = data;
    scrub.max = 100;
    frame();
  })
  .catch((e) => {
    readout.textContent = "Failed to load sample_timeseries.json: " + e;
  });
