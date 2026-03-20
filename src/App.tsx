/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChangeEvent, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- Types ---
type Mode = 'TREE' | 'SCATTER' | 'FOCUS';

interface State {
  mode: Mode;
  targetPhoto: Particle | null;
  handRotation: { x: number; y: number };
  lastGestureTime: number;
}

class Particle {
  type: string;
  mesh: THREE.Object3D;
  velocity: THREE.Vector3;
  treePos: THREE.Vector3;
  scatterPos: THREE.Vector3;
  targetPos: THREE.Vector3;
  targetScale: THREE.Vector3;

  constructor(type: string, texture: THREE.Texture | null, geoBox: THREE.BoxGeometry, geoSphere: THREE.SphereGeometry, materials: any) {
    this.type = type;
    this.mesh = this.createMesh(type, texture, geoBox, geoSphere, materials);
    this.velocity = new THREE.Vector3((Math.random() - 0.5) * 0.2, (Math.random() - 0.5) * 0.2, (Math.random() - 0.5) * 0.2);
    
    this.treePos = new THREE.Vector3();
    this.scatterPos = new THREE.Vector3();
    this.targetPos = new THREE.Vector3();
    this.targetScale = new THREE.Vector3(1, 1, 1);
  }

  createMesh(type: string, texture: THREE.Texture | null, geoBox: THREE.BoxGeometry, geoSphere: THREE.SphereGeometry, materials: any) {
    let mesh: THREE.Object3D;
    if (type === 'BOX') {
      mesh = new THREE.Mesh(geoBox, Math.random() > 0.5 ? materials.matGoldStandard : materials.matGreenStandard);
    } else if (type === 'SPHERE') {
      mesh = new THREE.Mesh(geoSphere, Math.random() > 0.5 ? materials.matGoldPhysical : materials.matRedPhysical);
    } else if (type === 'CANDY') {
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0.5, 0),
        new THREE.Vector3(0.5, 1, 0), new THREE.Vector3(0.8, 0.5, 0)
      ]);
      const geoTube = new THREE.TubeGeometry(curve, 20, 0.15, 8, false);
      mesh = new THREE.Mesh(geoTube, materials.matCandy);
    } else if (type === 'PHOTO') {
      const group = new THREE.Group();
      const frameGeo = new THREE.BoxGeometry(3.2, 3.2, 0.2);
      const frame = new THREE.Mesh(frameGeo, materials.matGoldStandard);
      const photoGeo = new THREE.PlaneGeometry(2.8, 2.8);
      const photoMat = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.4 });
      const photo = new THREE.Mesh(photoGeo, photoMat);
      photo.position.z = 0.11;
      group.add(frame, photo);
      mesh = group;
    } else {
      mesh = new THREE.Object3D();
    }
    
    mesh.position.set((Math.random() - 0.5) * 50, (Math.random() - 0.5) * 50, (Math.random() - 0.5) * 50);
    mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
    return mesh;
  }

  calculatePositions(index: number, total: number) {
    const t = index / total;
    const maxRadius = 12;
    const radius = maxRadius * (1 - t);
    const angle = t * 50 * Math.PI;
    const y = -10 + (t * 25);
    this.treePos.set(Math.cos(angle) * radius, y, Math.sin(angle) * radius);

    const r = 8 + Math.random() * 12;
    const theta = Math.random() * 2 * Math.PI;
    const phi = Math.acos(2 * Math.random() - 1);
    this.scatterPos.set(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
  }

  update(mode: Mode, isTargetPhoto: boolean) {
    if (mode === 'TREE') {
      this.targetPos.copy(this.treePos);
      this.targetScale.set(1, 1, 1);
    } else if (mode === 'SCATTER') {
      this.targetPos.copy(this.scatterPos);
      this.targetScale.set(1, 1, 1);
      this.mesh.rotation.x += this.velocity.x;
      this.mesh.rotation.y += this.velocity.y;
    } else if (mode === 'FOCUS') {
      if (isTargetPhoto) {
        this.targetPos.set(0, 2, 35);
        this.targetScale.set(4.5, 4.5, 4.5);
        this.mesh.rotation.set(0, 0, 0);
      } else {
        this.targetPos.copy(this.scatterPos).multiplyScalar(1.5);
        this.targetScale.set(1, 1, 1);
      }
    }

    this.mesh.position.lerp(this.targetPos, 0.05);
    this.mesh.scale.lerp(this.targetScale, 0.05);
    
    if(mode === 'TREE' && this.type !== 'PHOTO') {
      this.mesh.rotation.y += 0.01;
    }
  }
}

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(true);
  const [uiHidden, setUiHidden] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const stateRef = useRef<State>({
    mode: 'TREE',
    targetPhoto: null,
    handRotation: { x: 0, y: 0 },
    lastGestureTime: 0
  });
  const lastVideoTimeRef = useRef(-1);

  const particlesRef = useRef<Particle[]>([]);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const mainGroupRef = useRef<THREE.Group | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const handLandmarkerRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // --- Setup Scene ---
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    const mainGroup = new THREE.Group();
    mainGroupRef.current = mainGroup;
    scene.add(mainGroup);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 2, 50);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    rendererRef.current = renderer;
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ReinhardToneMapping;
    renderer.toneMappingExposure = 2.2;
    containerRef.current.appendChild(renderer.domElement);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const pointLight = new THREE.PointLight(0xff8800, 2, 50);
    pointLight.position.set(0, 5, 0);
    scene.add(pointLight);

    const spotLightGold = new THREE.SpotLight(0xd4af37, 1200);
    spotLightGold.position.set(30, 40, 40);
    scene.add(spotLightGold);

    const spotLightBlue = new THREE.SpotLight(0x4488ff, 600);
    spotLightBlue.position.set(-30, 20, -30);
    scene.add(spotLightBlue);

    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.45,
      0.4,
      0.7
    );
    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // --- Materials ---
    const materials = {
      matGoldStandard: new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.8, roughness: 0.2 }),
      matGreenStandard: new THREE.MeshStandardMaterial({ color: 0x0a3b16, metalness: 0.2, roughness: 0.8 }),
      matGoldPhysical: new THREE.MeshPhysicalMaterial({ color: 0xd4af37, metalness: 1, roughness: 0.1, clearcoat: 1 }),
      matRedPhysical: new THREE.MeshPhysicalMaterial({ color: 0xaa0000, metalness: 0.3, roughness: 0.2, clearcoat: 1 }),
      matCandy: new THREE.MeshStandardMaterial({ 
        map: (() => {
          const cvs = document.createElement('canvas');
          cvs.width = 256; cvs.height = 256;
          const ctx = cvs.getContext('2d')!;
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, 256, 256);
          ctx.lineWidth = 40;
          ctx.strokeStyle = '#d00000';
          for (let i = -256; i < 512; i += 64) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i + 256, 256);
            ctx.stroke();
          }
          const tex = new THREE.CanvasTexture(cvs);
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
          tex.repeat.set(4, 1);
          return tex;
        })(), 
        roughness: 0.3 
      })
    };

    const createDefaultPhotoTexture = () => {
      const cvs = document.createElement('canvas');
      cvs.width = 512; cvs.height = 512;
      const ctx = cvs.getContext('2d')!;
      ctx.fillStyle = '#111111';
      ctx.fillRect(0, 0, 512, 512);
      ctx.strokeStyle = '#d4af37';
      ctx.lineWidth = 15;
      ctx.strokeRect(20, 20, 472, 472);
      ctx.fillStyle = '#fceea7';
      ctx.font = 'bold 60px Cinzel, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('JOYEUX', 256, 220);
      ctx.fillText('NOEL', 256, 290);
      const tex = new THREE.CanvasTexture(cvs);
      tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    };

    // --- Particles ---
    const geoBox = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    const geoSphere = new THREE.SphereGeometry(0.5, 32, 32);
    const totalMain = 1500;
    const defaultPhotoTex = createDefaultPhotoTexture();
    
    for (let i = 0; i < totalMain; i++) {
      let type = 'SPHERE';
      const rand = Math.random();
      if (rand < 0.4) type = 'BOX';
      else if (rand < 0.8) type = 'SPHERE';
      else if (rand < 0.95) type = 'CANDY';
      else type = 'PHOTO';
      
      const p = new Particle(type, type === 'PHOTO' ? defaultPhotoTex : null, geoBox, geoSphere, materials);
      particlesRef.current.push(p);
      mainGroup.add(p.mesh);
    }
    
    particlesRef.current.forEach((p, i) => p.calculatePositions(i, totalMain));

    // --- Dust ---
    const dustGeo = new THREE.BufferGeometry();
    const dustPos = [];
    for(let i=0; i<2500; i++) {
      dustPos.push((Math.random()-0.5)*60, (Math.random()-0.5)*60, (Math.random()-0.5)*60);
    }
    dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dustPos, 3));
    const dustMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.1, transparent: true, opacity: 0.6 });
    const dustSystem = new THREE.Points(dustGeo, dustMat);
    scene.add(dustSystem);

    // --- MediaPipe ---
    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        handLandmarkerRef.current = landmarker;

        await startCamera();
      } catch (err: any) {
        console.error("MediaPipe error:", err);
        setCameraError(err.message || "Permission denied");
        setLoading(false);
      }
    };

    const startCamera = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadeddata = () => {
              videoRef.current?.play().then(() => {
                setLoading(false);
                setCameraError(null);
              }).catch((e) => {
                console.error("Video play error:", e);
                setLoading(false);
              });
            };
          }
        } else {
          setLoading(false);
          setCameraError("Camera not supported on this device.");
        }
      } catch (err: any) {
        console.error("Camera error:", err);
        setCameraError(err.message || "Permission denied");
        setLoading(false);
      }
    };

    // Store startCamera in a ref so we can call it from the UI
    (window as any).retryCamera = startCamera;

    initMediaPipe();

    // --- Animation Loop ---
    const animate = () => {
      requestAnimationFrame(animate);

      if (handLandmarkerRef.current && videoRef.current && videoRef.current.readyState >= 2) {
        let results;
        if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = videoRef.current.currentTime;
          results = handLandmarkerRef.current.detectForVideo(videoRef.current, performance.now());
        }

        if (results) {
          if (results.landmarks && results.landmarks.length > 0) {
            const marks = results.landmarks[0];
          const wrist = marks[0];
          const thumb = marks[4];
          const index = marks[8];
          const middle = marks[12];
          const ring = marks[16];
          const pinky = marks[20];
          const palmCenter = marks[9];

          const dist = (p1: any, p2: any) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
          const avgFingersToWrist = (dist(index, wrist) + dist(middle, wrist) + dist(ring, wrist) + dist(pinky, wrist)) / 4;

          const now = performance.now();
          if (now - stateRef.current.lastGestureTime > 1000) {
            if (dist(thumb, index) < 0.05) {
              setMode('FOCUS');
              stateRef.current.lastGestureTime = now;
            } else if (avgFingersToWrist < 0.25) {
              setMode('TREE');
              stateRef.current.lastGestureTime = now;
            } else if (avgFingersToWrist > 0.4) {
              setMode('SCATTER');
              stateRef.current.lastGestureTime = now;
            }
          }

          stateRef.current.handRotation.y = (palmCenter.x - 0.5) * Math.PI * -1;
          stateRef.current.handRotation.x = (palmCenter.y - 0.5) * Math.PI * 0.5;
        } else {
          stateRef.current.handRotation.y = 0;
          stateRef.current.handRotation.x = 0;
        }
        }
      }

      mainGroup.rotation.y = THREE.MathUtils.lerp(mainGroup.rotation.y, stateRef.current.handRotation.y, 0.05);
      mainGroup.rotation.x = THREE.MathUtils.lerp(mainGroup.rotation.x, stateRef.current.handRotation.x, 0.05);

      dustSystem.rotation.y += 0.001;

      particlesRef.current.forEach(p => p.update(stateRef.current.mode, p === stateRef.current.targetPhoto));

      composer.render();
    };
    animate();

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      composer.setSize(window.innerWidth, window.innerHeight);
      bloomPass.resolution.set(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'h') {
        setUiHidden(prev => !prev);
      }
      if(e.key === '1') setMode('TREE');
      if(e.key === '2') setMode('SCATTER');
      if(e.key === '3') setMode('FOCUS');
    };
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('keydown', handleKeyDown);
      renderer.dispose();
    };
  }, []);

  const setMode = (newMode: Mode) => {
    if (stateRef.current.mode === newMode) return;
    stateRef.current.mode = newMode;
    if (newMode === 'FOCUS') {
      const photos = particlesRef.current.filter(p => p.type === 'PHOTO');
      stateRef.current.targetPhoto = photos[Math.floor(Math.random() * photos.length)];
    } else {
      stateRef.current.targetPhoto = null;
    }
  };

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (ev.target?.result) {
        new THREE.TextureLoader().load(ev.target.result as string, (t) => {
          t.colorSpace = THREE.SRGBColorSpace;
          addPhotoToScene(t);
        });
      }
    };
    reader.readAsDataURL(f);
  };

  const addPhotoToScene = (texture: THREE.Texture) => {
    // Re-using geometries and materials from the first particle if possible, 
    // but for simplicity in this conversion we'll just create new ones or assume they exist.
    // In a real app we'd share them.
    const geoBox = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    const geoSphere = new THREE.SphereGeometry(0.5, 32, 32);
    const materials = {
      matGoldStandard: new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.8, roughness: 0.2 }),
    };

    const p = new Particle('PHOTO', texture, geoBox, geoSphere, materials);
    particlesRef.current.push(p);
    if (mainGroupRef.current) {
      mainGroupRef.current.add(p.mesh);
    }
    particlesRef.current.forEach((pt, i) => pt.calculatePositions(i, particlesRef.current.length));
    setMode('FOCUS');
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black">
      {loading && !cameraError && (
        <div id="loader">
          <div className="spinner"></div>
          <div className="loader-text uppercase">Loading Holiday Magic</div>
        </div>
      )}

      {cameraError && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/80 text-white p-6 text-center">
          <div className="text-red-400 text-xl mb-4 font-bold">Camera Access Error</div>
          <p className="mb-6 max-w-md text-gray-300">
            {cameraError}
          </p>
          <p className="mb-6 max-w-md text-sm text-gray-400">
            Please ensure you have granted camera permissions in your browser. If you are on a mobile device or strict browser, you may need to click the button below to request access.
          </p>
          <button 
            onClick={() => (window as any).retryCamera?.()}
            className="px-6 py-3 bg-yellow-600 hover:bg-yellow-500 text-white rounded-md font-bold tracking-wider transition-colors"
          >
            RETRY CAMERA ACCESS
          </button>
          <button 
            onClick={() => setCameraError(null)}
            className="mt-4 px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
          >
            Continue without camera (View Only)
          </button>
        </div>
      )}

      <div id="ui-layer" className={uiHidden ? 'ui-hidden' : ''}>
        <h1 className="select-none">Merry Christmas</h1>
        <div className="upload-wrapper">
          <button 
            className="upload-btn" 
            onClick={() => fileInputRef.current?.click()}
          >
            ADD MEMORIES
          </button>
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept="image/*"
            onChange={handleFileUpload}
          />
        </div>
        <div className="hint-text select-none">Press 'H' to Hide Controls</div>
      </div>

      <div id="webcam-container">
        <video ref={videoRef} autoPlay playsInline muted className="absolute opacity-0 pointer-events-none w-[320px] h-[240px] -z-10"></video>
      </div>

      <div ref={containerRef} className="absolute inset-0 z-0" />
    </div>
  );
}
