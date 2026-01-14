// Glass pyramid portal effect with Three.js
// Replicates the central triangular element

const portalScene = new THREE.Scene();
const portalCamera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
const portalRenderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('portalCanvas'),
    alpha: true,
    antialias: true
});

portalRenderer.setSize(window.innerWidth, window.innerHeight);
portalRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
portalCamera.position.z = 8;

// Create pyramid geometry
const pyramidGeometry = new THREE.ConeGeometry(2.5, 4, 3);
pyramidGeometry.rotateY(Math.PI);

// Glass material with reflections
const pyramidMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x88ccff,
    metalness: 0.2,
    roughness: 0.1,
    opacity: 0.6,
    transparent: true,
    transmission: 0.95,
    thickness: 0.5,

    emissive: 0x0066ff,
    emissiveIntensity: 0.1,
    clearcoat: 1.0,
    clearcoatRoughness: 0.1,
    ior: 1.5
});

const pyramid = new THREE.Mesh(pyramidGeometry, pyramidMaterial);
pyramid.rotation.x = Math.PI;
portalScene.add(pyramid);

// Add inner glow
const innerGlowGeometry = new THREE.ConeGeometry(2.3, 3.8, 3);
innerGlowGeometry.rotateY(Math.PI);
const innerGlowMaterial = new THREE.MeshBasicMaterial({
    color: 0x66aaff,
    transparent: true,
    opacity: 0.15,
    side: THREE.BackSide
});
const innerGlow = new THREE.Mesh(innerGlowGeometry, innerGlowMaterial);
innerGlow.rotation.x = Math.PI;
portalScene.add(innerGlow);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
portalScene.add(ambientLight);

const pointLight1 = new THREE.PointLight(0x00aaff, 1, 50);
pointLight1.position.set(5, 5, 5);
portalScene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0xff0088, 0.8, 50);
pointLight2.position.set(-5, -5, -5);
portalScene.add(pointLight2);

// Mouse interaction
let mouseX = 0;
let mouseY = 0;

document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
});

// Animation
let portalTime = 0;
function animatePortal() {
    requestAnimationFrame(animatePortal);
    portalTime += 0.01;

    // Slow rotation
    pyramid.rotation.y = portalTime * 0.1;
    innerGlow.rotation.y = portalTime * 0.1;

    // Subtle floating
    pyramid.position.y = Math.sin(portalTime * 0.5) * 0.1;

    // Reactive to mouse
    pyramid.rotation.x = Math.PI + mouseY * 0.3;
    pyramid.rotation.z = mouseX * 0.2;

    // Glow pulsing
    pyramidMaterial.emissiveIntensity = 0.1 + Math.sin(portalTime) * 0.05;

    portalRenderer.render(portalScene, portalCamera);
}

animatePortal();

// Handle resize
window.addEventListener('resize', () => {
    portalCamera.aspect = window.innerWidth / window.innerHeight;
    portalCamera.updateProjectionMatrix();
    portalRenderer.setSize(window.innerWidth, window.innerHeight);
});
