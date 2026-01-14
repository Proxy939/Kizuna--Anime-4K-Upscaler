// Background grid animation with Three.js
// Creates the subtle animated grid background

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('bgCanvas'),
    alpha: true,
    antialias: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
camera.position.z = 5;

// Create grid particles
const particleCount = 150;
const particles = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);
const alphas = new Float32Array(particleCount);

for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 20;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
    alphas[i] = Math.random();
}

particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
particles.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));

const particleShader = new THREE.ShaderMaterial({
    transparent: true,
    vertexShader: `
        attribute float alpha;
        varying float vAlpha;
        
        void main() {
            vAlpha = alpha;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = 2.0 * (300.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
        }
    `,
    fragmentShader: `
        varying float vAlpha;
        
        void main() {
            vec2 center = gl_PointCoord - vec2(0.5);
            float dist = length(center);
            if (dist > 0.5) discard;
            
            float alpha = (1.0 - dist * 2.0) * vAlpha * 0.3;
            gl_FragColor = vec4(1.0, 1.0, 1.0, alpha);
        }
    `
});

const particleSystem = new THREE.Points(particles, particleShader);
scene.add(particleSystem);

// Animation loop
let time = 0;
function animateBackground() {
    requestAnimationFrame(animateBackground);
    time += 0.001;

    particleSystem.rotation.y = time * 0.1;
    particleSystem.rotation.x = Math.sin(time * 0.5) * 0.1;

    renderer.render(scene, camera);
}

animateBackground();

// Handle resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
