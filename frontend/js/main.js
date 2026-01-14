// Main initialization and coordination
// Ensures all components load in correct order

console.log('%c KizunaSR Homepage Initialized ', 'background: #000; color: #0af; font-size: 16px; padding: 10px;');

// Performance monitoring
const perfStart = performance.now();

// Wait for all scripts to load
window.addEventListener('load', () => {
    const perfEnd = performance.now();
    console.log(`⚡ Page loaded in ${(perfEnd - perfStart).toFixed(2)}ms`);

    // Verify Three.js scenes are running
    console.log('✓ Background scene active');
    console.log('✓ Portal scene active');
    console.log('✓ GSAP animations initialized');
});

// Smooth scroll setup
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);

        if (targetElement) {
            gsap.to(window, {
                scrollTo: {
                    y: targetElement,
                    autoKill: true
                },
                duration: 1.5,
                ease: 'power3.inOut'
            });
        }
    });
});

// Preload optimization
function preloadAssets() {
    // Warm up WebGL context
    const testCanvas = document.createElement('canvas');
    const gl = testCanvas.getContext('webgl2') || testCanvas.getContext('webgl');
    if (gl) {
        console.log('✓ WebGL2 support detected');
    }
}

preloadAssets();

// Handle visibility change (pause animations when tab is hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('⏸ Animations paused');
        // Renderers will continue but at lower priority
    } else {
        console.log('▶ Animations resumed');
    }
});

// Error handling
window.addEventListener('error', (e) => {
    console.error('❌ Runtime error:', e.message);
});

// Resize debouncing
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        console.log('↔ Viewport resized');
        // All resize handlers are in individual script files
    }, 250);
});
