// GSAP Animations - Opening sequence and scroll effects
// Matches exact behavior from design

gsap.registerPlugin(ScrollTrigger);

// Opening Animation Sequence
function initOpeningAnimation() {
    const tl = gsap.timeline({
        defaults: { ease: 'power3.inOut' }
    });

    // Initially hide everything
    gsap.set('.title-letter', { opacity: 0, y: 100, rotateX: -90 });
    gsap.set('#portal-container', { opacity: 0, scale: 0.5 });
    gsap.set('.fixed', { opacity: 0 });
    gsap.set('#bgCanvas', { opacity: 0 });

    // Sequence:
    // 1. Fade in background grid (0.5s)
    tl.to('#bgCanvas', {
        opacity: 1,
        duration: 1,
        delay: 0.3
    })

        // 2. Portal appears (1s)
        .to('#portal-container', {
            opacity: 1,
            scale: 1,
            duration: 1.5,
            ease: 'power2.out'
        }, '-=0.5')

        // 3. Title letters assemble (staggered, 1.5s total)
        .to('.title-letter', {
            opacity: 1,
            y: 0,
            rotateX: 0,
            duration: 1,
            stagger: {
                amount: 0.8,
                from: 'start'
            },
            ease: 'back.out(1.7)'
        }, '-=0.8')

        // 4. UI elements fade in
        .to('.fixed', {
            opacity: 1,
            duration: 0.8,
            stagger: 0.1
        }, '-=0.5');

    return tl;
}

// Title hover effects
function initTitleEffects() {
    const letters = document.querySelectorAll('.title-letter');

    letters.forEach((letter, index) => {
        letter.addEventListener('mouseenter', () => {
            gsap.to(letter, {
                y: -20,
                scale: 1.1,
                color: index === 2 ? '#00aaff' : '#ff0088', // Pyramid letter gets blue
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        letter.addEventListener('mouseleave', () => {
            gsap.to(letter, {
                y: 0,
                scale: 1,
                color: '#ffffff',
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    });
}

// Parallax scroll effects
function initScrollAnimations() {
    // Title parallax on scroll
    gsap.to('#main-title', {
        y: () => window.innerHeight * 0.3,
        opacity: 0,
        scrollTrigger: {
            trigger: '#hero',
            start: 'top top',
            end: 'bottom top',
            scrub: true
        }
    });

    // Portal depth parallax
    gsap.to('#portal-container', {
        scale: 1.5,
        y: () => window.innerHeight * 0.2,
        scrollTrigger: {
            trigger: '#hero',
            start: 'top top',
            end: 'bottom top',
            scrub: true
        }
    });

    // Controls fade out on scroll
    gsap.to('.fixed', {
        opacity: 0,
        y: 50,
        scrollTrigger: {
            trigger: '#hero',
            start: 'top top',
            end: '+=300',
            scrub: true
        }
    });
}

// Cursor glow effect
function initCursorGlow() {
    const cursorGlow = document.createElement('div');
    cursorGlow.style.cssText = `
        position: fixed;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,170,255,0.15) 0%, transparent 70%);
        pointer-events: none;
        z-index: 9999;
        transform: translate(-50%, -50%);
        mix-blend-mode: screen;
        transition: opacity 0.3s;
    `;
    document.body.appendChild(cursorGlow);

    document.addEventListener('mousemove', (e) => {
        gsap.to(cursorGlow, {
            x: e.clientX,
            y: e.clientY,
            duration: 0.8,
            ease: 'power2.out'
        });
    });
}

// Initialize all animations
document.addEventListener('DOMContentLoaded', () => {
    // Wait for assets to load
    setTimeout(() => {
        initOpeningAnimation();
        initTitleEffects();
        initScrollAnimations();
        initCursorGlow();
    }, 100);
});
