// ============================================================================
// KizunaSR - Stage 4: Perceptual Enhancement
// ============================================================================
// Purpose: Enhance visual quality with contrast, sharpening, and saturation
// Input: Upscaled frame from Stage 3
// Output: Enhanced frame with improved visual appeal
// =============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uUpscaledFrame;

// Parameters
layout(binding = 1) uniform EnhanceParams {
    float uContrastBoost;       // 1.0 = no change, 1.2 = 20% boost
    float uSaturationBoost;     // 1.0 = no change, 1.1 = 10% boost
    float uSharpening;          // 0.0 = no sharpening, 0.5 = moderate
    float uLineDarkening;       // 0.0 = no darkening, 0.2 = 20% darker
};

// Convert RGB to HSV
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Convert HSV to RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Unsharp masking for sharpening
vec3 sharpen(vec2 uv, vec3 center) {
    vec2 texelSize = 1.0 / textureSize(uUpscaledFrame, 0);
    
    // 3x3 Gaussian blur for unsharp mask
    vec3 blurred = vec3(0.0);
    float weights[9] = float[](
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0
    );
    float totalWeight = 16.0;
    
    int idx = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sample = texture(uUpscaledFrame, uv + offset).rgb;
            blurred += sample * weights[idx++];
        }
    }
    blurred /= totalWeight;
    
    // Unsharp mask: sharp = original + (original - blurred) * strength
    vec3 detail = center - blurred;
    return center + detail * uSharpening;
}

// S-curve contrast adjustment
vec3 adjustContrast(vec3 color, float contrast) {
    // S-curve formula: 0.5 + (color - 0.5) * contrast
    // Enhanced with smooth step for subtle enhancement
    vec3 adjusted = (color - 0.5) * contrast + 0.5;
    return clamp(adjusted, 0.0, 1.0);
}

// Line art darkening (darken near-black pixels)
vec3 darkenLines(vec3 color, float strength) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    
    // Only darken dark pixels (line art)
    if (luma < 0.3) {
        float darkenFactor = 1.0 - strength * (1.0 - luma / 0.3);
        return color * darkenFactor;
    }
    
    return color;
}

void main() {
    vec3 color = texture(uUpscaledFrame, vTexCoord).rgb;
    
    // Step 1: Sharpen (unsharp masking)
    if (uSharpening > 0.01) {
        color = sharpen(vTexCoord, color);
    }
    
    // Step 2: Darken line art for definition
    if (uLineDarkening > 0.01) {
        color = darkenLines(color, uLineDarkening);
    }
    
    // Step 3: Contrast enhancement
    if (abs(uContrastBoost - 1.0) > 0.01) {
        color = adjustContrast(color, uContrastBoost);
    }
    
    // Step 4: Saturation boost
    if (abs(uSaturationBoost - 1.0) > 0.01) {
        vec3 hsv = rgb2hsv(color);
        hsv.y *= uSaturationBoost;  // Boost saturation
        hsv.y = clamp(hsv.y, 0.0, 1.0);
        color = hsv2rgb(hsv);
    }
    
    // Final clamp
    color = clamp(color, 0.0, 1.0);
    
    outColor = vec4(color, 1.0);
}
