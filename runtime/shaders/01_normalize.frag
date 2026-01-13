// ============================================================================
// KizunaSR - Stage 1: Normalize
// ============================================================================
// Purpose: Convert input to consistent color space and remove artifacts
// Input: Raw frame texture (sRGB or arbitrary color space)
// Output: Normalized linear RGB in [0.0, 1.0] range
// ============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uInputFrame;

// Parameters
const float GAMMA = 2.2;
const float DENOISE_STRENGTH = 0.05;  // Subtle denoising for compression artifacts

// sRGB to linear conversion
vec3 srgbToLinear(vec3 srgb) {
    // Accurate sRGB to linear conversion
    vec3 linear;
    for (int i = 0; i < 3; i++) {
        if (srgb[i] <= 0.04045) {
            linear[i] = srgb[i] / 12.92;
        } else {
            linear[i] = pow((srgb[i] + 0.055) / 1.055, 2.4);
        }
    }
    return linear;
}

// Simple bilateral-style denoising (very lightweight)
vec3 denoise(vec2 uv) {
    vec3 center = texture(uInputFrame, uv).rgb;
    vec3 sum = center;
    float totalWeight = 1.0;
    
    // 3x3 neighborhood with distance-based weighting
    const float offsets[3] = float[](-1.0, 0.0, 1.0);
    vec2 texelSize = 1.0 / textureSize(uInputFrame, 0);
    
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (x == 1 && y == 1) continue;  // Skip center
            
            vec2 offset = vec2(offsets[x], offsets[y]) * texelSize;
            vec3 sample = texture(uInputFrame, uv + offset).rgb;
            
            // Weight by color similarity (bilateral filter concept)
            float colorDist = length(sample - center);
            float weight = exp(-colorDist / DENOISE_STRENGTH);
            
            sum += sample * weight;
            totalWeight += weight;
        }
    }
    
    return sum / totalWeight;
}

void main() {
    // Sample input
    vec3 color = denoise(vTexCoord);
    
    // Convert from sRGB to linear RGB
    vec3 linearColor = srgbToLinear(color);
    
    // Clamp to valid range
    linearColor = clamp(linearColor, 0.0, 1.0);
    
    // Output normalized linear RGB
    outColor = vec4(linearColor, 1.0);
}
