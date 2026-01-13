// ============================================================================
// KizunaSR - GPU Stage 2: Structural Reconstruction
// ============================================================================
// Purpose: Detect edges and apply edge-aware smoothing
// Input: Normalized linear RGB from Stage 1
// Output: RGB + Edge strength in alpha channel
// ============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uNormalizedFrame;

// Convert RGB to luminance
float rgb2luma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

// Sobel edge detection
float detectEdges(vec2 uv) {
    vec2 texelSize = 1.0 / textureSize(uNormalizedFrame, 0);
    
    // Sample 3x3 neighborhood luminance
    float luma[9];
    int idx = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sample = texture(uNormalizedFrame, uv + offset).rgb;
            luma[idx++] = rgb2luma(sample);
        }
    }
    
    // Sobel operator
    // Gx: [-1  0  1]    Gy: [-1 -2 -1]
    //     [-2  0  2]        [ 0  0  0]
    //     [-1  0  1]        [ 1  2  1]
    
    float gx = -luma[0] + luma[2] - 2.0*luma[3] + 2.0*luma[5] - luma[6] + luma[8];
    float gy = -luma[0] - 2.0*luma[1] - luma[2] + luma[6] + 2.0*luma[7] + luma[8];
    
    // Gradient magnitude
    float edge = sqrt(gx * gx + gy * gy);
    
    // Normalize to [0, 1] range
    return clamp(edge / 0.5, 0.0, 1.0);
}

// Edge-aware smoothing
vec3 edgeAwareSmooth(vec2 uv, float edgeStrength) {
    vec3 center = texture(uNormalizedFrame, uv).rgb;
    
    // Strong edges: preserve sharpness (no smoothing)
    // Weak edges: apply smoothing
    const float SMOOTHING_STRENGTH = 0.3;
    float smoothFactor = SMOOTHING_STRENGTH * (1.0 - edgeStrength);
    
    if (smoothFactor < 0.01) {
        return center;
    }
    
    vec2 texelSize = 1.0 / textureSize(uNormalizedFrame, 0);
    
    // 3x3 Gaussian kernel
    const float kernel[9] = float[](
        0.0625, 0.125, 0.0625,
        0.125,  0.25,  0.125,
        0.0625, 0.125, 0.0625
    );
    
    vec3 result = vec3(0.0);
    int idx = 0;
    
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sample = texture(uNormalizedFrame, uv + offset).rgb;
            result += sample * kernel[idx++];
        }
    }
    
    // Blend between original and smoothed based on edge strength
    return mix(result, center, edgeStrength);
}

void main() {
    // Detect edge strength
    float edgeStrength = detectEdges(vTexCoord);
    
    // Apply edge-aware smoothing
    vec3 color = edgeAwareSmooth(vTexCoord, edgeStrength);
    
    // Clamp to valid range
    color = clamp(color, 0.0, 1.0);
    
    // Output: RGB + edge strength in alpha
    outColor = vec4(color, edgeStrength);
}
