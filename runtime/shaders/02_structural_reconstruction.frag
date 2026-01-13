// ============================================================================
// KizunaSR - Stage 2: Structural Reconstruction
// ============================================================================
// Purpose: Detect edges, classify regions, and prepare for upscaling
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

// Parameters
const float EDGE_THRESHOLD = 0.1;
const float SMOOTHING_STRENGTH = 0.3;

// Sobel edge detection
float detectEdges(vec2 uv) {
    vec2 texelSize = 1.0 / textureSize(uNormalizedFrame, 0);
    
    // Sobel kernels for X and Y gradients
    float gx = 0.0;
    float gy = 0.0;
    
    // Sample 3x3 neighborhood and convert to luminance
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sample = texture(uNormalizedFrame, uv + offset).rgb;
            float luma = dot(sample, vec3(0.299, 0.587, 0.114));
            
            // Sobel weights
            float wx = float(x);
            float wy = float(y);
            
            // Horizontal gradient
            if (x == -1) gx += luma * -1.0 * (y == 0 ? 2.0 : 1.0);
            if (x == 1)  gx += luma *  1.0 * (y == 0 ? 2.0 : 1.0);
            
            // Vertical gradient
            if (y == -1) gy += luma * -1.0 * (x == 0 ? 2.0 : 1.0);
            if (y == 1)  gy += luma *  1.0 * (x == 0 ? 2.0 : 1.0);
        }
    }
    
    // Gradient magnitude
    return sqrt(gx * gx + gy * gy);
}

// Edge-aware smoothing
vec3 edgeAwareSmooth(vec2 uv, float edgeStrength) {
    vec3 center = texture(uNormalizedFrame, uv).rgb;
    
    // Strong edges: preserve sharpness
    // Weak edges or flat regions: apply smoothing
    float smoothFactor = SMOOTHING_STRENGTH * (1.0 - edgeStrength);
    
    if (smoothFactor < 0.01) {
        return center;  // No smoothing needed
    }
    
    vec3 sum = center;
    float totalWeight = 1.0;
    vec2 texelSize = 1.0 / textureSize(uNormalizedFrame, 0);
    
    // 3x3 smoothing kernel
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sample = texture(uNormalizedFrame, uv + offset).rgb;
            
            // Distance-based weight
            float dist = length(vec2(x, y));
            float weight = exp(-dist) * smoothFactor;
            
            sum += sample * weight;
            totalWeight += weight;
        }
    }
    
    return sum / totalWeight;
}

void main() {
    // Detect edge strength
    float edgeStrength = detectEdges(vTexCoord);
    
    // Normalize edge strength to [0, 1]
    edgeStrength = clamp(edgeStrength / EDGE_THRESHOLD, 0.0, 1.0);
    
    // Apply edge-aware smoothing
    vec3 color = edgeAwareSmooth(vTexCoord, edgeStrength);
    
    // Output: RGB color + edge strength in alpha
    outColor = vec4(color, edgeStrength);
}
