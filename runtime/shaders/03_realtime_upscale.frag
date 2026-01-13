// ============================================================================
// KizunaSR - Stage 3: Real-Time Upscale
// ============================================================================
// Purpose: Increase resolution using edge-aware interpolation
// Input: Processed frame from Stage 2 (RGB + edge in alpha)
// Output: Upscaled frame at target resolution
// Scale Factor: 2× (can be adjusted via uniforms)
// ============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uProcessedFrame;

// Parameters
layout(binding = 1) uniform UpscaleParams {
    vec2 uSourceSize;      // Source resolution
    vec2 uTargetSize;      // Target resolution (typically 2× source)
    float uSharpness;      // Edge sharpening strength
};

const float EDGE_THRESHOLD = 0.3;  // When to use edge-directed sampling

// Enhanced bicubic interpolation
vec3 bicubicSample(vec2 uv) {
    vec2 texelSize = 1.0 / uSourceSize;
    vec2 pixel = uv * uSourceSize - 0.5;
    vec2 frac = fract(pixel);
    pixel = floor(pixel);
    
    // Cubic interpolation weights
    vec4 xcubic = vec4(
        ((-0.5 * frac.x + 1.0) * frac.x - 0.5) * frac.x,
        (1.5 * frac.x - 2.5) * frac.x * frac.x + 1.0,
        ((-1.5 * frac.x + 2.0) * frac.x + 0.5) * frac.x,
        (0.5 * frac.x - 0.5) * frac.x * frac.x
    );
    
    vec4 ycubic = vec4(
        ((-0.5 * frac.y + 1.0) * frac.y - 0.5) * frac.y,
        (1.5 * frac.y - 2.5) * frac.y * frac.y + 1.0,
        ((-1.5 * frac.y + 2.0) * frac.y + 0.5) * frac.y,
        (0.5 * frac.y - 0.5) * frac.y * frac.y
    );
    
    vec3 result = vec3(0.0);
    
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 samplePos = (pixel + vec2(float(x) - 1.0, float(y) - 1.0) + 0.5) * texelSize;
            vec3 sample = texture(uProcessedFrame, samplePos).rgb;
            result += sample * xcubic[x] * ycubic[y];
        }
    }
    
    return result;
}

// Edge-directed interpolation
vec3 edgeDirectedSample(vec2 uv, float edgeStrength) {
    vec2 texelSize = 1.0 / uSourceSize;
    
    // Sample edge strengths in a 3x3 neighborhood
    float edges[9];
    int idx = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            edges[idx++] = texture(uProcessedFrame, uv + offset).a;
        }
    }
    
    // Determine edge direction by gradient
    float gx = (edges[2] + 2.0 * edges[5] + edges[8]) - 
               (edges[0] + 2.0 * edges[3] + edges[6]);
    float gy = (edges[6] + 2.0 * edges[7] + edges[8]) - 
               (edges[0] + 2.0 * edges[1] + edges[2]);
    
    float angle = atan(gy, gx);
    
    // Sample along edge direction (perpendicular to gradient)
    vec2 edgeDir = vec2(cos(angle + 1.57079632679), sin(angle + 1.57079632679));
    
    vec3 sample1 = texture(uProcessedFrame, uv + edgeDir * texelSize).rgb;
    vec3 sample2 = texture(uProcessedFrame, uv - edgeDir * texelSize).rgb;
    vec3 center = texture(uProcessedFrame, uv).rgb;
    
    // Weighted blend favoring edge-aligned samples
    return mix(center, (sample1 + sample2) * 0.5, 0.3);
}

void main() {
    // Convert output pixel coordinate to source texture space
    vec2 sourceUV = vTexCoord;
    
    // Sample edge strength at this location
    float edgeStrength = texture(uProcessedFrame, sourceUV).a;
    
    vec3 color;
    
    // Use edge-directed sampling for strong edges, bicubic for smooth regions
    if (edgeStrength > EDGE_THRESHOLD) {
        // Strong edge: use directional sampling
        vec3 bicubic = bicubicSample(sourceUV);
        vec3 edgeDirected = edgeDirectedSample(sourceUV, edgeStrength);
        color = mix(bicubic, edgeDirected, edgeStrength * uSharpness);
    } else {
        // Flat or gradient region: use bicubic interpolation
        color = bicubicSample(sourceUV);
    }
    
    // Clamp to valid range
    color = clamp(color, 0.0, 1.0);
    
    outColor = vec4(color, 1.0);
}
