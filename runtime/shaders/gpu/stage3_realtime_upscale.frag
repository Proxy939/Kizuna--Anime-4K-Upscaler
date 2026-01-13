// ============================================================================
// KizunaSR - GPU Stage 3: Real-Time Upscale
// ============================================================================
// Purpose: Analytic upscaling using bicubic interpolation
// Input: Processed frame from Stage 2 (RGB + edge in alpha)
// Output: Upscaled frame at target resolution
// ============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uProcessedFrame;

layout(binding = 1) uniform UpscaleParams {
    vec2 uSourceSize;      // Source resolution
    vec2 uTargetSize;      // Target resolution
    float uScaleFactor;    // Scale factor (2.0 or 4.0)
};

// Cubic interpolation weight
float cubicWeight(float x) {
    float ax = abs(x);
    if (ax <= 1.0) {
        return (1.5 * ax - 2.5) * ax * ax + 1.0;
    } else if (ax < 2.0) {
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
    }
    return 0.0;
}

// Bicubic texture sampling
vec3 bicubicSample(vec2 uv) {
    vec2 texelSize = 1.0 / uSourceSize;
    
    // Calculate the pixel center in source texture space
    vec2 pixel = uv * uSourceSize - 0.5;
    vec2 frac = fract(pixel);
    vec2 pixelInt = floor(pixel);
    
    // Cubic weights for 4x4 sampling
    vec4 xWeights = vec4(
        cubicWeight(frac.x + 1.0),
        cubicWeight(frac.x),
        cubicWeight(1.0 - frac.x),
        cubicWeight(2.0 - frac.x)
    );
    
    vec4 yWeights = vec4(
        cubicWeight(frac.y + 1.0),
        cubicWeight(frac.y),
        cubicWeight(1.0 - frac.y),
        cubicWeight(2.0 - frac.y)
    );
    
    vec3 result = vec3(0.0);
    float weightSum = 0.0;
    
    // Sample 4x4 neighborhood
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 samplePos = (pixelInt + vec2(float(x) - 1.0, float(y) - 1.0) + 0.5) * texelSize;
            vec3 sample = texture(uProcessedFrame, samplePos).rgb;
            
            float weight = xWeights[x] * yWeights[y];
            result += sample * weight;
            weightSum += weight;
        }
    }
    
    // Normalize
    return result / weightSum;
}

void main() {
    // Bicubic interpolation
    vec3 color = bicubicSample(vTexCoord);
    
    // Clamp to valid range
    color = clamp(color, 0.0, 1.0);
    
    outColor = vec4(color, 1.0);
}
