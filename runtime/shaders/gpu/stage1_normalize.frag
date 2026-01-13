// ============================================================================
// KizunaSR - GPU Stage 1: Normalize
// ============================================================================
// Purpose: Convert sRGB to linear RGB and apply light denoising
// Input: Raw frame texture (sRGB)
// Output: Normalized linear RGB
// ============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uInputFrame;

// sRGB to linear conversion (accurate)
vec3 srgbToLinear(vec3 srgb) {
    vec3 linear;
    linear.r = (srgb.r <= 0.04045) ? srgb.r / 12.92 : pow((srgb.r + 0.055) / 1.055, 2.4);
    linear.g = (srgb.g <= 0.04045) ? srgb.g / 12.92 : pow((srgb.g + 0.055) / 1.055, 2.4);
    linear.b = (srgb.b <= 0.04045) ? srgb.b / 12.92 : pow((srgb.b + 0.055) / 1.055, 2.4);
    return linear;
}

// Lightweight Gaussian-style smoothing for compression artifacts
vec3 denoise(vec2 uv) {
    vec2 texelSize = 1.0 / textureSize(uInputFrame, 0);
    
    // 3x3 Gaussian weights (normalized)
    const float kernel[9] = float[](
        0.0625, 0.125, 0.0625,
        0.125,  0.25,  0.125,
        0.0625, 0.125, 0.0625
    );
    
    vec3 result = vec3(0.0);
    int idx = 0;
    
    // Sample 3x3 neighborhood
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sample = texture(uInputFrame, uv + offset).rgb;
            result += sample * kernel[idx++];
        }
    }
    
    return result;
}

void main() {
    // Light denoising
    vec3 color = denoise(vTexCoord);
    
    // Convert from sRGB to linear RGB
    vec3 linearColor = srgbToLinear(color);
    
    // Clamp to valid range
    linearColor = clamp(linearColor, 0.0, 1.0);
    
    // Output linear RGB
    outColor = vec4(linearColor, 1.0);
}
