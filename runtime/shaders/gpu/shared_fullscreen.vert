// ============================================================================
// KizunaSR - Shared Vertex Shader (GPU Pipeline)
// ============================================================================
// Purpose: Simple fullscreen quad vertex shader for all pipeline stages
// Usage: Used by all fragment shaders (stage1-5)
// ============================================================================

#version 450 core

// Vertex attributes
layout(location = 0) in vec2 aPosition;   // Vertex position in NDC (-1 to 1)
layout(location = 1) in vec2 aTexCoord;   // Texture coordinates (0 to 1)

// Outputs to fragment shader
layout(location = 0) out vec2 vTexCoord;

void main() {
    // Pass through texture coordinates
    vTexCoord = aTexCoord;
    
    // Set vertex position
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
