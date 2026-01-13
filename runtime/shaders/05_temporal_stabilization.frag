// ============================================================================
// KizunaSR - Stage 5: Temporal Stabilization
// ============================================================================
// Purpose: Reduce flicker and shimmer using previous frame blending
// Input: Enhanced frame from Stage 4 + Previous output frame
// Output: Temporally stabilized final frame
// ============================================================================

#version 450 core

// Inputs
layout(location = 0) in vec2 vTexCoord;

// Outputs
layout(location = 0) out vec4 outColor;

// Uniforms
layout(binding = 0) uniform sampler2D uCurrentFrame;   // From Stage 4
layout(binding = 1) uniform sampler2D uPreviousFrame;  // History buffer

// Parameters
layout(binding = 2) uniform TemporalParams {
    float uTemporalWeight;     // Blend weight for history (e.g., 0.15)
    float uMotionThreshold;    // Difference threshold for motion detection
    bool uFirstFrame;          // Skip temporal on first frame
};

// Motion detection via pixel difference
float detectMotion(vec3 current, vec3 previous) {
    // Compute color difference
    vec3 diff = abs(current - previous);
    float motion = max(diff.r, max(diff.g, diff.b));
    return motion;
}

void main() {
    vec3 current = texture(uCurrentFrame, vTexCoord).rgb;
    
    // On first frame, no history available
    if (uFirstFrame) {
        outColor = vec4(current, 1.0);
        return;
    }
    
    vec3 previous = texture(uPreviousFrame, vTexCoord).rgb;
    
    // Detect motion
    float motion = detectMotion(current, previous);
    
    // Adaptive blending:
    // - Low motion (static content): Blend with history to reduce flicker
    // - High motion: Use current frame only to avoid ghosting
    float blend = smoothstep(uMotionThreshold * 0.5, uMotionThreshold * 1.5, motion);
    
    // blend = 0 → static (use temporal smoothing)
    // blend = 1 → motion (use current frame)
    float historyWeight = uTemporalWeight * (1.0 - blend);
    
    vec3 result = mix(current, previous, historyWeight);
    
    // Clamp to valid range
    result = clamp(result, 0.0, 1.0);
    
    outColor = vec4(result, 1.0);
}
