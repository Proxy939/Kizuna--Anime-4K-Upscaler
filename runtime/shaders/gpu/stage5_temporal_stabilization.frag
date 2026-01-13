// ============================================================================
// KizunaSR - GPU Stage 5: Temporal Stabilization
// ============================================================================
// Purpose: Reduce flicker using previous frame blending
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

layout(binding = 2) uniform TemporalParams {
    float uTemporalWeight;     // Blend weight for history (e.g., 0.15)
    float uMotionThreshold;    // Difference threshold for motion detection
    bool uFirstFrame;          // Skip temporal on first frame
};

// Motion detection via pixel difference
float detectMotion(vec3 current, vec3 previous) {
    vec3 diff = abs(current - previous);
    return max(diff.r, max(diff.g, diff.b));
}

// Smoothstep function for smooth transitions
float smootherstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
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
    
    // Adaptive blending using smoothstep:
    // - Low motion (static content): blend = 0 → use temporal smoothing
    // - High motion: blend = 1 → use current frame only
    float blend = smootherstep(uMotionThreshold * 0.5, uMotionThreshold * 1.5, motion);
    
    // Calculate history contribution weight
    // blend = 0 → static → use temporal weight
    // blend = 1 → motion → no temporal weight
    float historyWeight = uTemporalWeight * (1.0 - blend);
    
    // Blend current with previous
    vec3 result = mix(current, previous, historyWeight);
    
    // Clamp to valid range
    result = clamp(result, 0.0, 1.0);
    
    outColor = vec4(result, 1.0);
}
