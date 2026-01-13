#version 450 core

// YUV to RGB Color Space Conversion Shader
// =========================================
// Converts YUV420 (BT.709) to linear RGB
// Input: Y plane (single channel), UV plane (two channels)
// Output: RGB

in vec2 vTexCoord;
out vec4 FragColor;

// Input textures
uniform sampler2D uYPlane;    // Y (luminance) - single channel
uniform sampler2D uUVPlane;   // UV (chrominance) - two channels

// BT.709 YUV to RGB conversion matrix
// Standard for HD video (720p, 1080p, 4K)
const mat3 YUV_TO_RGB_BT709 = mat3(
    1.0,     1.0,      1.0,
    0.0,    -0.18732,  1.8556,
    1.5748, -0.46812,  0.0
);

void main() {
    // Sample Y plane (luminance)
    float y = texture(uYPlane, vTexCoord).r;
    
    // Sample UV plane (chrominance)
    vec2 uv = texture(uUVPlane, vTexCoord).rg;
    
    // Normalize from [0, 1] to YUV range
    // Y: [16/255, 235/255] -> [0, 1]
    // UV: [16/255, 240/255] -> [-0.5, 0.5]
    y = (y - 0.0625) / 0.85937;  // (y - 16/255) / (235-16)/255
    uv = (uv - 0.5);  // Center around 0
    
    // Construct YUV vector
    vec3 yuv = vec3(y, uv.x, uv.y);
    
    // Convert to RGB using BT.709 matrix
    vec3 rgb = YUV_TO_RGB_BT709 * yuv;
    
    // Clamp to valid range
    rgb = clamp(rgb, 0.0, 1.0);
    
    // Output linear RGB
    FragColor = vec4(rgb, 1.0);
}
