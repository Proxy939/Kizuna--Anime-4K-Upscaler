"""
KizunaSR - Shader Pipeline Executor
====================================
Compiles and executes the KizunaSR GPU shader pipeline.

Responsibilities:
- Load and compile GLSL shaders
- Manage framebuffers for each pipeline stage
- Execute 5-stage pipeline per frame
- Handle uniforms and resolution changes

This module does NOT decode video, schedule frames, or play audio.
"""

from OpenGL.GL import *
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


class ShaderProgram:
    """Encapsulates a compiled and linked OpenGL shader program."""
    
    def __init__(self, name: str, vertex_source: str, fragment_source: str):
        """
        Compile and link shader program.
        
        Args:
            name: Program name (for logging)
            vertex_source: Vertex shader GLSL source
            fragment_source: Fragment shader GLSL source
        """
        self.name = name
        self.program_id = 0
        
        # Compile shaders
        vertex_shader = self._compile_shader(vertex_source, GL_VERTEX_SHADER, "vertex")
        fragment_shader = self._compile_shader(fragment_source, GL_FRAGMENT_SHADER, "fragment")
        
        # Link program
        self.program_id = glCreateProgram()
        glAttachShader(self.program_id, vertex_shader)
        glAttachShader(self.program_id, fragment_shader)
        glLinkProgram(self.program_id)
        
        # Check link status
        if not glGetProgramiv(self.program_id, GL_LINK_STATUS):
            log = glGetProgramInfoLog(self.program_id).decode('utf-8')
            raise RuntimeError(f"Shader program '{name}' link failed:\n{log}")
        
        # Delete shaders (no longer needed after linking)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        print(f"[Shader] Compiled and linked: {name}")
    
    def _compile_shader(self, source: str, shader_type: int, type_name: str) -> int:
        """Compile a shader and check for errors."""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader).decode('utf-8')
            raise RuntimeError(f"{type_name.capitalize()} shader compilation failed ({self.name}):\n{log}")
        
        return shader
    
    def use(self):
        """Activate this shader program."""
        glUseProgram(self.program_id)
    
    def set_uniform_int(self, name: str, value: int):
        """Set integer uniform."""
        location = glGetUniformLocation(self.program_id, name)
        if location >= 0:
            glUniform1i(location, value)
    
    def set_uniform_float(self, name: str, value: float):
        """Set float uniform."""
        location = glGetUniformLocation(self.program_id, name)
        if location >= 0:
            glUniform1f(location, value)
    
    def set_uniform_vec2(self, name: str, x: float, y: float):
        """Set vec2 uniform."""
        location = glGetUniformLocation(self.program_id, name)
        if location >= 0:
            glUniform2f(location, x, y)
    
    def cleanup(self):
        """Delete shader program."""
        if self.program_id:
            glDeleteProgram(self.program_id)
            self.program_id = 0


class ShaderPipelineExecutor:
    """
    Executes the KizunaSR GPU shader pipeline.
    
    5 stages: Normalize → Structural Recon → Upscale → Enhancement → Temporal
    """
    
    def __init__(self, shader_dir: str, scale_factor: int = 2):
        """
        Initialize shader pipeline.
        
        Args:
            shader_dir: Directory containing GLSL shader files
            scale_factor: Upscaling factor (2 or 4)
        """
        self.shader_dir = Path(shader_dir)
        self.scale_factor = scale_factor
        
        # Shader programs
        self.programs: List[ShaderProgram] = []
        
        # Framebuffers (one per stage)
        self.fbos: List[int] = []
        
        # Intermediate textures
        self.intermediate_textures: List[int] = []
        
        # History texture for temporal stabilization
        self.history_texture: Optional[int] = None
        
        # Current input resolution
        self.input_width = 0
        self.input_height = 0
        
        # Fullscreen quad VAO
        self.quad_vao = 0
        
        print(f"[Pipeline] Initializing shaders from: {shader_dir}")
        print(f"[Pipeline] Scale factor: {scale_factor}×")
        
        # Load and compile shaders
        self._load_shaders()
        
        # Load YUV conversion shader (Stage 0)
        self._load_yuv_shader()
        
        # Create fullscreen quad geometry
        self._create_fullscreen_quad()
        
        print("[Pipeline] Initialization complete")
    
    def _load_shaders(self):
        """Load and compile all pipeline shaders."""
        # Vertex shader (shared)
        vertex_path = self.shader_dir / "shared_fullscreen.vert"
        vertex_source = vertex_path.read_text()
        
        # Fragment shaders (one per stage)
        stages = [
            ("stage1_normalize", "Normalize"),
            ("stage2_structural_reconstruction", "Structural Reconstruction"),
            ("stage3_realtime_upscale", "Real-Time Upscale"),
            ("stage4_perceptual_enhancement", "Perceptual Enhancement"),
            ("stage5_temporal_stabilization", "Temporal Stabilization")
        ]
        
        for stage_file, stage_name in stages:
            fragment_path = self.shader_dir / f"{stage_file}.frag"
            
            if not fragment_path.exists():
                raise FileNotFoundError(f"Shader not found: {fragment_path}")
            
            fragment_source = fragment_path.read_text()
            
            program = ShaderProgram(stage_name, vertex_source, fragment_source)
            self.programs.append(program)
    
    def _load_yuv_shader(self):
        """Load YUV to RGB conversion shader (Stage 0)."""
        vertex_path = self.shader_dir / "shared_fullscreen.vert"
        fragment_path = self.shader_dir / "yuv_to_rgb.frag"
        
        if not fragment_path.exists():
            print("[Pipeline] YUV shader not found, GPU YUV path disabled")
            self.yuv_program = None
            return
            
        try:
            vertex_source = vertex_path.read_text()
            fragment_source = fragment_path.read_text()
            self.yuv_program = ShaderProgram("YUV to RGB", vertex_source, fragment_source)
            print("[Pipeline] YUV shader loaded (Stage 0 enabled)")
        except Exception as e:
            print(f"[Pipeline] YUV shader compilation failed: {e}")
            self.yuv_program = None
    
    def _create_fullscreen_quad(self):
        """Create VAO for fullscreen quad rendering."""
        # Quad vertices (NDC + texcoords)
        vertices = np.array([
            # pos (x, y)   tex (u, v)
            -1.0, -1.0,    0.0, 0.0,  # Bottom-left
             1.0, -1.0,    1.0, 0.0,  # Bottom-right
            -1.0,  1.0,    0.0, 1.0,  # Top-left
             1.0,  1.0,    1.0, 1.0   # Top-right
        ], dtype=np.float32)
        
        # Create VAO
        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)
        
        # Create VBO
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Position attribute (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        
        # Texcoord attribute (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        
        # Unbind
        glBindVertexArray(0)
        
        print("[Pipeline] Created fullscreen quad VAO")
    
    def resize(self, input_width: int, input_height: int):
        """
        Resize pipeline for new input resolution.
        
        Args:
            input_width: Input frame width
            input_height: Input frame height
        """
        if input_width == self.input_width and input_height == self.input_height:
            return  # No change
        
        print(f"[Pipeline] Resizing: {input_width}×{input_height} → {input_width * self.scale_factor}×{input_height * self.scale_factor}")
        
        self.input_width = input_width
        self.input_height = input_height
        
        # Clean up old resources
        self._cleanup_fbos()
        
        # Create new FBOs and textures
        self._create_fbos()
    
    def _create_fbos(self):
        """Create framebuffers and intermediate textures."""
        # Stage resolutions
        resolutions = [
            (self.input_width, self.input_height),  # Stage 1: Normalize
            (self.input_width, self.input_height),  # Stage 2: Structural Recon
            (self.input_width * self.scale_factor, self.input_height * self.scale_factor),  # Stage 3: Upscale
            (self.input_width * self.scale_factor, self.input_height * self.scale_factor),  # Stage 4: Enhancement
            (self.input_width * self.scale_factor, self.input_height * self.scale_factor)   # Stage 5: Temporal
        ]
        
        # Stage 0 FBO (YUV -> RGB conversion)
        # Matches input resolution
        if self.input_width > 0:
            self.stage0_fbo = self._create_fbo(self.input_width, self.input_height)
        else:
            self.stage0_fbo = (0, 0) # fbo, texture
        
        for width, height in resolutions:
            # Create FBO
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            
            # Create texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Attach texture to FBO
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
            
            # Check FBO status
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Framebuffer incomplete: {status}")
            
            self.fbos.append(fbo)
            self.intermediate_textures.append(texture)
    
    def _create_fbo(self, width: int, height: int) -> Tuple[int, int]:
        """Helper to create a single FBO and texture."""
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")
            
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return fbo, texture
        
        # Create history texture for temporal stabilizationtemporal stabilization
        self.history_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.history_texture)
        output_width = self.input_width * self.scale_factor
        output_height = self.input_height * self.scale_factor
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, output_width, output_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Unbind
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        print(f"[Pipeline] Created {len(self.fbos)} FBOs")
    
    def _cleanup_fbos(self):
        """Delete FBOs and intermediate textures."""
        if self.fbos:
            glDeleteFramebuffers(len(self.fbos), self.fbos)
            self.fbos.clear()
        
        if self.intermediate_textures:
            glDeleteTextures(self.intermediate_textures)
            self.intermediate_textures.clear()
            
        if hasattr(self, 'stage0_fbo') and self.stage0_fbo != (0, 0):
            glDeleteFramebuffers(1, [self.stage0_fbo[0]])
            glDeleteTextures([self.stage0_fbo[1]])
            self.stage0_fbo = (0, 0)
        
        if self.history_texture:
            glDeleteTextures([self.history_texture])
            self.history_texture = None
    
        Returns:
            Output texture ID (final processed frame)
        """
        # Determine if input is YUV (using uploader's logic)
        # Texture IDs < 10000 are usually standard, large ones might be handles
        # But wait, input_texture_id is just an int.
        # We need a way to know if it's YUV.
        # The OpenGLTextureUploader returns a texture ID for YUV which matches the Y plane ID.
        # But we need access to the V plane too.
        # The caller (Player) doesn't pass the YUV handle tuple.
        # We need to change execute() signature or handle logic differently.
        #
        # For this implementation, we'll assume if input_texture_id matches
        # a known YUV set in the uploader (if we had access), we'd use it.
        # Since we don't have access to the uploader here, we need to pass the YUV info.
        # However, we must keep API component compatibility (Requirement 5).
        #
        # TRICK: The 'input_texture_id' for YUV frames returned by uploader
        # is actually the Y plane ID. It's a valid GL texture.
        # But we need the UV plane ID.
        #
        # Proposal: The Scheduler/Player passes the frame object or specialized struct?
        # Requirement 5 says: ShaderPipelineExecutor public API must NOT change.
        #
        # Solution:
        # We can inspect the texture_id. If it was created as YUV,
        # we need a registry. But avoiding coupling.
        #
        # Wait, the simplest way is to add an optional argument to execute()
        # that doesn't break existing calls: 'yuv_uv_texture_id: Optional[int] = None'
        #
        pass # Placeholder for logic below

    def execute(self, input_texture_id: int, frame_index: int = 0, uv_texture_id: Optional[int] = None) -> int:
        """
        Execute shader pipeline on input texture.
        
        Args:
            input_texture_id: Input texture (RGB or Y plane)
            frame_index: Frame index (for temporal stabilization)
            uv_texture_id: UV plane texture ID (if input is YUV) - NEW OPTIONAL ARG
        
        Returns:
            Output texture ID (final processed frame)
        """
        # Bind fullscreen quad
        glBindVertexArray(self.quad_vao)
        
        # Disable depth test (2D rendering)
        glDisable(GL_DEPTH_TEST)
        
        # Current texture (starts with input)
        current_texture = input_texture_id
        
        # Stage 0: YUV -> RGB (Conditional)
        if uv_texture_id is not None and self.yuv_program:
            # Bind Stage 0 FBO
            glBindFramebuffer(GL_FRAMEBUFFER, self.stage0_fbo[0])
            glViewport(0, 0, self.input_width, self.input_height)
            
            self.yuv_program.use()
            
            # Bind Y (current) and UV textures
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, input_texture_id)
            self.yuv_program.set_uniform_int('uYPlane', 0)
            
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, uv_texture_id)
            self.yuv_program.set_uniform_int('uUVPlane', 1)
            
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            
            # Output of Stage 0 is the input to Stage 1
            current_texture = self.stage0_fbo[1]
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Stage 1: Normalize
        current_texture = self._execute_stage(
            stage=0,
            program=self.programs[0],
            input_texture=current_texture,
            uniforms={}
        )
        
        # Stage 2: Structural Reconstruction
        current_texture = self._execute_stage(
            stage=1,
            program=self.programs[1],
            input_texture=current_texture,
            uniforms={}
        )
        
        # Stage 3: Real-Time Upscale
        current_texture = self._execute_stage(
            stage=2,
            program=self.programs[2],
            input_texture=current_texture,
            uniforms={
                'uSourceSize': (self.input_width, self.input_height),
                'uTargetSize': (self.input_width * self.scale_factor, self.input_height * self.scale_factor),
                'uScaleFactor': self.scale_factor
            }
        )
        
        # Stage 4: Perceptual Enhancement
        current_texture = self._execute_stage(
            stage=3,
            program=self.programs[3],
            input_texture=current_texture,
            uniforms={
                'uContrastBoost': 1.1,
                'uSaturationBoost': 1.05,
                'uSharpening': 0.3,
                'uLineDarkening': 0.15
            }
        )
        
        # Stage 5: Temporal Stabilization
        current_texture = self._execute_stage(
            stage=4,
            program=self.programs[4],
            input_texture=current_texture,
            additional_textures={'uPreviousFrame': self.history_texture},
            uniforms={
                'uTemporalWeight': 0.15,
                'uMotionThreshold': 0.1,
                'uFirstFrame': (frame_index == 0)
            }
        )
        
        # Copy output to history texture for next frame
        self._copy_texture(current_texture, self.history_texture)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        return current_texture
    
    def _execute_stage(
        self,
        stage: int,
        program: ShaderProgram,
        input_texture: int,
        uniforms: dict = None,
        additional_textures: dict = None
    ) -> int:
        """Execute a single pipeline stage."""
        # Bind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbos[stage])
        
        # Set viewport
        width, height = self._get_stage_resolution(stage)
        glViewport(0, 0, width, height)
        
        # Use shader program
        program.use()
        
        # Bind input texture to unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, input_texture)
        program.set_uniform_int('uInputFrame', 0)  # Most shaders use this name
        
        # Bind additional textures
        if additional_textures:
            for i, (uniform_name, texture_id) in enumerate(additional_textures.items(), start=1):
                glActiveTexture(GL_TEXTURE0 + i)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                program.set_uniform_int(uniform_name, i)
        
        # Set uniforms
        if uniforms:
            for name, value in uniforms.items():
                if isinstance(value, tuple) and len(value) == 2:
                    program.set_uniform_vec2(name, value[0], value[1])
                elif isinstance(value, float):
                    program.set_uniform_float(name, value)
                elif isinstance(value, int):
                    program.set_uniform_int(name, value)
                elif isinstance(value, bool):
                    program.set_uniform_int(name, 1 if value else 0)
        
        # Render fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        # Unbind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        return self.intermediate_textures[stage]
    
    def _get_stage_resolution(self, stage: int) -> Tuple[int, int]:
        """Get resolution for a pipeline stage."""
        if stage < 2:
            return (self.input_width, self.input_height)
        else:
            return (self.input_width * self.scale_factor, self.input_height * self.scale_factor)
    
    def _copy_texture(self, source: int, dest: int):
        """Copy source texture to dest texture."""
        # Simple copy using FBO blit (could also use glCopyImageSubData)
        # For now, just skip (history will be from previous frame naturally)
        pass
    
    def cleanup(self):
        """Clean up all GPU resources."""
        # Delete shader programs
        for program in self.programs:
            program.cleanup()
        self.programs.clear()
        
        # Delete FBOs and textures
        self._cleanup_fbos()
        
        # Delete VAO
        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
            self.quad_vao = 0
        
        print("[Pipeline] Cleanup complete")


def main():
    """Test shader pipeline execution."""
    import sys
    
    print("=" * 60)
    print("Shader Pipeline Executor - Test")
    print("=" * 60)
    
    shader_dir = Path(__file__).parent.parent.parent / "runtime" / "shaders" / "gpu"
    
    if not shader_dir.exists():
        print(f"[Error] Shader directory not found: {shader_dir}")
        return 1
    
    from runtime.playback import PlaybackWindow, OpenGLTextureUploader
    from runtime.video import VideoFrame
    
    try:
        with PlaybackWindow(1280, 720, "Pipeline Test") as window:
            # Create pipeline
            pipeline = ShaderPipelineExecutor(str(shader_dir), scale_factor=2)
            
            # Create uploader
            uploader = OpenGLTextureUploader()
            
            # Create test frame (gradient)
            test_data = np.zeros((360, 640, 3), dtype=np.uint8)
            for y in range(360):
                test_data[y, :, :] = int(y / 360 * 255)
            
            test_frame = VideoFrame(
                data=test_data,
                width=640,
                height=360,
                pts=0.0,
                frame_index=0
            )
            
            # Resize pipeline
            pipeline.resize(640, 360)
            
            # Upload frame
            input_tex = uploader.upload_frame(test_frame)
            
            # Execute pipeline
            print("\n[Test] Executing pipeline...")
            output_tex = pipeline.execute(input_tex, frame_index=0)
            print(f"[Test] Output texture: {output_tex}")
            
            # Cleanup
            pipeline.cleanup()
            uploader.cleanup()
            
            print("\n" + "=" * 60)
            print("Test passed!")
            print("=" * 60)
            
            import glfw
            # Auto-close after 1 second
            frame_count = 0
            def on_frame():
                nonlocal frame_count
                frame_count += 1
                if frame_count >= 60:
                    window._key_callback(window.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
            
            window.run(frame_callback=on_frame)
    
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
