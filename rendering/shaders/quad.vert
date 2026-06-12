#version 330 core

// Each quad is drawn as 2 triangles (6 vertices) generated entirely
// in the vertex shader from a single instance index.
// We pass the quad's NDC position and size as uniforms per draw call.
// This means zero per-vertex buffer uploads for every shape — all geometry
// is computed on the GPU from uniform data.

uniform vec2 u_pos;     // centre of the quad in NDC [-1, 1]
uniform vec2 u_size;    // half-extents in NDC space

void main() {
    // Build the 4 corners of the quad from gl_VertexID (0-5, two triangles)
    // Triangle 0: verts 0,1,2  — Triangle 1: verts 3,4,5
    vec2 corners[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0)
    );
    int indices[6] = int[6](0, 1, 2, 0, 2, 3);
    vec2 corner = corners[indices[gl_VertexID]];

    gl_Position = vec4(u_pos + corner * u_size, 0.0, 1.0);
}
