#version 330 core

uniform vec4 u_color;   // RGBA fill colour
out vec4 fragColor;

void main() {
    fragColor = u_color;
}
