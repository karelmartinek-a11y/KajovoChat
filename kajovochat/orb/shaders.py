VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 v_uv;
out vec4 f_color;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_radius;
uniform float u_core_radius;
uniform float u_core_intensity;
uniform float u_glow_intensity;
uniform float u_aura_radius;
uniform float u_aura_intensity;
uniform float u_shell_deformation;
uniform float u_warp_strength;
uniform float u_turbulence;
uniform float u_shimmer;
uniform float u_interference;
uniform float u_focus;
uniform float u_thinking_activity;
uniform float u_speaking_mix;
uniform float u_listening_tension;
uniform float u_speaking_boost;
uniform float u_transient_spike;
uniform float u_detail_sharpness;
uniform float u_low_pulse;
uniform float u_mid_motion;
uniform float u_high_shimmer;
uniform float u_background_intensity;
uniform vec3 u_background_color;
uniform vec3 u_haze_color;
uniform vec3 u_core_color;
uniform vec3 u_glow_color;
uniform vec3 u_aura_color;
uniform vec3 u_edge_color;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.55;
    for (int i = 0; i < 5; i++) {
        value += amp * noise(p);
        p = p * 2.03 + vec2(19.7, 7.3);
        amp *= 0.52;
    }
    return value;
}

void main() {
    vec2 aspect = vec2(u_resolution.x / max(u_resolution.y, 1.0), 1.0);
    vec2 p = (v_uv - 0.5) * 2.0 * aspect;
    float r = length(p);
    float angle = atan(p.y, p.x);

    float slow_t = u_time * 0.16;
    float medium_t = u_time * (0.48 + u_mid_motion * 0.35 + u_thinking_activity * 0.18);
    float fast_t = u_time * (1.85 + u_high_shimmer * 1.1 + u_speaking_mix * 0.55);

    vec2 warp_input = p * (2.4 + u_focus * 1.2);
    vec2 warp = vec2(
        fbm(warp_input + vec2(slow_t, medium_t)),
        fbm(warp_input + vec2(13.1 - medium_t, 5.7 + slow_t))
    );
    vec2 shell_uv = p + (warp - 0.5) * u_warp_strength;

    float shell_noise = fbm(shell_uv * (3.0 + u_turbulence * 4.5) + vec2(0.0, medium_t));
    float shell_noise2 = fbm(shell_uv.yx * (5.6 + u_detail_sharpness * 5.0) + vec2(fast_t, -fast_t * 0.6));
    float domain_noise = mix(shell_noise, shell_noise2, 0.45 + u_detail_sharpness * 0.25);
    float interference = sin(p.y * (90.0 + u_listening_tension * 46.0) + medium_t * 8.0) * 0.5 + 0.5;
    float micro = fbm(shell_uv * (17.0 + u_detail_sharpness * 22.0) + vec2(0.0, fast_t * 1.9));

    float deformed_radius = u_radius
        + (domain_noise - 0.5) * u_shell_deformation
        + (interference - 0.5) * u_interference
        + sin(angle * 6.0 + medium_t * 1.5) * 0.010 * (0.4 + u_mid_motion);
    float orb_mask = smoothstep(deformed_radius + 0.030, deformed_radius - 0.018, r);

    float radial = smoothstep(deformed_radius, 0.0, r);
    float inner_field = smoothstep(u_core_radius + 0.16, u_core_radius * 0.35, r);
    float shell_band = smoothstep(deformed_radius + 0.01, deformed_radius - 0.10, r);
    float aura = smoothstep(u_aura_radius + 0.30, u_aura_radius - 0.02, r);
    float background_haze = fbm(p * 1.1 + vec2(slow_t, -slow_t * 0.7));
    float edge = pow(clamp(1.0 - abs(r - deformed_radius) / 0.07, 0.0, 1.0), 2.4);

    float thinking_field = fbm(vec2(angle * 2.8, r * 6.0 + medium_t * 0.7));
    float speaking_spark = pow(clamp(1.0 - abs(fract(angle / 3.14159 * 5.0 + fast_t * 0.20) - 0.5) * 2.0, 0.0, 1.0), 4.0);
    speaking_spark *= (0.20 + u_transient_spike * 0.85 + u_speaking_boost * 1.4) * shell_band;

    vec3 color = u_background_color;
    color += u_haze_color * (0.20 + background_haze * 0.55) * u_background_intensity;
    color += u_aura_color * aura * (0.20 + u_aura_intensity * 0.42 + u_low_pulse * 0.18);
    color += u_glow_color * radial * (0.22 + u_glow_intensity * 0.46 + shell_noise * 0.18);
    color += u_core_color * inner_field * (0.18 + u_core_intensity * 0.62 + u_low_pulse * 0.12);
    color += u_edge_color * edge * (0.20 + micro * 0.26 + u_high_shimmer * 0.36);
    color += u_glow_color * shell_band * (domain_noise * 0.18 + interference * 0.10 + u_turbulence * 0.30);
    color += u_edge_color * micro * shell_band * (u_shimmer * 0.28 + u_high_shimmer * 0.22);
    color += u_core_color * shell_band * speaking_spark;
    color += u_glow_color * shell_band * thinking_field * u_thinking_activity * 0.22;

    float vignette = smoothstep(1.45, 0.18, r);
    color *= vignette;
    color = pow(max(color, vec3(0.0)), vec3(0.92));
    float alpha = clamp(max(vignette * 0.9, aura * 0.58) + orb_mask * 0.34, 0.0, 1.0);
    f_color = vec4(color, alpha);
}
"""
