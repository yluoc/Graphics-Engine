// ═══════════════════════════════════════════════
// shadow_vertex.hlsl
//
// Shadow Map Vertex Shader
//
// Renders geometry from the light's point of view.
// Only outputs clip-space position — no lighting
// or interpolation needed.  The depth buffer write
// IS the shadow map.
// ═══════════════════════════════════════════════

// ─── Constant Buffers ───────────────────────────
cbuffer PerObjectConstants : register(b0)
{
    float4x4 g_WorldMatrix;
    float4x4 g_NormalMatrix; // Unused in shadow pass, but kept for CB layout compatibility
};

cbuffer ShadowPassConstants : register(b1)
{
    float4x4 g_LightViewMatrix; // World → Light view space
    float4x4 g_LightProjectionMatrix; // Light view → Clip
};

// ─── Vertex Input ───────────────────────────────
// Uses only position — we don't need normals/UVs for depth-only pass
struct VS_INPUT_SHADOW
{
    float3 Position : POSITION;
    // Remaining attributes (Normal, TexCoord, Tangent) are present in the
    // vertex buffer but ignored here via semantic binding.
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float3 Tangent : TANGENT;
};

// ─── Vertex Output ──────────────────────────────
struct VS_OUTPUT_SHADOW
{
    float4 ClipPosition : SV_Position;
};


// ═══════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════
VS_OUTPUT_SHADOW VS_Shadow_Main(VS_INPUT_SHADOW input)
{
    VS_OUTPUT_SHADOW output;

    // Object → World → Light View → Light Clip
    float4 worldPos = mul(float4(input.Position, 1.0f), g_WorldMatrix);
    float4 lightViewPos = mul(worldPos, g_LightViewMatrix);
    output.ClipPosition = mul(lightViewPos, g_LightProjectionMatrix);

    return output;
}
