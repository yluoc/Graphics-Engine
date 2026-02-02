// ═══════════════════════════════════════════════
// pbr_vertex.hlsl
//
// PBR Vertex Shader
// Transforms vertex data from object space → clip space.
// Outputs interpolants needed by the PBR pixel shader.
// ═══════════════════════════════════════════════

// ─── Constant Buffers ───────────────────────────
cbuffer PerObjectConstants : register(b0)
{
    float4x4 g_WorldMatrix; // Object → World
    float4x4 g_NormalMatrix; // Inverse-transpose of 3x3 (for normal transform)
};

cbuffer PerFrameConstants : register(b1)
{
    float4x4 g_ViewMatrix; // World → View
    float4x4 g_ProjectionMatrix; // View  → Clip
    float3 g_CameraPosition; // World-space camera position
    float g_Pad0;
};

// ─── Vertex Input ───────────────────────────────
struct VS_INPUT
{
    float3 Position : POSITION; // Object-space vertex position
    float3 Normal : NORMAL; // Object-space normal
    float2 TexCoord : TEXCOORD0; // UV coordinates
    float3 Tangent : TANGENT; // Object-space tangent (for TBN)
};

// ─── Vertex Output (interpolated to pixel shader) ─
struct VS_OUTPUT
{
    float4 ClipPosition : SV_Position; // Clip-space position
    float3 WorldPosition : TEXCOORD0; // World-space position (for lighting)
    float3 WorldNormal : TEXCOORD1; // World-space normal
    float3 WorldTangent : TEXCOORD2; // World-space tangent
    float3 WorldBinormal : TEXCOORD3; // World-space binormal (computed)
    float2 TexCoord : TEXCOORD4; // Passthrough UV
};


// ═══════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════
VS_OUTPUT VS_Main(VS_INPUT input)
{
    VS_OUTPUT output;

    // ── Position: Object → World → View → Clip ──
    float4 worldPos = mul(float4(input.Position, 1.0f), g_WorldMatrix);
    float4 viewPos = mul(worldPos, g_ViewMatrix);
    output.ClipPosition = mul(viewPos, g_ProjectionMatrix);

    // ── World-space position (for lighting calculations) ──
    output.WorldPosition = worldPos.xyz;

    // ── Normal transform: use the normal matrix (inverse-transpose)
    //    to correctly handle non-uniform scaling.
    //    We zero out w to treat it as a direction (no translation). ──
    float4 worldNormal = mul(float4(input.Normal, 0.0f), g_NormalMatrix);
    float4 worldTangent = mul(float4(input.Tangent, 0.0f), g_NormalMatrix);

    output.WorldNormal = normalize(worldNormal.xyz);
    output.WorldTangent = normalize(worldTangent.xyz);

    // ── Binormal = Normal × Tangent (right-hand rule) ──
    output.WorldBinormal = cross(output.WorldNormal, output.WorldTangent);

    // ── UV passthrough ──
    output.TexCoord = input.TexCoord;

    return output;
}
