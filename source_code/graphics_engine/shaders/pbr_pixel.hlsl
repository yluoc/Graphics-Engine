// ═══════════════════════════════════════════════
// pbr_pixel.hlsl
//
// Physically Based Rendering Pixel Shader
//
// Implements the Cook-Torrance microfacet BRDF:
//   f(l,v) = (D · G · F) / (4 · NdotL · NdotV)
//
// Light models:
//   • Directional
//   • Point
//   • Spot (with soft edges)
//
// Shadow mapping:
//   • Single shadow map (directional light)
//   • 3×3 PCF (Percentage Closer Filtering)
// ═══════════════════════════════════════════════

// ─── Constant Buffers ───────────────────────────
cbuffer PerObjectConstants : register(b0)
{
    float4x4 g_WorldMatrix;
    float4x4 g_NormalMatrix;
};

cbuffer PerFrameConstants : register(b1)
{
    float4x4 g_ViewMatrix;
    float4x4 g_ProjectionMatrix;
    float3 g_CameraPosition;
    float g_Pad0;
};

cbuffer MaterialConstants : register(b2)
{
    float3 g_Albedo;
    float g_Metallic;
    float g_Roughness;
    float g_EmissiveScale;
    float3 g_EmissiveColor;
    float g_AOScale;
};

// ─── Light data (array, max 32 lights) ──────────
struct LightData
{
    float3 Position; // World-space position (point/spot)
    float Intensity;

    float3 Direction; // Normalized direction (directional/spot)
    float Range; // Attenuation range (point/spot)

    float3 Color;
    int Type; // 0=Directional, 1=Point, 2=Spot

    float SpotAngle; // Half-angle (radians)
    float SpotSoftness; // Softness band
    float ShadowBias;
    float Pad;
};

cbuffer LightConstants : register(b3)
{
    int g_LightCount;
    float3 g_AmbientColor; // Environment / ambient fill
    LightData g_Lights[32];
};

// ─── Shadow Map Constants ───────────────────────
cbuffer ShadowConstants : register(b4)
{
    float4x4 g_ShadowViewProj;      // Light's view-projection matrix
    float g_ShadowBias;
    float g_ShadowMapSize; // e.g. 2048.0
    float2 g_Pad1;
};

// ─── Textures & Samplers ────────────────────────
Texture2D g_ShadowMap : register(t0);
SamplerState g_ShadowSampler : register(s0); // Border, clamp
SamplerState g_MaterialSampler : register(s1); // Linear, wrap

// ─── Vertex Shader Output (Pixel Shader Input) ─
struct PS_INPUT
{
    float4 ClipPosition : SV_Position;
    float3 WorldPosition : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float3 WorldTangent : TEXCOORD2;
    float3 WorldBinormal : TEXCOORD3;
    float2 TexCoord : TEXCOORD4;
};


// ═══════════════════════════════════════════════
// PBR Helper Functions
// ═══════════════════════════════════════════════

// ── GGX / Trowbridge-Reitz Normal Distribution Function ──
// D(h) = α² / (π · ((n·h)²·(α²-1) + 1)²)
float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness; // Roughness remapping
    float a2 = a * a;
    float NdotH = saturate(dot(N, H));
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    denom = 3.14159265358979f * denom * denom;

    return a2 / denom;
}

// ── Smith's Masking-Shadowing (GX1 approximation) ──
// G1(v) = NdotV / (NdotV · (1 - k) + k),  k = α/2
float GeometrySmithG1(float NdotV, float roughness)
{
    float k = (roughness * roughness) * 0.5f; // Remapped k for IBL
    return NdotV / (NdotV * (1.0f - k) + k);
}

// ── Smith's Joint Masking-Shadowing G(l,v) = G1(l)·G1(v) ──
float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    return GeometrySmithG1(NdotV, roughness) * GeometrySmithG1(NdotL, roughness);
}

// ── Schlick Fresnel approximation ──
// F(v,h) = F0 + (1-F0)·(1 - v·h)^5
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

// ── Linearize sRGB → linear ──
float3 SRGBToLinear(float3 srgb)
{
    return pow(max(srgb, 0.0f), 2.2f);
}

// ── Tone mapping (simple Reinhard) ──
float3 ToneMap(float3 hdr)
{
    return hdr / (1.0f + hdr);
}

// ── Linear → sRGB gamma ──
float3 LinearToSRGB(float3 linear)
{
    return pow(max(linear, 0.0f), 1.0f / 2.2f);
}


// ═══════════════════════════════════════════════
// Shadow Sampling (3×3 PCF)
// ═══════════════════════════════════════════════
float SampleShadow(float3 worldPos)
{
    // Transform to shadow (light) space
    float4 shadowPos = mul(float4(worldPos, 1.0f), g_ShadowViewProj);
    shadowPos.xyz /= shadowPos.w; // Perspective divide

    // Map from NDC [-1,1] to texture [0,1]
    shadowPos.x = shadowPos.x * 0.5f + 0.5f;
    shadowPos.y = -shadowPos.y * 0.5f + 0.5f; // Flip Y (NDC vs texture)

    // Check bounds
    if (shadowPos.x < 0.0f || shadowPos.x > 1.0f ||
        shadowPos.y < 0.0f || shadowPos.y > 1.0f)
        return 1.0f; // Outside shadow map → fully lit

    float shadow = 0.0f;
    float texelSize = 1.0f / g_ShadowMapSize;

    // 3×3 PCF kernel
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            float depth = g_ShadowMap.Sample(g_ShadowSampler,
                                shadowPos.xy + offset).r;

            // Compare: if fragment is behind shadow map depth (+ bias), it's in shadow
            shadow += (shadowPos.z - g_ShadowBias > depth) ? 0.0f : 1.0f;
        }
    }
    return shadow / 9.0f; // Average PCF
}


// ═══════════════════════════════════════════════
// Light Evaluation
// ═══════════════════════════════════════════════
float3 EvaluateLight(LightData light, float3 N, float3 V, float3 worldPos,
                     float3 albedo, float metallic, float roughness)
{
    float3 L; // Light direction (toward light)
    float attenuation = 1.0f;

    if (light.Type == 0)
    {
        // ── Directional ──
        L = normalize(-light.Direction);
    }
    else if (light.Type == 1)
    {
        // ── Point ──
        float3 toLight = light.Position - worldPos;
        float dist = length(toLight);
        L = toLight / dist;

        // Inverse-square falloff, clamped at range
        attenuation = saturate(1.0f - (dist / light.Range));
        attenuation = attenuation * attenuation; // Smooth falloff
    }
    else
    {
        // ── Spot ──
        float3 toLight = light.Position - worldPos;
        float dist = length(toLight);
        L = toLight / dist;

        // Distance attenuation
        attenuation = saturate(1.0f - (dist / light.Range));
        attenuation = attenuation * attenuation;

        // Cone attenuation
        float cosAngle = dot(L, normalize(-light.Direction));
        float cosInner = cos(light.SpotAngle);
        float cosOuter = cos(light.SpotAngle + light.SpotSoftness);
        float spotFactor = saturate((cosAngle - cosOuter) / (cosInner - cosOuter));
        attenuation *= spotFactor * spotFactor; // Smooth edge
    }

    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0f || attenuation <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);

    // ── Cook-Torrance BRDF ──
    float3 H = normalize(V + L);

    // Base reflectance at normal incidence
    // Dielectrics ≈ 0.04, metals = albedo
    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(saturate(dot(H, V)), F0);

    // Specular numerator
    float3 specular = (D * G * F) / max(4.0f * saturate(dot(N, V)) * NdotL, 0.001f);

    // Diffuse: Lambertian, energy-conserving (1-F) and zero for metals
    float3 kD = (1.0f - F) * (1.0f - metallic);
    float3 diffuse = kD * albedo / 3.14159265358979f;

    // Combined radiance from this light
    float3 radiance = light.Color * light.Intensity * attenuation;
    return (diffuse + specular) * radiance * NdotL;
}


// ═══════════════════════════════════════════════
// Main Pixel Shader
// ═══════════════════════════════════════════════
float4 PS_Main(PS_INPUT input) : SV_Target
{
    // ── Reconstruct shading vectors ──
    float3 N = normalize(input.WorldNormal);
    float3 V = normalize(g_CameraPosition - input.WorldPosition);

    // ── Material parameters (could be textured; here using constants) ──
    float3 albedo = SRGBToLinear(g_Albedo);
    float metallic = g_Metallic;
    float roughness = clamp(g_Roughness, 0.04f, 1.0f);

    // ── Shadow test (for the first directional light that casts shadows) ──
    float shadow = 1.0f;
    for (int i = 0; i < g_LightCount; ++i)
    {
        if (g_Lights[i].Type == 0) // Directional
        {
            shadow = SampleShadow(input.WorldPosition);
            break; // Only one shadow map supported
        }
    }

    // ── Accumulate lighting ──
    float3 finalColor = float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < g_LightCount; ++i)
    {
        float3 contrib = EvaluateLight(g_Lights[i], N, V,
                                       input.WorldPosition,
                                       albedo, metallic, roughness);

        // Apply shadow only to directional light contribution
        if (g_Lights[i].Type == 0)
            contrib *= shadow;

        finalColor += contrib;
    }

    // ── Ambient / Environment fill ──
    // Simple hemisphere model: lerp between ground and sky by up-vector alignment
    float skyFactor = saturate(dot(N, float3(0.0f, 1.0f, 0.0f)) * 0.5f + 0.5f);
    float3 ambient = lerp(g_AmbientColor * 0.3f, g_AmbientColor, skyFactor);
    float3 kD_ambient = (1.0f - metallic); // Metals don't have diffuse ambient
    finalColor += kD_ambient * albedo * ambient * g_AOScale;

    // ── Emissive ──
    finalColor += SRGBToLinear(g_EmissiveColor) * g_EmissiveScale;

    // ── Tone mapping + gamma ──
    finalColor = ToneMap(finalColor);
    finalColor = LinearToSRGB(finalColor);

    return float4(finalColor, 1.0f);
}
