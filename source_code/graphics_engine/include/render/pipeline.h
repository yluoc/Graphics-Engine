#pragma once

#include "math/simd_math.h"
#include "gpu/vertex_manager.h"
#include "memory/allocator.h"
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_map>

namespace engine {
namespace render {

using namespace engine::math;
using namespace engine::gpu;

// ═══════════════════════════════════════════════
// Shader / Material IDs
// ═══════════════════════════════════════════════
using ShaderHandle = uint32_t;
using MaterialHandle = uint32_t;
constexpr ShaderHandle InvalidShader = 0;
constexpr MaterialHandle InvalidMaterial = 0;

enum class ShaderStage : uint8_t { Vertex, Pixel };

struct ShaderSource {
    std::string name;
    ShaderStage stage;
    std::string hlslCode; // HLSL source (stored as string)
};


// ═══════════════════════════════════════════════
// Light Models
// ═══════════════════════════════════════════════
enum class LightType : uint8_t { Directional, Point, Spot };

struct Light {
    LightType type = LightType::Directional;
    Vec3 position = Vec3::zero();
    Vec3 direction = Vec3(0.f, -1.f, 0.f); // directional / spot
    Vec3 color = Vec3::one();
    float intensity = 1.f;
    float range = 10.f; // point / spot
    float spotAngle = radians(45.f); // half-angle (radians)
    float spotSoftness = 0.2f; // falloff softness
    bool castsShadow = true;
};

// PBR material parameters
struct PBRMaterial {
    Vec3 albedo = {0.8f, 0.8f, 0.8f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float emissiveScale = 0.0f;
    Vec3 emissiveColor = Vec3::zero();
    float aoScale = 1.0f;
    // Texture handles would go here in a full engine
};

// Shadow map descriptor
struct ShadowMap {
    int resolution = 2048; // e.g. 2048x2048
    float nearPlane = 0.1f;
    float farPlane = 100.f;
    Mat4 lightProj; // projection from light's POV
    Mat4 lightView; // view from light's POV
    float bias = 0.005f; // shadow acne bias
};


// ═══════════════════════════════════════════════
// Scene Graph Node
// ═══════════════════════════════════════════════
using NodeHandle = uint32_t;
constexpr NodeHandle InvalidNode = 0;

struct SceneNode {
    // Local transform components
    Vec3 position = Vec3::zero();
    Vec3 rotation = Vec3::zero(); // Euler angles (radians)
    Vec3 scale = Vec3::one();

    // Computed world transform (updated during scene graph traversal)
    Mat4 localTransform = Mat4::identity();
    Mat4 worldTransform = Mat4::identity();

    // Graph structure
    NodeHandle parent = InvalidNode;
    std::vector<NodeHandle> children;

    // Payload
    MeshHandle mesh = InvalidMesh;
    MaterialHandle material = InvalidMaterial;
    bool visible = true;
    std::string name;

    // Dirty flag — only recompute world transform if something changed
    bool dirty = true;
};


// ═══════════════════════════════════════════════
// Draw Call  — what gets submitted to the GPU
// ═══════════════════════════════════════════════
struct DrawCall {
    MeshHandle mesh;
    MaterialHandle material;
    Mat4 worldTransform;
    Mat4 normalMatrix; // inverse-transpose of 3x3
    uint32_t instanceCount = 1;
    // Instancing data pointer (world transforms for each instance)
    const Mat4* instanceTransforms = nullptr;
};

// Batch: a group of draw calls that share the same material
// (allows state-change minimization)
struct DrawBatch {
    MaterialHandle material;
    std::vector<DrawCall> calls;
};


// ═══════════════════════════════════════════════
// Render Pipeline
// ═══════════════════════════════════════════════
class RenderPipeline {
public:
    static RenderPipeline& instance();

    // ── Scene Management ──
    NodeHandle createNode(const std::string& name = "");
    void destroyNode(NodeHandle handle);
    SceneNode* getNode(NodeHandle handle);
    void setParent(NodeHandle child, NodeHandle parent);

    // ── Light Management ──
    void addLight(const Light& light);
    void clearLights();
    const std::vector<Light>& lights() const { return m_lights; }

    // ── Material / Shader ──
    MaterialHandle createMaterial(const PBRMaterial& mat);
    ShaderHandle registerShader(const ShaderSource& src);

    // ── Shadow Mapping ──
    void setShadowMap(const ShadowMap& shadow);
    const ShadowMap& shadowMap() const { return m_shadowMap; }

    // ── Camera ──
    void setCamera(const Vec3& eye, const Vec3& target, const Vec3& up,
                             float fovDeg, float aspect, float nearP, float farP);
    const Mat4& viewMatrix() const { return m_viewMatrix; }
    const Mat4& projectionMatrix() const { return m_projMatrix; }

    // ── Per-Frame Pipeline ──
    // 1. Update scene graph (recompute world transforms)
    void updateSceneGraph();

    // 2. Frustum culling — removes invisible nodes
    void frustumCull();

    // 3. Build draw calls + batch by material
    void buildDrawCalls();

    // 4. Sort batches (by material to minimize state changes)
    void sortBatches();

    // 5. Submit — multithreaded draw call preparation
    void submit();

    // ── Stats ──
    struct FrameStats {
        uint32_t totalNodes = 0;
        uint32_t culledNodes = 0;
        uint32_t visibleNodes = 0;
        uint32_t totalDrawCalls = 0;
        uint32_t batchedDrawCalls = 0; // after instancing
        uint32_t totalBatches = 0;
        uint64_t drawCallsIssued = 0; // lifetime
    };
    const FrameStats& stats() const { return m_stats; }

    // Worker thread count
    void setThreadCount(unsigned count);
    unsigned threadCount() const { return m_threadCount; }

private:
    RenderPipeline() = default;

    // ── Scene Graph ──
    std::unordered_map<NodeHandle, SceneNode> m_nodes;
    NodeHandle m_nextNode = 1;
    std::vector<NodeHandle> m_rootNodes; // nodes with no parent
    std::vector<NodeHandle> m_visibleNodes; // post-culling

    // ── Lights ──
    std::vector<Light> m_lights;

    // ── Materials / Shaders ──
    std::unordered_map<MaterialHandle, PBRMaterial> m_materials;
    std::unordered_map<ShaderHandle, ShaderSource> m_shaders;
    MaterialHandle m_nextMaterial = 1;
    ShaderHandle m_nextShader = 1;

    // ── Shadow ──
    ShadowMap m_shadowMap;

    // ── Camera ──
    Mat4 m_viewMatrix;
    Mat4 m_projMatrix;
    Vec3 m_cameraPos;

    // ── Draw Calls ──
    std::vector<DrawCall> m_drawCalls;
    std::vector<DrawBatch> m_batches;

    // ── Frustum planes (6 planes, normal + distance) ──
    struct Plane { Vec3 normal; float dist; };
    Plane m_frustumPlanes[6];
    void computeFrustumPlanes();
    bool aabbInsideFrustum(const Vec3& min, const Vec3& max) const;

    // ── Scene graph traversal ──
    void updateNode(NodeHandle handle, const Mat4& parentWorld);

    // ── Multithreading ──
    unsigned m_threadCount = 4;

    // ── Stats ──
    FrameStats m_stats;
};

} // namespace render
} // namespace engine
