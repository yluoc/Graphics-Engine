#include "render/pipeline.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <future>
#include <numeric>

namespace engine {
namespace render {

// ═══════════════════════════════════════════════
// Singleton
// ═══════════════════════════════════════════════
RenderPipeline& RenderPipeline::instance() {
    static RenderPipeline inst;
    return inst;
}

void RenderPipeline::setThreadCount(unsigned count) {
    m_threadCount = std::max(1u, count);
}


// ═══════════════════════════════════════════════
// Scene Graph — node CRUD
// ═══════════════════════════════════════════════
NodeHandle RenderPipeline::createNode(const std::string& name) {
    NodeHandle h = m_nextNode++;
    SceneNode& node = m_nodes[h];
    node.name = name;
    node.dirty = true;
    m_rootNodes.push_back(h);
    return h;
}

void RenderPipeline::destroyNode(NodeHandle handle) {
    auto it = m_nodes.find(handle);
    if (it == m_nodes.end()) return;

    // Remove from parent's children list
    SceneNode& node = it->second;
    if (node.parent != InvalidNode) {
        auto pit = m_nodes.find(node.parent);
        if (pit != m_nodes.end()) {
            auto& ch = pit->second.children;
            ch.erase(std::remove(ch.begin(), ch.end(), handle), ch.end());
        }
    } else {
        m_rootNodes.erase(std::remove(m_rootNodes.begin(), m_rootNodes.end(), handle), m_rootNodes.end());
    }

    // Recursively destroy children
    std::vector<NodeHandle> childrenCopy = node.children;
    for (auto c : childrenCopy) destroyNode(c);

    m_nodes.erase(it);
}

SceneNode* RenderPipeline::getNode(NodeHandle handle) {
    auto it = m_nodes.find(handle);
    return (it != m_nodes.end()) ? &it->second : nullptr;
}

void RenderPipeline::setParent(NodeHandle child, NodeHandle parent) {
    SceneNode* childNode = getNode(child);
    SceneNode* parentNode = getNode(parent);
    if (!childNode || !parentNode) return;

    // Remove from old parent / root list
    if (childNode->parent != InvalidNode) {
        SceneNode* oldParent = getNode(childNode->parent);
        if (oldParent) {
            auto& ch = oldParent->children;
            ch.erase(std::remove(ch.begin(), ch.end(), child), ch.end());
        }
    } else {
        m_rootNodes.erase(std::remove(m_rootNodes.begin(), m_rootNodes.end(), child), m_rootNodes.end());
    }

    childNode->parent = parent;
    parentNode->children.push_back(child);
    childNode->dirty = true;
}


// ═══════════════════════════════════════════════
// Lights
// ═══════════════════════════════════════════════
void RenderPipeline::addLight(const Light& light) { m_lights.push_back(light); }
void RenderPipeline::clearLights() { m_lights.clear(); }


// ═══════════════════════════════════════════════
// Materials & Shaders
// ═══════════════════════════════════════════════
MaterialHandle RenderPipeline::createMaterial(const PBRMaterial& mat) {
    MaterialHandle h = m_nextMaterial++;
    m_materials[h] = mat;
    return h;
}

ShaderHandle RenderPipeline::registerShader(const ShaderSource& src) {
    ShaderHandle h = m_nextShader++;
    m_shaders[h] = src;
    return h;
}


// ═══════════════════════════════════════════════
// Shadow Map
// ═══════════════════════════════════════════════
void RenderPipeline::setShadowMap(const ShadowMap& shadow) {
    m_shadowMap = shadow;
}


// ═══════════════════════════════════════════════
// Camera
// ═══════════════════════════════════════════════
void RenderPipeline::setCamera(const Vec3& eye, const Vec3& target, const Vec3& up,
                               float fovDeg, float aspect, float nearP, float farP) {
    m_cameraPos = eye;
    m_viewMatrix = Mat4::lookAt(eye, target, up);
    m_projMatrix = Mat4::perspective(radians(fovDeg), aspect, nearP, farP);
}


// ═══════════════════════════════════════════════
// Scene Graph Update — recursive, dirty-flag gated
// ═══════════════════════════════════════════════
void RenderPipeline::updateNode(NodeHandle handle, const Mat4& parentWorld) {
    auto it = m_nodes.find(handle);
    if (it == m_nodes.end()) return;
    SceneNode& node = it->second;

    if (node.dirty) {
        // Rebuild local transform from TRS
        node.localTransform = Mat4::translate(node.position)
                            * Mat4::rotateY(node.rotation.y)
                            * Mat4::rotateX(node.rotation.x)
                            * Mat4::rotateZ(node.rotation.z)
                            * Mat4::scale(node.scale);
        node.dirty = false;
    }

    node.worldTransform = parentWorld * node.localTransform;

    // Propagate to children
    for (NodeHandle child : node.children) {
        updateNode(child, node.worldTransform);
    }
}

void RenderPipeline::updateSceneGraph() {
    m_stats.totalNodes = static_cast<uint32_t>(m_nodes.size());
    for (NodeHandle root : m_rootNodes) {
        updateNode(root, Mat4::identity());
    }
}


// ═══════════════════════════════════════════════
// Frustum Culling
// ═══════════════════════════════════════════════

// Extract 6 frustum planes from the projection-view matrix.
// Our Mat4 is column-major, and operator* follows the convention
// that transformPoint does: result = M.col[0]*x + M.col[1]*y + ...
// For Gribb & Hartmann plane extraction we need the rows of the
// combined VP matrix.  Row r of a column-major mat is:
//     col[0][r], col[1][r], col[2][r], col[3][r]
void RenderPipeline::computeFrustumPlanes() {
    // Build VP = Proj * View using the same multiply as the rest of the engine
    Mat4 vp = m_projMatrix * m_viewMatrix;

    // Helper: extract row r as (a, b, c, d) where the plane equation is
    // a*x + b*y + c*z + d <= 0  means "outside"
    auto row = [&](int r) -> Vec4 {
        return { vp.col[0].data()[r],
                 vp.col[1].data()[r],
                 vp.col[2].data()[r],
                 vp.col[3].data()[r] };
    };

    Vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    // Plane extraction using Gribb & Hartmann (2001):
    //   Left  :  r3 + r0     Right :  r3 - r0
    //   Bottom:  r3 + r1     Top   :  r3 - r1
    //   Near  :  r3 + r2     Far   :  r3 - r2
    // Each plane normal = (a,b,c), dist = d  (point is inside when n·p + d >= 0)
    auto makePlane = [](Vec4 p) -> Plane {
        return { Vec3(p.x, p.y, p.z), p.w };
    };

    m_frustumPlanes[0] = makePlane({ r3.x+r0.x, r3.y+r0.y, r3.z+r0.z, r3.w+r0.w }); // Left
    m_frustumPlanes[1] = makePlane({ r3.x-r0.x, r3.y-r0.y, r3.z-r0.z, r3.w-r0.w }); // Right
    m_frustumPlanes[2] = makePlane({ r3.x+r1.x, r3.y+r1.y, r3.z+r1.z, r3.w+r1.w }); // Bottom
    m_frustumPlanes[3] = makePlane({ r3.x-r1.x, r3.y-r1.y, r3.z-r1.z, r3.w-r1.w }); // Top
    m_frustumPlanes[4] = makePlane({ r3.x+r2.x, r3.y+r2.y, r3.z+r2.z, r3.w+r2.w }); // Near
    m_frustumPlanes[5] = makePlane({ r3.x-r2.x, r3.y-r2.y, r3.z-r2.z, r3.w-r2.w }); // Far

    // Normalize each plane
    for (auto& p : m_frustumPlanes) {
        float len = p.normal.length();
        if (len > 1e-10f) {
            float inv = 1.f / len;
            p.normal *= inv;
            p.dist *= inv;
        }
    }
}

bool RenderPipeline::aabbInsideFrustum(const Vec3& min, const Vec3& max) const {
    for (int i = 0; i < 6; ++i) {
        const auto& p = m_frustumPlanes[i];
        // Find the corner of the AABB that is furthest in the direction of the plane normal
        Vec3 corner;
        corner.x = (p.normal.x > 0.f) ? max.x : min.x;
        corner.y = (p.normal.y > 0.f) ? max.y : min.y;
        corner.z = (p.normal.z > 0.f) ? max.z : min.z;

        float dist = p.normal.dot(corner) + p.dist;
        if (dist < 0.f) return false; // entirely outside this plane
    }
    return true;
}

void RenderPipeline::frustumCull() {
    computeFrustumPlanes();
    m_visibleNodes.clear();

    for (auto& [handle, node] : m_nodes) {
        if (!node.visible || node.mesh == InvalidMesh) continue;

        // Get mesh bounds
        Mesh* mesh = VertexManager::instance().getMesh(node.mesh);
        if (!mesh) continue;

        const MeshDescriptor& desc = mesh->descriptor();

        // Transform AABB corners by world matrix and find new AABB
        Vec3 corners[8] = {
            {desc.boundsMin.x, desc.boundsMin.y, desc.boundsMin.z},
            {desc.boundsMax.x, desc.boundsMin.y, desc.boundsMin.z},
            {desc.boundsMin.x, desc.boundsMax.y, desc.boundsMin.z},
            {desc.boundsMax.x, desc.boundsMax.y, desc.boundsMin.z},
            {desc.boundsMin.x, desc.boundsMin.y, desc.boundsMax.z},
            {desc.boundsMax.x, desc.boundsMin.y, desc.boundsMax.z},
            {desc.boundsMin.x, desc.boundsMax.y, desc.boundsMax.z},
            {desc.boundsMax.x, desc.boundsMax.y, desc.boundsMax.z},
        };

        Vec3 wsMin{1e30f, 1e30f, 1e30f}, wsMax{-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < 8; ++i) {
            Vec3 wc = node.worldTransform.transformPoint(corners[i]);
            wsMin.x = std::min(wsMin.x, wc.x); wsMin.y = std::min(wsMin.y, wc.y); wsMin.z = std::min(wsMin.z, wc.z);
            wsMax.x = std::max(wsMax.x, wc.x); wsMax.y = std::max(wsMax.y, wc.y); wsMax.z = std::max(wsMax.z, wc.z);
        }

        if (aabbInsideFrustum(wsMin, wsMax)) {
            m_visibleNodes.push_back(handle);
        }
    }

    m_stats.culledNodes = m_stats.totalNodes - static_cast<uint32_t>(m_visibleNodes.size());
    m_stats.visibleNodes = static_cast<uint32_t>(m_visibleNodes.size());
}


// ═══════════════════════════════════════════════
// Draw Call Construction + Batching + Instancing
// ═══════════════════════════════════════════════
void RenderPipeline::buildDrawCalls() {
    m_drawCalls.clear();
    m_batches.clear();

    // Group visible nodes by material for instancing
    std::unordered_map<MaterialHandle, std::vector<NodeHandle>> materialGroups;

    for (NodeHandle h : m_visibleNodes) {
        SceneNode* node = getNode(h);
        if (!node || node->mesh == InvalidMesh) continue;
        materialGroups[node->material].push_back(h);
    }

    // Build one DrawCall per material group.
    // If multiple nodes share (mesh + material), collapse into a single instanced draw call.
    for (auto& [mat, nodes] : materialGroups) {
        // Sub-group by mesh
        std::unordered_map<MeshHandle, std::vector<NodeHandle>> meshGroups;
        for (auto nh : nodes) {
            SceneNode* n = getNode(nh);
            if (n) meshGroups[n->mesh].push_back(nh);
        }

        DrawBatch batch;
        batch.material = mat;

        for (auto& [meshH, meshNodes] : meshGroups) {
            DrawCall dc;
            dc.mesh = meshH;
            dc.material = mat;

            if (meshNodes.size() == 1) {
                // Single instance — direct draw
                SceneNode* n = getNode(meshNodes[0]);
                dc.worldTransform = n->worldTransform;
                dc.normalMatrix = (n->worldTransform.inverse3x3()).transpose();
                dc.instanceCount = 1;
                dc.instanceTransforms = nullptr;
            } else {
                // Multiple instances — instanced draw.
                // Store instance transforms in a vector (simplified; a real engine
                // would upload to a GPU buffer).
                static thread_local std::vector<Mat4> instanceBuf;
                instanceBuf.clear();
                instanceBuf.reserve(meshNodes.size());
                for (auto nh : meshNodes) {
                    SceneNode* n = getNode(nh);
                    if (n) instanceBuf.push_back(n->worldTransform);
                }
                dc.worldTransform      = instanceBuf[0]; // "base" transform
                dc.normalMatrix = Mat4::identity();
                dc.instanceCount = static_cast<uint32_t>(instanceBuf.size());
                dc.instanceTransforms = instanceBuf.data();
            }

            batch.calls.push_back(dc);
        }

        m_batches.push_back(std::move(batch));
    }

    // Tally stats
    m_stats.totalDrawCalls = 0;
    m_stats.batchedDrawCalls = 0;
    m_stats.totalBatches = static_cast<uint32_t>(m_batches.size());
    for (auto& b : m_batches) {
        m_stats.totalDrawCalls += static_cast<uint32_t>(b.calls.size());
        for (auto& dc : b.calls) {
            m_stats.batchedDrawCalls += dc.instanceCount;
        }
    }
}


// ═══════════════════════════════════════════════
// Sort Batches  — by material ID (minimize state changes)
// ═══════════════════════════════════════════════
void RenderPipeline::sortBatches() {
    std::sort(m_batches.begin(), m_batches.end(),
        [](const DrawBatch& a, const DrawBatch& b) {
            return a.material < b.material;
        });
}


// ═══════════════════════════════════════════════
// Submit — multithreaded draw call preparation
//
// Splits batches across worker threads.  Each thread
// prepares its slice of draw calls (normal matrix,
// uniform bindings, etc.) independently.
// ═══════════════════════════════════════════════
void RenderPipeline::submit() {
    if (m_batches.empty()) return;

    unsigned numThreads = std::min(m_threadCount, static_cast<unsigned>(m_batches.size()));

    // Partition batches across threads
    size_t batchesPerThread = m_batches.size() / numThreads;
    size_t remainder = m_batches.size() % numThreads;

    std::vector<std::future<uint64_t>> futures;
    futures.reserve(numThreads);

    size_t offset = 0;
    for (unsigned t = 0; t < numThreads; ++t) {
        size_t count = batchesPerThread + (t < remainder ? 1 : 0);
        size_t start = offset;
        offset += count;

        futures.push_back(std::async(std::launch::async,
            [this, start, count]() -> uint64_t {
                uint64_t issued = 0;
                for (size_t i = start; i < start + count; ++i) {
                    DrawBatch& batch = m_batches[i];
                    for (DrawCall& dc : batch.calls) {
                        // Simulate per-draw-call work:
                        // - Compute normal matrix (if not instanced)
                        // - Bind uniforms (world, normal, material)
                        // - Issue draw

                        if (dc.instanceCount == 1) {
                            dc.normalMatrix = dc.worldTransform.inverse3x3().transpose();
                        }

                        // Count issued draw calls (each instance = 1 logical draw)
                        issued += dc.instanceCount;
                    }
                }
                return issued;
            }));
    }

    // Gather results
    uint64_t totalIssued = 0;
    for (auto& f : futures) {
        totalIssued += f.get();
    }
    m_stats.drawCallsIssued += totalIssued;
}

} // namespace render
} // namespace engine
