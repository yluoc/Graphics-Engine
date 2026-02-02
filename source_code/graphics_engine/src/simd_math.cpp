#include "math/simd_math.h"
#include <cstring>

namespace engine::math {

// ─────────────────────────────────────────────
// Mat4 * Mat4  (scalar — used as fallback)
// ─────────────────────────────────────────────
Mat4 Mat4::operator*(const Mat4& r) const {
    Mat4 res;
    for (int c = 0; c < 4; ++c) {
        res.col[c].x = col[0].x*r.col[c].x + col[1].x*r.col[c].y + col[2].x*r.col[c].z + col[3].x*r.col[c].w;
        res.col[c].y = col[0].y*r.col[c].x + col[1].y*r.col[c].y + col[2].y*r.col[c].z + col[3].y*r.col[c].w;
        res.col[c].z = col[0].z*r.col[c].x + col[1].z*r.col[c].y + col[2].z*r.col[c].z + col[3].z*r.col[c].w;
        res.col[c].w = col[0].w*r.col[c].x + col[1].w*r.col[c].y + col[2].w*r.col[c].z + col[3].w*r.col[c].w;
    }
    return res;
}


// ═══════════════════════════════════════════════
// SSE2 helpers
// ═══════════════════════════════════════════════
#if defined(__SSE2__) || defined(__AVX2__)
#include <emmintrin.h>  // SSE2

// Multiply one column of a Mat4 by a row vector stored in 4 separate __m128 registers
static inline __m128 sse_mat4_mul_col(
    __m128 col0, __m128 col1, __m128 col2, __m128 col3,
    __m128 row)
{
    // Broadcast each element of 'row' and accumulate
    __m128 x = _mm_shuffle_ps(row, row, _MM_SHUFFLE(0,0,0,0));
    __m128 y = _mm_shuffle_ps(row, row, _MM_SHUFFLE(1,1,1,1));
    __m128 z = _mm_shuffle_ps(row, row, _MM_SHUFFLE(2,2,2,2));
    __m128 w = _mm_shuffle_ps(row, row, _MM_SHUFFLE(3,3,3,3));

    __m128 res = _mm_mul_ps(col0, x);
    res = _mm_add_ps(res, _mm_mul_ps(col1, y));
    res = _mm_add_ps(res, _mm_mul_ps(col2, z));
    res = _mm_add_ps(res, _mm_mul_ps(col3, w));
    return res;
}
#endif


// ═══════════════════════════════════════════════
// batchTransformPoints  — Mat4 * Vec3 (w=1), N points
// ═══════════════════════════════════════════════
void batchTransformPoints(
    const Mat4& m, const Vec3* input, Vec3* output, size_t count)
{
#if defined(__SSE2__) || defined(__AVX2__)
    // Load the 4 columns of M into SSE registers (each column is a Vec4 = 16 bytes)
    __m128 c0 = _mm_load_ps((const float*)&m.col[0]);
    __m128 c1 = _mm_load_ps((const float*)&m.col[1]);
    __m128 c2 = _mm_load_ps((const float*)&m.col[2]);
    __m128 c3 = _mm_load_ps((const float*)&m.col[3]); // translation column

    size_t i = 0;
    for (; i < count; ++i) {
        // Load xyz, set w=1 for point transform
        // Vec3 is 16 bytes (x,y,z,_pad) so aligned load is safe
        __m128 p = _mm_load_ps((const float*)&input[i]);
        // Force w = 1.0
#if defined(__SSE4_1__) || defined(__AVX2__)
        p = _mm_blend_ps(p, _mm_set1_ps(1.0f), 0x8); // blend bit 3 = w
#else
        // SSE2 fallback: zero w, then OR in 1.0 for w
        const __m128 mask_xyz = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
        p = _mm_and_ps(p, mask_xyz);
        p = _mm_or_ps(p, _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f));
#endif

        // result = c0*px + c1*py + c2*pz + c3*1
        __m128 px = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0));
        __m128 py = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1));
        __m128 pz = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2));

        __m128 res = _mm_mul_ps(c0, px);
        res = _mm_add_ps(res, _mm_mul_ps(c1, py));
        res = _mm_add_ps(res, _mm_mul_ps(c2, pz));
        res = _mm_add_ps(res, c3); // +translation (w=1 implied)

        // Store result (only xyz used; _pad is harmless)
        _mm_store_ps((float*)&output[i], res);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        output[i] = m.transformPoint(input[i]);
    }
#endif
}


// ═══════════════════════════════════════════════
// batchMultiplyMat4  — parent[i] * local[i] → result[i]
// ═══════════════════════════════════════════════
void batchMultiplyMat4(
    const Mat4* parents, const Mat4* locals, Mat4* results, size_t count)
{
#if defined(__SSE2__) || defined(__AVX2__)
    for (size_t i = 0; i < count; ++i) {
        const Mat4& P = parents[i];
        const Mat4& L = locals[i];

        // Load parent columns
        __m128 p0 = _mm_load_ps((const float*)&P.col[0]);
        __m128 p1 = _mm_load_ps((const float*)&P.col[1]);
        __m128 p2 = _mm_load_ps((const float*)&P.col[2]);
        __m128 p3 = _mm_load_ps((const float*)&P.col[3]);

        // For each column of L, compute P * that column
        for (int c = 0; c < 4; ++c) {
            __m128 lc = _mm_load_ps((const float*)&L.col[c]);
            __m128 rc = sse_mat4_mul_col(p0, p1, p2, p3, lc);
            _mm_store_ps((float*)&results[i].col[c], rc);
        }
    }
#else
    for (size_t i = 0; i < count; ++i) {
        results[i] = parents[i] * locals[i];
    }
#endif
}


// ═══════════════════════════════════════════════
// batchDot3  — dot(a[i], b[i]) → out[i]
// ═══════════════════════════════════════════════
void batchDot3(
    const Vec3* a, const Vec3* b, float* out, size_t count)
{
#if defined(__SSE4_1__) || defined(__AVX2__)
#include <smmintrin.h>  // _mm_dp_ps (SSE4.1)
    size_t i = 0;
    for (; i < count; ++i) {
        __m128 va = _mm_load_ps((const float*)&a[i]);
        __m128 vb = _mm_load_ps((const float*)&b[i]);
        // dp mask 0x71: dot xyz (bits 0-2), store to element 0 only
        __m128 d = _mm_dp_ps(va, vb, 0x71);
        out[i] = _mm_cvtss_f32(d);
    }
#elif defined(__SSE2__)
    size_t i = 0;
    for (; i < count; ++i) {
        __m128 va = _mm_load_ps((const float*)&a[i]);
        __m128 vb = _mm_load_ps((const float*)&b[i]);
        __m128 mul = _mm_mul_ps(va, vb);
        // Horizontal add: x+y+z  (ignore w/_pad)
        // shuffle mul to get [y, x, w, z]
        __m128 s1 = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1,0,3,2)); // [z, w, x, y]
        __m128 sum = _mm_add_ps(mul, s1); // [x+z, y+w, ...]
        __m128 s2 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0,1,0,1)); // [y+w, x+z, ...]
        sum = _mm_add_ps(sum, s2); // [x+y+z+w, ...]
        // We need x+y+z only. Since _pad = 0 in Vec3, x+y+z+0 = correct.
        out[i] = _mm_cvtss_f32(sum);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i].dot(b[i]);
    }
#endif
}

} // namespace engine::math
