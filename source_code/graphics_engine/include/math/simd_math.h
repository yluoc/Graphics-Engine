#pragma once

#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>   // SSE/AVX intrinsics
#include <array>
#include <numeric>

// ─────────────────────────────────────────────
// Compile-time SIMD feature detection
// ─────────────────────────────────────────────
#if defined(__AVX2__)
  #define ENGINE_USE_AVX2 1
#elif defined(__SSE4_1__)
  #define ENGINE_USE_SSE4 1
#elif defined(__SSE2__)
  #define ENGINE_USE_SSE2 1
#else
  #define ENGINE_USE_SCALAR 1
#endif

namespace engine {
namespace math {

// ═══════════════════════════════════════════════
// Vec2
// ═══════════════════════════════════════════════
struct alignas(8) Vec2 {
    float x, y;

    Vec2() : x(0.f), y(0.f) {}
    Vec2(float x, float y) : x(x), y(y) {}
    explicit Vec2(float v) : x(v), y(v) {}

    Vec2 operator+(const Vec2& r) const { return {x+r.x, y+r.y}; }
    Vec2 operator-(const Vec2& r) const { return {x-r.x, y-r.y}; }
    Vec2 operator*(float s) const { return {x*s, y*s}; }
    Vec2 operator/(float s) const { float inv=1.f/s; return {x*inv, y*inv}; }
    Vec2& operator+=(const Vec2& r) { x+=r.x; y+=r.y; return *this; }
    Vec2& operator-=(const Vec2& r) { x-=r.x; y-=r.y; return *this; }
    Vec2& operator*=(float s) { x*=s; y*=s; return *this; }

    float dot(const Vec2& r) const { return x*r.x + y*r.y; }
    float lengthSq() const { return dot(*this); }
    float length() const { return std::sqrt(lengthSq()); }
    Vec2 normalized() const { float l = length(); return l > 0.f ? *this / l : Vec2(); }
    void normalize() { float l = length(); if (l > 0.f) { x/=l; y/=l; } }

    static Vec2 zero() { return {0.f, 0.f}; }
    static Vec2 one() { return {1.f, 1.f}; }
};


// ═══════════════════════════════════════════════
// Vec3
// ═══════════════════════════════════════════════
struct alignas(16) Vec3 {
    float x, y, z, _pad; // 16-byte aligned for SIMD

    Vec3() : x(0.f), y(0.f), z(0.f), _pad(0.f) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z), _pad(0.f) {}
    explicit Vec3(float v) : x(v), y(v), z(v), _pad(0.f) {}

    Vec3 operator+(const Vec3& r) const { return {x+r.x, y+r.y, z+r.z}; }
    Vec3 operator-(const Vec3& r) const { return {x-r.x, y-r.y, z-r.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { float inv=1.f/s; return {x*inv, y*inv, z*inv}; }
    Vec3& operator+=(const Vec3& r) { x+=r.x; y+=r.y; z+=r.z; return *this; }
    Vec3& operator-=(const Vec3& r) { x-=r.x; y-=r.y; z-=r.z; return *this; }
    Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
    bool operator==(const Vec3& r) const { return x==r.x && y==r.y && z==r.z; }

    float dot(const Vec3& r) const { return x*r.x + y*r.y + z*r.z; }
    Vec3 cross(const Vec3& r) const {
        return { y*r.z - z*r.y,
                 z*r.x - x*r.z,
                 x*r.y - y*r.x };
    }
    float lengthSq() const { return dot(*this); }
    float length() const { return std::sqrt(lengthSq()); }
    Vec3 normalized() const { float l = length(); return l > 0.f ? *this / l : Vec3(); }
    void normalize() { float l = length(); if (l > 0.f) { x/=l; y/=l; z/=l; } }

    static Vec3 zero() { return {0.f, 0.f, 0.f}; }
    static Vec3 one() { return {1.f, 1.f, 1.f}; }
    static Vec3 up() { return {0.f, 1.f, 0.f}; }
    static Vec3 forward(){ return {0.f, 0.f,-1.f}; }
    static Vec3 right() { return {1.f, 0.f, 0.f}; }
};


// ═══════════════════════════════════════════════
// Vec4
// ═══════════════════════════════════════════════
struct alignas(16) Vec4 {
    float x, y, z, w;

    Vec4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    Vec4(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}
    explicit Vec4(float v) : x(v), y(v), z(v), w(v) {}

    Vec4 operator+(const Vec4& r) const { return {x+r.x, y+r.y, z+r.z, w+r.w}; }
    Vec4 operator-(const Vec4& r) const { return {x-r.x, y-r.y, z-r.z, w-r.w}; }
    Vec4 operator*(float s) const { return {x*s, y*s, z*s, w*s}; }
    Vec4& operator+=(const Vec4& r) { x+=r.x; y+=r.y; z+=r.z; w+=r.w; return *this; }
    Vec4& operator*=(float s) { x*=s; y*=s; z*=s; w*=s; return *this; }

    float dot(const Vec4& r) const { return x*r.x + y*r.y + z*r.z + w*r.w; }
    float lengthSq() const { return dot(*this); }
    float length() const { return std::sqrt(lengthSq()); }

    Vec3 xyz() const { return {x, y, z}; }

    // Raw pointer access for row extraction
    const float* data() const { return &x; }
    float* data() { return &x; }

    static Vec4 zero() { return {0.f, 0.f, 0.f, 0.f}; }
    static Vec4 one() { return {1.f, 1.f, 1.f, 1.f}; }
};


// ═══════════════════════════════════════════════
// Mat4 — column-major 4x4 matrix
// ═══════════════════════════════════════════════
struct alignas(64) Mat4 {
    // Stored as 4 column vectors, each 16-byte aligned
    Vec4 col[4];

    Mat4() { col[0]=col[1]=col[2]=col[3]=Vec4(); }

    // Identity constructor
    static Mat4 identity() {
        Mat4 m;
        m.col[0] = {1,0,0,0};
        m.col[1] = {0,1,0,0};
        m.col[2] = {0,0,1,0};
        m.col[3] = {0,0,0,1};
        return m;
    }

    // Element access  m[col][row]
    float& operator()(int c, int r) { return ((float*)&col[c])[r]; }
    float operator()(int c, int r) const { return ((const float*)&col[c])[r]; }

    // Raw data pointer (column-major)
    const float* data() const { return (const float*)col; }
    float* data() { return (float*)col; }

    // ── Multiplication (scalar fallback; SIMD version below) ──
    Mat4 operator*(const Mat4& r) const;

    // ── Transform a point (w=1) ──
    Vec3 transformPoint(const Vec3& v) const {
        Vec4 res;
        res.x = col[0].x*v.x + col[1].x*v.y + col[2].x*v.z + col[3].x;
        res.y = col[0].y*v.x + col[1].y*v.y + col[2].y*v.z + col[3].y;
        res.z = col[0].z*v.x + col[1].z*v.y + col[2].z*v.z + col[3].z;
        return res.xyz();
    }

    // ── Transform a direction (w=0) ──
    Vec3 transformDir(const Vec3& v) const {
        return {
            col[0].x*v.x + col[1].x*v.y + col[2].x*v.z,
            col[0].y*v.x + col[1].y*v.y + col[2].y*v.z,
            col[0].z*v.x + col[1].z*v.y + col[2].z*v.z
        };
    }

    // ── Common constructors ──
    static Mat4 translate(const Vec3& t) {
        Mat4 m = identity();
        m.col[3] = {t.x, t.y, t.z, 1.f};
        return m;
    }

    static Mat4 scale(const Vec3& s) {
        Mat4 m = identity();
        m.col[0].x = s.x;
        m.col[1].y = s.y;
        m.col[2].z = s.z;
        return m;
    }

    static Mat4 rotateY(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 m = identity();
        m.col[0].x = c; m.col[0].z = -s;
        m.col[2].x = s; m.col[2].z = c;
        return m;
    }

    static Mat4 rotateX(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 m = identity();
        m.col[1].y = c; m.col[1].z = s;
        m.col[2].y = -s; m.col[2].z = c;
        return m;
    }

    static Mat4 rotateZ(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 m = identity();
        m.col[0].x = c; m.col[0].y = s;
        m.col[1].x = -s; m.col[1].y = c;
        return m;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = (center - eye).normalized();
        Vec3 s = f.cross(up).normalized();
        Vec3 u = s.cross(f);
        Mat4 m = identity();
        m.col[0] = { s.x, u.x, -f.x, 0.f};
        m.col[1] = { s.y, u.y, -f.y, 0.f};
        m.col[2] = { s.z, u.z, -f.z, 0.f};
        m.col[3] = {-s.dot(eye), -u.dot(eye), f.dot(eye), 1.f};
        return m;
    }

    static Mat4 perspective(float fovRad, float aspect, float near, float far) {
        float f = 1.f / std::tan(fovRad * 0.5f);
        Mat4 m;
        m.col[0].x = f / aspect;
        m.col[1].y = f;
        m.col[2].z = (far + near) / (near - far);
        m.col[2].w = -1.f;
        m.col[3].z = (2.f * far * near) / (near - far);
        return m;
    }

    Mat4 transpose() const {
        Mat4 m;
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                m(c,r) = (*this)(r,c);
        return m;
    }

    // 3x3 inverse (for normal matrices)
    Mat4 inverse3x3() const {
        float m00=col[0].x, m01=col[1].x, m02=col[2].x;
        float m10=col[0].y, m11=col[1].y, m12=col[2].y;
        float m20=col[0].z, m21=col[1].z, m22=col[2].z;

        float det = m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20);
        float inv = (std::fabs(det) > 1e-10f) ? 1.f/det : 0.f;

        Mat4 r = identity();
        r.col[0].x = (m11*m22-m12*m21)*inv;
        r.col[0].y = (m12*m20-m10*m22)*inv;
        r.col[0].z = (m10*m21-m11*m20)*inv;
        r.col[1].x = (m02*m21-m01*m22)*inv;
        r.col[1].y = (m00*m22-m02*m20)*inv;
        r.col[1].z = (m01*m20-m00*m21)*inv;
        r.col[2].x = (m01*m12-m02*m11)*inv;
        r.col[2].y = (m02*m10-m00*m12)*inv;
        r.col[2].z = (m00*m11-m01*m10)*inv;
        return r;
    }
};


// ═══════════════════════════════════════════════
// SIMD Batch Operations
// ═══════════════════════════════════════════════

// Transform N points by a single Mat4 — the hot path.
// Uses SSE2 by default; AVX path if available at compile time.
void batchTransformPoints(const Mat4& m, const Vec3* input, Vec3* output, size_t count);

// Multiply two arrays of Mat4s element-wise (parent * child) — scene graph update.
void batchMultiplyMat4(const Mat4* parents, const Mat4* locals,
                       Mat4* results, size_t count);

// Dot-product of two Vec3 arrays into a float array (e.g. lighting NdotL)
void batchDot3(const Vec3* a, const Vec3* b, float* out, size_t count);


// ═══════════════════════════════════════════════
// Utility
// ═══════════════════════════════════════════════
inline float clamp(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline float radians(float deg) { return deg * 3.14159265358979323846f / 180.f; }
inline float degrees(float rad) { return rad * 180.f / 3.14159265358979323846f; }

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float HALF_PI = 1.57079632679489661923f;

} // namespace math
} // namespace engine
