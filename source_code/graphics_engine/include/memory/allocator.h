#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <mutex>

namespace engine {
namespace memory {

// ═══════════════════════════════════════════════
// Aligned allocation utilities
// ═══════════════════════════════════════════════
inline void* alignedAlloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) ptr = nullptr;
#endif
    return ptr;
}

inline void alignedFree(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}


// ═══════════════════════════════════════════════
// AlignedAllocator<T, Alignment>
// STL-compatible allocator that guarantees alignment (e.g. 16 or 32 for SIMD).
// ═══════════════════════════════════════════════
template<typename T, size_t Alignment = 16>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        pointer p = static_cast<pointer>(alignedAlloc(n * sizeof(T), Alignment));
        if (!p) throw std::bad_alloc();
        return p;
    }

    void deallocate(pointer p, size_type) noexcept {
        alignedFree(p);
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>& o) const noexcept { return !(*this == o); }
};


// ═══════════════════════════════════════════════
// PoolAllocator<T>
//
// Fixed-size block allocator.  Eliminates fragmentation for
// hot objects (vertices, draw calls, lights, particles).
//
// Layout per block:
//   [next_ptr | T data]
//
// Free list is a singly-linked list through the data region.
// ═══════════════════════════════════════════════
template<typename T>
class PoolAllocator {
    static_assert(sizeof(T) >= sizeof(void*),
        "PoolAllocator requires T >= pointer size. Wrap small types.");

public:
    explicit PoolAllocator(size_t initialCapacity = 256) : m_capacity(0), m_size(0) {
        m_blockSize = std::max(sizeof(T), sizeof(void*));
        m_buffer = nullptr;
        m_freeHead = nullptr;
        reserve(initialCapacity);
    }

    ~PoolAllocator() {
        alignedFree(m_buffer);
    }

    // Non-copyable, movable
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    PoolAllocator(PoolAllocator&& o) noexcept
        : m_buffer(o.m_buffer), m_freeHead(o.m_freeHead),
          m_capacity(o.m_capacity), m_size(o.m_size), m_blockSize(o.m_blockSize)
    { o.m_buffer = nullptr; o.m_freeHead = nullptr; o.m_capacity = 0; o.m_size = 0; }

    T* allocate() {
        if (m_size >= m_capacity) {
            reserve(m_capacity ? m_capacity * 2 : 64);
        }
        T* ptr;
        if (m_freeHead) {
            ptr = reinterpret_cast<T*>(m_freeHead);
            m_freeHead = *reinterpret_cast<void**>(m_freeHead);
        } else {
            // Allocate from end of used region
            ptr = reinterpret_cast<T*>(static_cast<char*>(m_buffer) + m_size * m_blockSize);
        }
        ++m_size;
        return ptr;
    }

    void deallocate(T* ptr) noexcept {
        if (!ptr) return;
        // Push onto free list
        *reinterpret_cast<void**>(ptr) = m_freeHead;
        m_freeHead = ptr;
        --m_size;
    }

    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        new (ptr) T(std::forward<Args>(args)...);
        return ptr;
    }

    void destroy(T* ptr) {
        if (ptr) { ptr->~T(); deallocate(ptr); }
    }

    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }

    // Reset the entire pool (no destructors — caller must manage lifetimes)
    void reset() {
        m_freeHead = nullptr;
        m_size = 0;
    }

private:
    void reserve(size_t newCap) {
        if (newCap <= m_capacity) return;

        void* newBuf = alignedAlloc(newCap * m_blockSize, 64); // 64-byte cache-line aligned
        if (!newBuf) throw std::bad_alloc();

        if (m_buffer) {
            std::memcpy(newBuf, m_buffer, m_capacity * m_blockSize);
            // Rebuild free list — old pointers are stale
            m_freeHead = nullptr;
            // Re-link any freed slots (we can't track them after copy, so just reset free list)
            // This is safe because reserve only grows; freed slots are logically lost on grow.
        }

        alignedFree(m_buffer);
        m_buffer = newBuf;
        m_capacity = newCap;
    }

    void* m_buffer;
    void* m_freeHead;
    size_t m_capacity;
    size_t m_size;
    size_t m_blockSize;
};


// ═══════════════════════════════════════════════
// ArenaAllocator
//
// Frame-scoped bump allocator.  Allocations are O(1).
// Everything is freed at once via reset() — ideal for
// per-frame temporaries (draw call lists, sort keys, etc).
// ═══════════════════════════════════════════════
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity = 1024 * 1024) // default 1 MB
        : m_capacity(capacity), m_offset(0)
    {
        m_buffer = static_cast<char*>(alignedAlloc(capacity, 64));
        if (!m_buffer) throw std::bad_alloc();
    }

    ~ArenaAllocator() { alignedFree(m_buffer); }

    // Non-copyable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Allocate n bytes with given alignment
    void* allocate(size_t size, size_t alignment = 8) {
        size_t aligned = (m_offset + alignment - 1) & ~(alignment - 1);
        if (aligned + size > m_capacity) {
            throw std::bad_alloc(); // Arena exhausted
        }
        m_offset = aligned + size;
        return m_buffer + aligned;
    }

    // Typed allocation
    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        T* ptr = static_cast<T*>(allocate(sizeof(T), alignof(T)));
        new (ptr) T(std::forward<Args>(args)...);
        return ptr;
    }

    // Allocate an array (no construction)
    template<typename T>
    T* allocateArray(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    // Reset — frees everything in O(1)
    void reset() { m_offset = 0; }

    size_t used() const { return m_offset; }
    size_t capacity() const { return m_capacity; }
    float usageRatio() const { return static_cast<float>(m_offset) / static_cast<float>(m_capacity); }

private:
    char* m_buffer;
    size_t m_capacity;
    size_t m_offset;
};


// ═══════════════════════════════════════════════
// PoolStats / global tracking (lightweight)
// ═══════════════════════════════════════════════
struct MemoryStats {
    size_t totalAllocated = 0; // bytes currently live
    size_t peakAllocated = 0; // high-water mark
    size_t totalAllocations = 0; // lifetime alloc count
    size_t totalFrees = 0; // lifetime free count

    size_t fragmentation() const {
        // Simple heuristic: (total - peak*0.75) / peak, clamped
        if (peakAllocated == 0) return 0;
        size_t waste = (totalAllocated > peakAllocated * 3 / 4)
                     ? totalAllocated - peakAllocated * 3 / 4 : 0;
        return (waste * 100) / peakAllocated;
    }
};

} // namespace memory
} // namespace engine
