//=====================================================================
//
// SimdVector.h - 
//
// Created by skywind on 2020/04/22
// Last Modified: 2020/04/22 08:39:28
//
//=====================================================================
#ifndef _SIMD_VECTOR_H_
#define _SIMD_VECTOR_H_

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>


#ifndef __cplusplus
#error This file must be compiled in C++ mode !!
#endif


//---------------------------------------------------------------------
// Platform Define
//---------------------------------------------------------------------
#ifndef SIMD_HAS_NONE
#define SIMD_HAS_NONE		1
#endif

#ifndef SIMD_HAS_SSE2
#define SIMD_HAS_SSE2		0
#endif

#ifndef SIMD_HAS_AVX
#define SIMD_HAS_AVX		0
#endif

#ifndef SIMD_HAS_AVX2
#define SIMD_HAS_AVX2		0
#endif



//---------------------------------------------------------------------
// Instruction Coverage
//---------------------------------------------------------------------
#if SIMD_HAS_AVX2
#undef SIMD_HAS_AVX
#define SIMD_HAS_AVX		1
#endif

#if SIMD_HAS_AVX
#undef SIMD_HAS_SSE2
#define SIMD_HAS_SSE2		1
#endif

#if SIMD_HAS_SSE2
#undef SIMD_HAS_SSE
#define SIMD_HAS_SSE		1
#endif

#if SIMD_HAS_SSE
#undef SIMD_HAS_NONE
#define SIMD_HAS_NONE		0
#endif


//---------------------------------------------------------------------
// includes
//---------------------------------------------------------------------
#if SIMD_HAS_SSE2
#include <emmintrin.h>
#endif

#if SIMD_HAS_AVX
#include <pmmintrin.h>
#endif


//---------------------------------------------------------------------
// Namespace
//---------------------------------------------------------------------
#ifndef NAMESPACE_BEGIN
#define NAMESPACE_BEGIN(x)  namespace x {
#endif

#ifndef NAMESPACE_END
#define NAMESPACE_END(x) }
#endif


//---------------------------------------------------------------------
// Alignment
//---------------------------------------------------------------------
#ifdef __GNUC__
#define SIMD_ALIGNED_DATA(x) __attribute__ ((aligned(x)))
#define SIMD_ALIGNED_STRUCT(x) struct __attribute__ ((aligned(x)))
#else
#define SIMD_ALIGNED_DATA(x) __declspec(align(x))
#define SIMD_ALIGNED_STRUCT(x) __declspec(align(x)) struct
#endif

#ifndef CHECK_ALIGNMENT
#define CHECK_ALIGNMENT(ptr, size) \
	((reinterpret_cast<uintptr_t>(ptr) & (size - 1)) == 0)
#endif


//---------------------------------------------------------------------
// Namespace Begin
//---------------------------------------------------------------------
NAMESPACE_BEGIN(SIMD);


//---------------------------------------------------------------------
// vector of 4 elements
//---------------------------------------------------------------------
SIMD_ALIGNED_STRUCT(16) Xmm
{
	union {
		float f[4];
		uint32_t u[4];
		int32_t i[4];
#if SIMD_HAS_SSE2
		__m128 r;
#endif
	};
};


//---------------------------------------------------------------------
// Consts
//---------------------------------------------------------------------
SIMD_ALIGNED_STRUCT(16) XmmU { union { uint32_t u[4]; Xmm v; }; };
SIMD_ALIGNED_STRUCT(16) XmmI { union { int32_t u[4]; Xmm v; }; };
SIMD_ALIGNED_STRUCT(16) XmmF { union { int32_t u[4]; Xmm v; }; };


//---------------------------------------------------------------------
// Conversion
//---------------------------------------------------------------------

inline Xmm XmmConvertIntToFloat(const Xmm& x) {
#if SIMD_HAS_NONE
	Xmm y;
	y.f[0] = static_cast<float>(x.i[0]);
	y.f[1] = static_cast<float>(x.i[1]);
	y.f[2] = static_cast<float>(x.i[2]);
	y.f[3] = static_cast<float>(x.i[3]);
	return y;
#elif SIMD_HAS_SSE2
	Xmm y;
	y.r = _mm_cvtepi32_ps(_mm_castps_si128(x.r));
	return y;
#endif
}

inline Xmm XmmConvertFloatToInt(const Xmm& x) {
#if SIMD_HAS_NONE
	Xmm y;
	y.i[0] = static_cast<int32_t>(x.f[0]);
	y.i[1] = static_cast<int32_t>(x.f[1]);
	y.i[2] = static_cast<int32_t>(x.f[2]);
	y.i[3] = static_cast<int32_t>(x.f[3]);
	return y;
#elif SIMD_HAS_SSE2
	Xmm y;
	y.r = _mm_cvtepi32_ps(_mm_castps_si128(x.r));
	return y;
#endif
}


//---------------------------------------------------------------------
// Load/Store
//---------------------------------------------------------------------

// load 1 element from memory
inline Xmm XmmLoadM1(const void *ptr) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	Xmm s;
	const uint32_t *source = reinterpret_cast<const uint32_t*>(ptr);
	s.u[0] = source[0];
	s.u[1] = 0;
	s.u[2] = 0;
	s.u[3] = 0;
	return s;
#elif SIMD_HAS_SSE2
	Xmm s;
	s.r = _mm_load_ss(reinterpret_cast<const float*>(ptr));
	return s;
#endif
}

// load 2 elements from memory
inline Xmm XmmLoadM2(const void *ptr) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	Xmm s;
	const uint32_t *source = reinterpret_cast<const uint32_t*>(ptr);
	s.u[0] = source[0];
	s.u[1] = source[1];
	s.u[2] = 0;
	s.u[3] = 0;
	return s;
#elif SIMD_HAS_SSE2
	Xmm s;
	s.r = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(ptr)));
	return s;
#endif
}

// load 3 elements from memory
inline Xmm XmmLoadM3(const void *ptr) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	Xmm s;
	const uint32_t *source = reinterpret_cast<const uint32_t*>(ptr);
	s.u[0] = source[0];
	s.u[1] = source[1];
	s.u[2] = source[2];
	s.u[3] = 0;
	return s;
#elif SIMD_HAS_SSE2
	Xmm s;
	const float *source = reinterpret_cast<const float*>(ptr);
	__m128 xy = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(ptr)));
    __m128 z = _mm_load_ss(source + 2);
	s.r = _mm_movelh_ps(xy, z);
	return s;
#endif
}

// load 4 elements from memory
inline Xmm XmmLoadM4(const void *ptr) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	Xmm s;
	const uint32_t *source = reinterpret_cast<const uint32_t*>(ptr);
	s.u[0] = source[0];
	s.u[1] = source[1];
	s.u[2] = source[2];
	s.u[3] = source[3];
	return s;
#elif SIMD_HAS_AVX
	Xmm s;
	s.r = _mm_castsi128_ps(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(ptr)));
	return s;
#elif SIMD_HAS_SSE2
	Xmm s;
	s.r = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
	return s;
#endif
}

// store 1 element to memory
inline void XmmStoreM1(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	uint32_t *dest = reinterpret_cast<uint32_t*>(ptr);
	dest[0] = s.u[0];
#elif SIMD_HAS_SSE2
	_mm_store_ss(reinterpret_cast<float*>(ptr), s.r);
#endif
}

// store 2 elements to memory
inline void XmmStoreM2(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	uint32_t *dest = reinterpret_cast<uint32_t*>(ptr);
	dest[0] = s.u[0];
	dest[1] = s.u[1];
#elif SIMD_HAS_SSE2
	_mm_store_sd(reinterpret_cast<double*>(ptr), _mm_castps_pd(s.r));
#endif
}

// store 3 elements to memory
inline void XmmStoreM3(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	uint32_t *dest = reinterpret_cast<uint32_t*>(ptr);
	dest[0] = s.u[0];
	dest[1] = s.u[1];
	dest[2] = s.u[2];
#elif SIMD_HAS_SSE2
	float *dest = reinterpret_cast<float*>(ptr);
    _mm_store_sd(reinterpret_cast<double*>(ptr), _mm_castps_pd(s.r));
	uint32_t mask = _MM_SHUFFLE(2, 2, 2, 2);
    __m128 z = _mm_shuffle_ps(s.r, s.r, mask);
    _mm_store_ss(reinterpret_cast<float*>(&dest[2]), z);
#endif
}

// store 4 elements to memory
inline void XmmStoreM4(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	uint32_t *dest = reinterpret_cast<uint32_t*>(ptr);
	dest[0] = s.u[0];
	dest[1] = s.u[1];
	dest[2] = s.u[2];
	dest[3] = s.u[3];
#elif SIMD_HAS_SSE2
	_mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm_castps_si128(s.r));
#endif
}

// load 2 elements from aligned memory
inline Xmm XmmLoadA2(const void *ptr) noexcept {
	assert(ptr);
	assert(CHECK_ALIGNMENT(ptr, 16));
#if SIMD_HAS_NONE
	Xmm s;
	const uint32_t *source = reinterpret_cast<const uint32_t*>(ptr);
	s.u[0] = source[0];
	s.u[1] = source[1];
	s.u[2] = 0;
	s.u[3] = 0;
	return s;
#elif SIMD_HAS_SSE2
	Xmm s;
	s.r = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(ptr)));
	return s;
#endif
}

// load 4 elements from aligned memory
inline Xmm XmmLoadA4(const void *ptr) noexcept {
	assert(ptr);
	assert(CHECK_ALIGNMENT(ptr, 16));
#if SIMD_HAS_NONE
	Xmm s;
	const uint32_t *source = reinterpret_cast<const uint32_t*>(ptr);
	s.u[0] = source[0];
	s.u[1] = source[1];
	s.u[2] = 0;
	s.u[3] = 0;
	return s;
#elif SIMD_HAS_SSE2
	Xmm s;
	__m128i V = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    s.r = _mm_castsi128_ps(V);
	return s;
#endif
}

// store 2 elements to aligned memory
inline void XmmStoreA2(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
	assert(CHECK_ALIGNMENT(ptr, 16));
#if SIMD_HAS_NONE
	uint32_t *dest = reinterpret_cast<uint32_t*>(ptr);
	dest[0] = s.u[0];
	dest[1] = s.u[1];
#elif SIMD_HAS_SSE2
	_mm_store_sd(reinterpret_cast<double*>(ptr), _mm_castps_pd(s.r));
#endif
}

// store 4 elements to memory
inline void XmmStoreA4(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
	assert(CHECK_ALIGNMENT(ptr, 16));
#if SIMD_HAS_NONE
	uint32_t *dest = reinterpret_cast<uint32_t*>(ptr);
	dest[0] = s.u[0];
	dest[1] = s.u[1];
	dest[2] = s.u[2];
	dest[3] = s.u[3];
#elif SIMD_HAS_SSE2
	_mm_store_si128(reinterpret_cast<__m128i*>(ptr), _mm_castps_si128(s.r));
#endif
}


//---------------------------------------------------------------------
// load/store raw int/float
//---------------------------------------------------------------------
inline Xmm XmmLoadInt1(const uint32_t *source) { return XmmLoadM1(source); }
inline Xmm XmmLoadFloat1(const float *source) { return XmmLoadM1(source); }
inline Xmm XmmLoadInt2(const uint32_t *source) { return XmmLoadM2(source); }
inline Xmm XmmLoadFloat2(const float *source) { return XmmLoadM2(source); }
inline Xmm XmmLoadInt3(const uint32_t *source) { return XmmLoadM3(source); }
inline Xmm XmmLoadFloat3(const float *source) { return XmmLoadM3(source); }
inline Xmm XmmLoadInt4(const uint32_t *source) { return XmmLoadM4(source); }
inline Xmm XmmLoadFloat4(const float *source) { return XmmLoadM4(source); }
inline Xmm XmmLoadInt4A(const uint32_t *source) { return XmmLoadA4(source); }
inline Xmm XmmLoadFloat4A(const float *source) { return XmmLoadA4(source); }

inline void XmmStoreInt1(uint32_t *dest, const Xmm& s) { XmmStoreM1(dest, s); }
inline void XmmStoreFloat1(float *dest, const Xmm& s) { XmmStoreM1(dest, s); }
inline void XmmStoreInt2(uint32_t *dest, const Xmm& s) { XmmStoreM2(dest, s); }
inline void XmmStoreFloat2(float *dest, const Xmm& s) { XmmStoreM2(dest, s); }
inline void XmmStoreInt3(uint32_t *dest, const Xmm& s) { XmmStoreM3(dest, s); }
inline void XmmStoreFloat3(float *dest, const Xmm& s) { XmmStoreM3(dest, s); }
inline void XmmStoreInt4(uint32_t *dest, const Xmm& s) { XmmStoreM4(dest, s); }
inline void XmmStoreFloat4(float *dest, const Xmm& s) { XmmStoreM4(dest, s); }
inline void XmmStoreInt4A(uint32_t *dest, const Xmm& s) { XmmStoreA4(dest, s); }
inline void XmmStoreFloat4A(float *dest, const Xmm& s) { XmmStoreA4(dest, s); }



//---------------------------------------------------------------------
// Namespace End
//---------------------------------------------------------------------
NAMESPACE_END(SIMD);


#endif



