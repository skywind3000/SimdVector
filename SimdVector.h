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
// SIMD register of 4-dimension vector
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
// Convinent for Static Initialization
//---------------------------------------------------------------------
SIMD_ALIGNED_STRUCT(16) XmmU32 { union { uint32_t u[4]; Xmm x; }; };
SIMD_ALIGNED_STRUCT(16) XmmI32 { union { int32_t i[4]; Xmm x; }; };
SIMD_ALIGNED_STRUCT(16) XmmF32 { union { float f[4]; Xmm x; }; };


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
	__m128 z = _mm_shuffle_ps(s.r, s.r, _MM_SHUFFLE(2, 2, 2, 2));
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
	// printf("debug: %f %f %f %f\n", s.f[0], s.f[1], s.f[2], s.f[3]);
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
// Const Table
//---------------------------------------------------------------------
#ifndef CONST_WEEK
#define CONST_WEEK extern const __declspec(selectany)
#endif

namespace Const {
	CONST_WEEK XmmF32 FixUnsigned           = { { { 32768.0f*65536.0f, 32768.0f*65536.0f, 32768.0f*65536.0f, 32768.0f*65536.0f } } };
	CONST_WEEK XmmF32 MaxInt                = { { { 65536.0f*32768.0f - 128.0f, 65536.0f*32768.0f - 128.0f, 65536.0f*32768.0f - 128.0f, 65536.0f*32768.0f - 128.0f } } };
	CONST_WEEK XmmF32 MaxUInt               = { { { 65536.0f*65536.0f - 256.0f, 65536.0f*65536.0f - 256.0f, 65536.0f*65536.0f - 256.0f, 65536.0f*65536.0f - 256.0f } } };
	CONST_WEEK XmmF32 UnsignedFix           = { { { 32768.0f*65536.0f, 32768.0f*65536.0f, 32768.0f*65536.0f, 32768.0f*65536.0f } } };
	CONST_WEEK XmmF32 One                   = { { { 1.0f, 1.0f, 1.0f, 1.0f } } };
	CONST_WEEK XmmF32 One3                  = { { { 1.0f, 1.0f, 1.0f, 0.0f } } };
	CONST_WEEK XmmF32 Zero                  = { { { 0.0f, 0.0f, 0.0f, 0.0f } } };

	CONST_WEEK XmmU32 NegativeZero          = { { { 0x80000000, 0x80000000, 0x80000000, 0x80000000 } } };
	CONST_WEEK XmmU32 Negate3               = { { { 0x80000000, 0x80000000, 0x80000000, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskXY                = { { { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 } } };
	CONST_WEEK XmmU32 Mask3                 = { { { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskX                 = { { { 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskY                 = { { { 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskZ                 = { { { 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskW                 = { { { 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF } } };

	CONST_WEEK XmmI32 AbsMask               = { { { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF } } };
};


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
	__m128 vOverflow = _mm_cmpgt_ps(x.r, Const::MaxInt.x.r);
	__m128i vResulti = _mm_cvttps_epi32(x.r);
	__m128 vResult = _mm_and_ps(vOverflow, Const::AbsMask.x.r);
	vOverflow = _mm_andnot_ps(vOverflow, _mm_castsi128_ps(vResulti));
	Xmm y;
	y.r = _mm_or_ps(vOverflow, vResult);
	return y;
#endif
}

inline Xmm XmmConvertUIntToFloat(const Xmm& x) {
#if SIMD_HAS_NONE
	Xmm y;
	y.f[0] = static_cast<float>(x.u[0]);
	y.f[1] = static_cast<float>(x.u[1]);
	y.f[2] = static_cast<float>(x.u[2]);
	y.f[3] = static_cast<float>(x.u[3]);
	return y;
#elif SIMD_HAS_SSE2
	__m128i v = _mm_castps_si128(x.r);
	__m128 vMask = _mm_and_ps(_mm_castsi128_ps(v), Const::NegativeZero.x.r);
	__m128 vResult = _mm_xor_ps(_mm_castsi128_ps(v), vMask);
	vResult = _mm_cvtepi32_ps(_mm_castps_si128(vResult));
	__m128i iMask = _mm_srai_epi32(_mm_castps_si128(vMask), 31);
	vMask = _mm_and_ps(_mm_castsi128_ps(iMask), Const::FixUnsigned.x.r);
	vResult = _mm_and_ps(vResult, vMask);
	// vResult = Const::NegativeZero.x.r;
	Xmm y;
	y.r = vResult;
	return y;
#endif
}

inline Xmm XmmConvertFloatToUInt(const Xmm& x) {
#if SIMD_HAS_NONE
	Xmm y;
	y.u[0] = static_cast<uint32_t>(x.f[0]);
	y.u[1] = static_cast<uint32_t>(x.f[1]);
	y.u[2] = static_cast<uint32_t>(x.f[2]);
	y.u[3] = static_cast<uint32_t>(x.f[3]);
	return y;
#elif SIMD_HAS_SSE2
	__m128 vResult = _mm_max_ps(x.r, Const::Zero.x.r);
	__m128 vOverflow = _mm_cmpgt_ps(vResult, Const::MaxUInt.x.r);
	__m128 vValue = Const::UnsignedFix.x.r;
	__m128 vMask = _mm_cmpge_ps(vResult, vValue);
	vValue = _mm_and_ps(vValue, vMask);
	vResult = _mm_sub_ps(vResult, vValue);
	__m128i vResulti = _mm_cvttps_epi32(vResult);
	vMask = _mm_and_ps(vMask, Const::NegativeZero.x.r);
	vResult = _mm_xor_ps(_mm_castsi128_ps(vResulti), vMask);
	Xmm y;
	y.r = _mm_or_ps(vResult, vOverflow);
	return y;
#endif
}


//---------------------------------------------------------------------
// Namespace End
//---------------------------------------------------------------------
NAMESPACE_END(SIMD);


#endif



