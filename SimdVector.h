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
#include <string.h>
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
#include <immintrin.h>
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
// inline 
//---------------------------------------------------------------------
#if SIMD_HAS_AVX
#define XMM_PERMUTE_PS( v, c ) _mm_permute_ps( v, c )
#elif SIMD_HAS_SSE2
#define XMM_PERMUTE_PS( v, c ) _mm_shuffle_ps( v, v, c )
#endif


//---------------------------------------------------------------------
// Load/Store
//---------------------------------------------------------------------

// load 1 element from memory
inline Xmm XmmLoadM1(const void *ptr) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	Xmm s;
	const uint8_t *source = reinterpret_cast<const uint8_t*>(ptr);
	memcpy(s.u, source, sizeof(uint32_t) * 1);
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
	const uint8_t *source = reinterpret_cast<const uint8_t*>(ptr);
	memcpy(s.u, source, sizeof(uint32_t) * 2);
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
	const uint8_t *source = reinterpret_cast<const uint8_t*>(ptr);
	memcpy(s.u, source, sizeof(uint32_t) * 3);
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
	const uint8_t *source = reinterpret_cast<const uint8_t*>(ptr);
	memcpy(s.u, source, sizeof(uint32_t) * 4);
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
	memcpy(ptr, s.u, sizeof(uint32_t));
#elif SIMD_HAS_SSE2
	_mm_store_ss(reinterpret_cast<float*>(ptr), s.r);
#endif
}

// store 2 elements to memory
inline void XmmStoreM2(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	memcpy(ptr, s.u, sizeof(uint32_t) * 2);
#elif SIMD_HAS_SSE2
	_mm_store_sd(reinterpret_cast<double*>(ptr), _mm_castps_pd(s.r));
#endif
}

// store 3 elements to memory
inline void XmmStoreM3(void *ptr, const Xmm &s) noexcept {
	assert(ptr);
#if SIMD_HAS_NONE
	memcpy(ptr, s.u, sizeof(uint32_t) * 3);
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
	memcpy(ptr, s.u, sizeof(uint32_t) * 4);
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
	const uint8_t *source = reinterpret_cast<const uint8_t*>(ptr);
	memcpy(s.u, source, sizeof(uint32_t) * 2);
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
	const uint8_t *source = reinterpret_cast<const uint8_t*>(ptr);
	memcpy(s.u, source, sizeof(uint32_t) * 4);
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
	memcpy(dest, s.u, sizeof(uint32_t) * 2);
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
	memcpy(dest, s.u, sizeof(uint32_t) * 4);
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
	CONST_WEEK XmmF32 Two                   = { { { 2.f, 2.f, 2.f, 2.f } } };
	CONST_WEEK XmmF32 Four                  = { { { 4.f, 4.f, 4.f, 4.f } } };
	CONST_WEEK XmmF32 Six                   = { { { 6.f, 6.f, 6.f, 6.f } } };
	CONST_WEEK XmmF32 NegativeOne           = { { { -1.0f, -1.0f, -1.0f, -1.0f } } };
	CONST_WEEK XmmF32 OneHalf               = { { { 0.5f, 0.5f, 0.5f, 0.5f } } };
	CONST_WEEK XmmF32 NegativeOneHalf       = { { { -0.5f, -0.5f, -0.5f, -0.5f } } };
	CONST_WEEK XmmF32 Epsilon               = { { { 1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f } } };

	CONST_WEEK XmmU32 NegativeZero          = { { { 0x80000000, 0x80000000, 0x80000000, 0x80000000 } } };
	CONST_WEEK XmmU32 Negate3               = { { { 0x80000000, 0x80000000, 0x80000000, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskXY                = { { { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 } } };
	CONST_WEEK XmmU32 Mask3                 = { { { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskX                 = { { { 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskY                 = { { { 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskZ                 = { { { 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 } } };
	CONST_WEEK XmmU32 MaskW                 = { { { 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF } } };

	CONST_WEEK XmmI32 AbsMask               = { { { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF } } };
	CONST_WEEK XmmI32 Infinity              = { { { 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000 } } };
	CONST_WEEK XmmI32 QNaN                  = { { { 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000 } } };
}


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
	__m128 vMask = _mm_and_ps(x.r, Const::NegativeZero.x.r);
	__m128 vResult = _mm_xor_ps(x.r, vMask);
	vResult = _mm_cvtepi32_ps(_mm_castps_si128(vResult));
	__m128i iMask = _mm_srai_epi32(_mm_castps_si128(vMask), 31);
	vMask = _mm_and_ps(_mm_castsi128_ps(iMask), Const::FixUnsigned.x.r);
	vResult = _mm_add_ps(vResult, vMask);
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
// General Vector
//---------------------------------------------------------------------
static inline Xmm XmmVectorZero() {
#if SIMD_HAS_NONE
	Xmm x = { { { 0.0f, 0.0f, 0.0f, 0.0f } } };
	return x;
#elif SIMD_HAS_SSE2
	Xmm x;
	x.r = _mm_setzero_ps();
	return x;
#endif
}

static inline Xmm XmmVectorSet(float x, float y, float z, float w) {
#if SIMD_HAS_NONE
	Xmm m = { { { x, y, z, w } } };
	return m;
#elif SIMD_HAS_SSE2
	Xmm m;
	m.r = _mm_set_ps(x, y, z, w);
	return m;
#endif
}

static inline Xmm XmmVectorSetInt(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
#if SIMD_HAS_NONE
	XmmU32 m = {{{ x, y, z, w }}};
	return m.x;
#elif SIMD_HAS_SSE2
	__m128i v = _mm_set_epi32(w, z, y, x);
	Xmm m;
	m.r = _mm_castsi128_ps(v);
	return m;
#endif
}

static inline Xmm XmmVectorReplicate(float value) {
#if SIMD_HAS_NONE
	Xmm x;
	x.f[0] = x.f[1] = x.f[2] = x.f[3] = value;
	return x;
#elif SIMD_HAS_SSE2
	Xmm x;
	x.r = _mm_set_ps1(value);
	return x;
#endif
}

static inline Xmm XmmVectorReplicatePtr(const float *pv) {
#if SIMD_HAS_NONE
	Xmm x;
	float value = pv[0];
	x.f[0] = x.f[1] = x.f[2] = x.f[3] = value;
	return x;
#elif SIMD_HAS_SSE2
	Xmm x;
	x.r = _mm_load_ps1(pv);
	return x;
#endif
}

static inline Xmm XmmVectorReplicateInt(uint32_t value) {
#if SIMD_HAS_NONE
	Xmm x;
	x.u[0] = x.u[1] = x.u[2] = x.u[3] = value;
	return x;
#elif SIMD_HAS_SSE2
	Xmm x;
	__m128i v = _mm_set1_epi32(value);
	x.r = _mm_castsi128_ps(v);
	return x;
#endif
}

static inline Xmm XmmVectorReplicateIntPtr(const uint32_t *pv) {
#if SIMD_HAS_NONE
	Xmm x;
	uint32_t value = pv[0];
	x.u[0] = x.u[1] = x.u[2] = x.u[3] = value;
	return x;
#elif SIMD_HAS_SSE2
	Xmm x;
	x.r = _mm_load_ps1(reinterpret_cast<const float*>(pv));
	return x;
#endif
}

static inline Xmm XmmVectorTrueInt() {
#if SIMD_HAS_NONE
	XmmU32 uu = {{{ 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU }}};
	return uu.x;
#elif SIMD_HAS_SSE2
	__m128i v = _mm_set1_epi32(-1);
	Xmm x;
	x.r = _mm_castsi128_ps(v);
	return x;
#endif
}

static inline Xmm XmmVectorFalseInt() {
#if SIMD_HAS_NONE
	XmmF32 ff = {{{ 0.0f, 0.0f, 0.0f, 0.0f }}};
	return ff.x;
#elif SIMD_HAS_SSE2
	Xmm x;
	x.r = _mm_setzero_ps();
	return x;
#endif
}

static inline Xmm XmmVectorSplatX(const Xmm& m) {
#if SIMD_HAS_NONE
	Xmm n;
	n.f[0] = n.f[1] = n.f[2] = n.f[3] = m.f[0];
	return n;
#elif SIMD_HAS_SSE2
	Xmm n;
	n.r = XMM_PERMUTE_PS(m.r, _MM_SHUFFLE(0, 0, 0, 0));
	return n;
#endif
}

static inline Xmm XmmVectorSplatY(const Xmm& m) {
#if SIMD_HAS_NONE
	Xmm n;
	n.f[0] = n.f[1] = n.f[2] = n.f[3] = m.f[1];
	return n;
#elif SIMD_HAS_SSE2
	Xmm n;
	n.r = XMM_PERMUTE_PS(m.r, _MM_SHUFFLE(1, 1, 1, 1));
	return n;
#endif
}

static inline Xmm XmmVectorSplatZ(const Xmm& m) {
#if SIMD_HAS_NONE
	Xmm n;
	n.f[0] = n.f[1] = n.f[2] = n.f[3] = m.f[2];
	return n;
#elif SIMD_HAS_SSE2
	Xmm n;
	n.r = XMM_PERMUTE_PS(m.r, _MM_SHUFFLE(2, 2, 2, 2));
	return n;
#endif
}

static inline Xmm XmmVectorSplatW(const Xmm& m) {
#if SIMD_HAS_NONE
	Xmm n;
	n.f[0] = n.f[1] = n.f[2] = n.f[3] = m.f[3];
	return n;
#elif SIMD_HAS_SSE2
	Xmm n;
	n.r = XMM_PERMUTE_PS(m.r, _MM_SHUFFLE(3, 3, 3, 3));
	return n;
#endif
}

static inline Xmm XmmVectorSplateOne() {
#if SIMD_HAS_NONE
	Xmm n;
	n.f[0] = n.f[1] = n.f[2] = n.f[3] = 1.0f;
	return n;
#elif SIMD_HAS_SSE2
	return Const::One.x;
#endif
}

static inline Xmm XmmVectorSplateInfinity() {
#if SIMD_HAS_NONE
	Xmm n;
	n.u[0] = n.u[1] = n.u[2] = n.u[3] = 0x7f800000;
	return n;
#elif SIMD_HAS_SSE2
	return Const::Infinity.x;
#endif
}

static inline Xmm XmmVectorSplateQNaN() {
#if SIMD_HAS_NONE
	Xmm n;
	n.u[0] = n.u[1] = n.u[2] = n.u[3] = 0x7fc00000;
	return n;
#elif SIMD_HAS_SSE2
	return Const::QNaN.x;
#endif
}

static inline Xmm XmmVectorSplateEpsilon() {
#if SIMD_HAS_NONE
	Xmm n;
	n.u[0] = n.u[1] = n.u[2] = n.u[3] = 0x34000000;
	return n;
#elif SIMD_HAS_SSE2
	return Const::Epsilon.x;
#endif
}

static inline Xmm XmmVectorSplateSignMask() {
#if SIMD_HAS_NONE
	Xmm n;
	n.u[0] = n.u[1] = n.u[2] = n.u[3] = 0x80000000;
	return n;
#elif SIMD_HAS_SSE2
	__m128i v = _mm_set1_epi32(0x80000000);
	Xmm n;
	n.r = _mm_castsi128_ps(v);
	return n;
#endif
}

static inline float XmmVectorGetByIndex(const Xmm& x, size_t i) {
	assert(i < 4);
#if SIMD_HAS_NONE
	return x.f[i];
#elif SIMD_HAS_SSE2
	XmmF32 m;
	m.x = x;
	return m.f[i];
#endif
}

static inline float XmmVectorGetX(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.f[0];
#elif SIMD_HAS_SSE2
	return _mm_cvtss_f32(x.r);
#endif
}


static inline float XmmVectorGetY(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.f[1];
#elif SIMD_HAS_SSE2
	__m128 temp = XMM_PERMUTE_PS(x.r, _MM_SHUFFLE(1, 1, 1, 1));
	return _mm_cvtss_f32(temp);
#endif
}

static inline float XmmVectorGetZ(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.f[2];
#elif SIMD_HAS_SSE2
	__m128 temp = XMM_PERMUTE_PS(x.r, _MM_SHUFFLE(2, 2, 2, 2));
	return _mm_cvtss_f32(temp);
#endif
}

static inline float XmmVectorGetW(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.f[3];
#elif SIMD_HAS_SSE2
	__m128 temp = XMM_PERMUTE_PS(x.r, _MM_SHUFFLE(3, 3, 3, 3));
	return _mm_cvtss_f32(temp);
#endif
}

static inline uint32_t XmmVectorGetIntByIndex(const Xmm& x, size_t i) {
	assert(i < 4);
#if SIMD_HAS_NONE
	return x.u[i];
#elif SIMD_HAS_SSE2
	XmmU32 u;
	u.x = x;
	return u.u[i];
#endif
}

static inline uint32_t XmmVectorGetIntX(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.u[0];
#elif SIMD_HAS_SSE2
	return static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_castps_si128(x.r)));
#endif
}

static inline uint32_t XmmVectorGetIntY(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.u[1];
#elif SIMD_HAS_SSE2
	__m128i ri = _mm_shuffle_epi32(_mm_castps_si128(x.r), _MM_SHUFFLE(1, 1, 1, 1));
	return static_cast<uint32_t>(_mm_cvtsi128_si32(ri));
#endif
}

static inline uint32_t XmmVectorGetIntZ(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.u[2];
#elif SIMD_HAS_SSE2
	__m128i ri = _mm_shuffle_epi32(_mm_castps_si128(x.r), _MM_SHUFFLE(2, 2, 2, 2));
	return static_cast<uint32_t>(_mm_cvtsi128_si32(ri));
#endif
}

static inline uint32_t XmmVectorGetIntW(const Xmm& x) {
#if SIMD_HAS_NONE
	return x.u[3];
#elif SIMD_HAS_SSE2
	__m128i ri = _mm_shuffle_epi32(_mm_castps_si128(x.r), _MM_SHUFFLE(3, 3, 3, 3));
	return static_cast<uint32_t>(_mm_cvtsi128_si32(ri));
#endif
}

static inline Xmm XmmVectorSetByIndex(const Xmm& x, float f, size_t i) {
	assert(i < 4);
	XmmF32 m;
	m.x = x;
	m.f[i] = f;
	return m.x;
}

static inline Xmm XmmVectorSetX(const Xmm& v, float x) {
#if SIMD_HAS_NONE
	XmmF32 m = {{{ x, v.f[1], v.f[2], v.f[3] }}};
	return m.x;
#elif SIMD_HAS_SSE2
	Xmm r;
	r.r = _mm_set_ss(x);
	r.r = _mm_move_ss(v.r, r.r);
	return r;
#endif
}

static inline Xmm XmmVectorSetY(const Xmm& v, float y) {
#if SIMD_HAS_NONE
	XmmF32 m = {{{ v.f[0], y, v.f[2], v.f[3] }}};
	return m.x;
#elif SIMD_HAS_SSE2
	__m128 result = XMM_PERMUTE_PS(v.r, _MM_SHUFFLE(3, 2, 0, 1));
	__m128 temp = _mm_set_ss(y);
	result = _mm_move_ss(result, temp);
	Xmm m;
	m.r = XMM_PERMUTE_PS(result, _MM_SHUFFLE(3, 2, 0, 1));
	return m;
#endif
}

static inline Xmm XmmVectorSetZ(const Xmm& v, float z) {
#if SIMD_HAS_NONE
	XmmF32 m = {{{ v.f[0], v.f[1], z, v.f[3] }}};
	return m.x;
#elif SIMD_HAS_SSE2
	__m128 result = XMM_PERMUTE_PS(v.r, _MM_SHUFFLE(3, 0, 1, 2));
	__m128 temp = _mm_set_ss(z);
	result = _mm_move_ss(result, temp);
	Xmm m;
	m.r = XMM_PERMUTE_PS(result, _MM_SHUFFLE(3, 0, 1, 2));
	return m;
#endif
}

static inline Xmm XmmVectorSetW(const Xmm& v, float w) {
#if SIMD_HAS_NONE
	XmmF32 m = {{{ v.f[0], v.f[1], v.f[2], w }}};
	return m.x;
#elif SIMD_HAS_SSE2
	__m128 result = XMM_PERMUTE_PS(v.r, _MM_SHUFFLE(0, 2, 1, 3));
	__m128 temp = _mm_set_ss(w);
	result = _mm_move_ss(result, temp);
	Xmm m;
	m.r = XMM_PERMUTE_PS(result, _MM_SHUFFLE(0, 2, 1, 3));
	return m;
#endif
}

static inline Xmm XmmVectorSetIntByIndex(const Xmm& v, uint32_t x, size_t i) {
	assert(i < 4);
	XmmU32 tmp;
	tmp.x = v;
	tmp.u[i] = x;
	return tmp.x;
}

static inline Xmm XmmVectorSetIntX(const Xmm& v, uint32_t x) {
#if SIMD_HAS_NONE
	XmmU32 tmp = {{{ x, v.u[1], v.u[2], v.u[3] }}};
	return tmp.x;
#elif SIMD_HAS_SSE2
	__m128i temp = _mm_cvtsi32_si128(x);
	Xmm result;
	result.r = _mm_move_ss(v.r, _mm_castsi128_ps(temp));
	return result;
#endif
}

static inline Xmm XmmVectorSetIntY(const Xmm& v, uint32_t y) {
#if SIMD_HAS_NONE
	XmmU32 tmp = {{{ v.u[0], y, v.u[2], v.u[3] }}};
	return tmp.x;
#elif SIMD_HAS_SSE2
	__m128 result = XMM_PERMUTE_PS(v.r, _MM_SHUFFLE(3, 2, 0, 1));
	__m128i temp = _mm_cvtsi32_si128(y);
	result = _mm_move_ss(result, _mm_castsi128_ps(temp));
	Xmm m;
	m.r = XMM_PERMUTE_PS(result, _MM_SHUFFLE(3, 2, 0, 1));
	return m;
#endif
}

static inline Xmm XmmVectorSetIntZ(const Xmm& v, uint32_t z) {
#if SIMD_HAS_NONE
	XmmU32 tmp = {{{ v.u[0], v.u[1], z, v.u[3] }}};
	return tmp.x;
#elif SIMD_HAS_SSE2
	__m128 result = XMM_PERMUTE_PS(v.r, _MM_SHUFFLE(3, 0, 1, 2));
	__m128i temp = _mm_cvtsi32_si128(z);
	result = _mm_move_ss(result, _mm_castsi128_ps(temp));
	Xmm m;
	m.r = XMM_PERMUTE_PS(result, _MM_SHUFFLE(3, 0, 1, 2));
	return m;
#endif
}

static inline Xmm XmmVectorSetIntW(const Xmm& v, uint32_t w) {
#if SIMD_HAS_NONE
	XmmU32 tmp = {{{ v.u[0], v.u[1], v.u[2], w }}};
	return tmp.x;
#elif SIMD_HAS_SSE2
	__m128 result = XMM_PERMUTE_PS(v.r, _MM_SHUFFLE(0, 2, 1, 3));
	__m128i temp = _mm_cvtsi32_si128(w);
	result = _mm_move_ss(result, _mm_castsi128_ps(temp));
	Xmm m;
	m.r = XMM_PERMUTE_PS(result, _MM_SHUFFLE(0, 2, 1, 3));
	return m;
#endif
}


//---------------------------------------------------------------------
// Swizzle / Permute
//---------------------------------------------------------------------
static inline Xmm XmmVectorSwizzle(const Xmm& v, uint32_t E0, uint32_t E1, 
		uint32_t E2, uint32_t E3) {
	assert((E0 < 4) && (E1 < 4) && (E2 < 4) && (E3 < 4));
#if SIMD_HAS_NONE
	XmmF32 result = {{{ v.f[E0], v.f[E1], v.f[E2], v.f[E3] }}};
	return result.x;
#elif SIMD_HAS_SSE
	Xmm result;
	memcpy(&(result.f[0]), v.f[E0]);
	memcpy(&(result.f[1]), v.f[E1]);
	memcpy(&(result.f[2]), v.f[E2]);
	memcpy(&(result.f[3]), v.f[E3]);
	return result;
#endif
}


//---------------------------------------------------------------------
// Namespace End
//---------------------------------------------------------------------
NAMESPACE_END(SIMD);


#endif



