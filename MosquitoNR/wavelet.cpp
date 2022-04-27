//------------------------------------------------------------------------------
// wavelet.cpp
//------------------------------------------------------------------------------

/*
  To divide an image into low and high frequency components, CDF 5/3 wavelet is used.
  Wavelet transform is applied to rows and columns independently.
  Outside the image, reflected data must be prepared. (like 0123.. -> 210123..)

  <Forward transform>
  1. From odd number pixels, subtract an average of both neighboring pixel.
     This data is called detail coefficients, and shows high frequency components.
     ex) [1] -= ([0] + [2]) / 2
  2. To even number pixels, add a quarter of a sum of both neighboring pixel (i.e. detail coefficients).
     This data is called approximation coefficients, and shows low frequency components.
     ex) [0] += ([1] + [1]) / 4
         [2] += ([1] + [3]) / 4

  <Inverse transform>
  Just do opposite operations to the forward transform.

  CDF 5/3 wavelet transform is reversible including edge points.
*/

#include "mosquito_nr.h"
#include <emmintrin.h>
#include <tmmintrin.h>

void MosquitoNR::WaveletVert1(int thread_id)
{
  const int y_start = (height + 7) / 8 * thread_id / threads * 8;
  const int y_end = (height + 7) / 8 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int hloop = (width + 7) / 8;

  for (int y = y_start; y < y_end; y += 8)
  {
    short* srcp = luma[0] + y * pitch + 8;
    short* dstp = bufy[0] + y / 2 * pitch + 8;

    uint8_t* esi = (uint8_t*)srcp;
    uint8_t* edi = (uint8_t*)dstp;
    const int eax = pitch * sizeof(short);

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    for (int horiz = 0; horiz < hloop; horiz++) {
      auto tmp_esi = esi;
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm2 = _mm_add_epi16(xmm2, xmm1);
      xmm2 = _mm_srai_epi16(xmm2, 1);
      xmm0 = _mm_sub_epi16(xmm0, xmm2);

      esi = esi + 3 * eax;

      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 3 * eax));
      xmm6 = xmm1;
      xmm7 = xmm3;
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm3 = _mm_srai_epi16(xmm3, 1);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm2 = _mm_add_epi16(xmm2, xmm4);
      xmm0 = _mm_srai_epi16(xmm0, 2);
      xmm2 = _mm_srai_epi16(xmm2, 2);
      xmm6 = _mm_add_epi16(xmm6, xmm0);
      xmm7 = _mm_add_epi16(xmm7, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + eax), xmm7);

      esi = esi + 4 * eax;

      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax));
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 3 * eax));
      xmm6 = xmm5;
      xmm7 = xmm1;
      xmm5 = _mm_add_epi16(xmm5, xmm1);
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm5 = _mm_srai_epi16(xmm5, 1);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm0 = _mm_sub_epi16(xmm0, xmm5);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_add_epi16(xmm4, xmm0);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm4 = _mm_srai_epi16(xmm4, 2);
      xmm0 = _mm_srai_epi16(xmm0, 2);
      xmm6 = _mm_add_epi16(xmm6, xmm4);
      xmm7 = _mm_add_epi16(xmm7, xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 2 * eax), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 3 * eax), xmm7);

      esi = tmp_esi;
      esi += 16;
      edi += 16;

    }

    // horizontal reflection
    short* p = dstp;
    for (int i = 0; i < 4; ++i, p += pitch)
      p[-2] = p[2], p[-1] = p[1], p[width] = p[width - 2], p[width + 1] = p[width - 3];
  }
}

void MosquitoNR::WaveletHorz1(int thread_id)
{
  const int y_start = (height + 15) / 16 * thread_id / threads * 8;
  const int y_end = (height + 15) / 16 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int hloop1 = (width + 4 + 2 + 3) / 4;
  const int hloop2 = (width + 3) / 4;
  short* work = this->work[thread_id];

  for (int y = y_start; y < y_end; y += 8)
  {
    short* srcp = bufy[0] + y * pitch + 4;
    short* dstp = luma[0] + y / 2 * pitch + 8;

    // shuffle
    uint8_t* esi = (uint8_t*)srcp;
    uint8_t* edi = (uint8_t*)work;
    const int eax = pitch * sizeof(short);
    uint8_t* edx = esi + 4 * eax; // edx = srcp + 4 * pitch

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    for (int horiz = 0; horiz < hloop1; horiz++) {

      xmm0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi)); // ] // 03, 02, 01, 00
      xmm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + eax)); // 13, 12, 11, 10
      xmm2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + 2 * eax)); // 23, 22, 21, 20
      xmm3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + 3 * eax)); // 33, 32, 31, 30
      xmm4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx)); // 43, 42, 41, 40
      xmm5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + eax)); // 53, 52, 51, 50
      xmm6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + 2 * eax)); // 63, 62, 61, 60
      xmm7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + 3 * eax)); // 73, 72, 71, 70
      xmm0 = _mm_unpacklo_epi16(xmm0, xmm1); // 13, 03, 12, 02, 11, 01, 10, 00
      xmm2 = _mm_unpacklo_epi16(xmm2, xmm3); // 33, 23, 32, 22, 31, 21, 30, 20
      xmm4 = _mm_unpacklo_epi16(xmm4, xmm5); // 53, 43, 52, 42, 51, 41, 50, 40
      xmm6 = _mm_unpacklo_epi16(xmm6, xmm7); // 73, 63, 72, 62, 71, 61, 70, 60
      xmm1 = xmm0;
      xmm5 = xmm4;
      xmm0 = _mm_unpacklo_epi32(xmm0, xmm2); // 31, 21, 11, 01, 30, 20, 10, 00
      xmm1 = _mm_unpackhi_epi32(xmm1, xmm2); // 33, 23, 13, 03, 32, 22, 12, 02
      xmm4 = _mm_unpacklo_epi32(xmm4, xmm6); // 71, 61, 51, 41, 70, 60, 50, 40
      xmm5 = _mm_unpackhi_epi32(xmm5, xmm6); // 73, 63, 53, 43, 72, 62, 52, 42
      xmm2 = xmm0;
      xmm3 = xmm1;
      xmm0 = _mm_unpacklo_epi64(xmm0, xmm4); // 70, 60, 50, 40, 30, 20, 10, 00
      xmm2 = _mm_unpackhi_epi64(xmm2, xmm4); // 71, 61, 51, 41, 31, 21, 11, 01
      xmm1 = _mm_unpacklo_epi64(xmm1, xmm5); // 72, 62, 52, 42, 32, 22, 12, 02
      xmm3 = _mm_unpackhi_epi64(xmm3, xmm5); // 73, 63, 53, 43, 33, 23, 13, 03
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm1);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm3);
      esi += 8;
      edx += 8;
      edi += 64;
    }

    // wavelet transform
    esi = (uint8_t*)work;
    edi = (uint8_t*)dstp;
    esi += 64; // esi = work + 32 (short*)

    xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 32));
    xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 16));
    xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
    xmm2 = _mm_add_epi16(xmm2, xmm1);
    xmm2 = _mm_srai_epi16(xmm2, 1);
    xmm0 = _mm_sub_epi16(xmm0, xmm2);

    // wavelet_next4columns:
    for (int horiz = 0; horiz < hloop2; horiz++) {
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 16));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 32));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 48));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 64));
      xmm6 = xmm1;
      xmm7 = xmm3;
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm3 = _mm_srai_epi16(xmm3, 1);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm2 = _mm_add_epi16(xmm2, xmm4);
      xmm0 = _mm_srai_epi16(xmm0, 2);
      xmm2 = _mm_srai_epi16(xmm2, 2);
      xmm6 = _mm_add_epi16(xmm6, xmm0);
      xmm7 = _mm_add_epi16(xmm7, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm7);
      xmm0 = xmm4;
      xmm1 = xmm5;
      esi += 64;
      edi += 32;
      edx += 32;
    }

    // horizontal reflection
    if (width % 2 == 0) {
      edi = (uint8_t*)dstp;
      const int eax = width << 3; // eax = width / 2 * 16
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(edi + eax - 16));
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + eax), xmm0);
    }
  }
}

void MosquitoNR::WaveletVert2(int thread_id)
{
  const int y_start = (height + 7) / 8 * thread_id / threads * 8;
  const int y_end = (height + 7) / 8 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int hloop = (width + 7) / 8;

  for (int y = y_start; y < y_end; y += 8)
  {
    short* srcp = luma[1] + y * pitch + 8;
    short* dstp1 = bufy[0] + y / 2 * pitch + 8;
    short* dstp2 = bufy[1] + (y / 2 + 1) * pitch + 8;

    uint8_t* esi = (uint8_t*)srcp;
    uint8_t* edi = (uint8_t*)dstp1;
    uint8_t* edx = (uint8_t*)dstp2;
    const int eax = pitch * sizeof(short);


    //next8columns:
    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    for (int horiz = 0; horiz < hloop; horiz++) {
      auto tmp_esi = esi;
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm2 = _mm_add_epi16(xmm2, xmm1);
      xmm2 = _mm_srai_epi16(xmm2, 1);
      xmm0 = _mm_sub_epi16(xmm0, xmm2);
      esi += 3 * eax;

      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 3 * eax));
      xmm6 = xmm1;
      xmm7 = xmm3;
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm3 = _mm_srai_epi16(xmm3, 1);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx + eax), xmm4);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm2 = _mm_add_epi16(xmm2, xmm4);
      xmm0 = _mm_srai_epi16(xmm0, 2);
      xmm2 = _mm_srai_epi16(xmm2, 2);
      xmm6 = _mm_add_epi16(xmm6, xmm0);
      xmm7 = _mm_add_epi16(xmm7, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + eax), xmm7);

      esi += 4 * eax; // lea esi, [esi + 4 * eax]

      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax));
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 3 * eax));
      xmm6 = xmm5;
      xmm7 = xmm1;
      xmm5 = _mm_add_epi16(xmm5, xmm1);
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm5 = _mm_srai_epi16(xmm5, 1);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm0 = _mm_sub_epi16(xmm0, xmm5);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx + 2 * eax), xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx + 3 * eax), xmm2);
      xmm4 = _mm_add_epi16(xmm4, xmm0);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm4 = _mm_srai_epi16(xmm4, 2);
      xmm0 = _mm_srai_epi16(xmm0, 2);
      xmm6 = _mm_add_epi16(xmm6, xmm4);
      xmm7 = _mm_add_epi16(xmm7, xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 2 * eax), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 3 * eax), xmm7);

      esi = tmp_esi;
      esi += 16;
      edi += 16;
      edx += 16;
    }

    // horizontal reflection
    short* p = dstp1;
    for (int i = 0; i < 4; ++i, p += pitch)
      p[-2] = p[2], p[-1] = p[1], p[width] = p[width - 2], p[width + 1] = p[width - 3];
  }

  // vertical reflection
  if (y_start == 0)
    memcpy(bufy[1], bufy[1] + pitch, pitch * sizeof(short));
  if (thread_id == threads - 1 && height % 2 == 0)
    memcpy(bufy[1] + (height / 2 + 1) * pitch, bufy[1] + (height / 2 - 1) * pitch, pitch * sizeof(short));
}

void MosquitoNR::WaveletHorz2(int thread_id)
{
  const int y_start = (height + 15) / 16 * thread_id / threads * 8;
  const int y_end = (height + 15) / 16 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int hloop1 = (width + 4 + 2 + 3) / 4;
  const int hloop2 = (width + 7) / 8;
  short* work = this->work[thread_id];

  for (int y = y_start; y < y_end; y += 8)
  {
    short* srcp = bufy[0] + y * pitch + 4;
    short* dstp = bufx[1] + y / 2 * pitch + 8;

    // shuffle
    uint8_t* esi = (uint8_t*)srcp;
    uint8_t* edi = (uint8_t*)work;
    const int eax = pitch * sizeof(short);
    uint8_t* edx = esi + eax * 4; // edx = srcp + 4 * pitch

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    // shuffle_next4columns:
    for (int horiz = 0; horiz < hloop1; horiz++) {
      xmm0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi)); // 03, 02, 01, 00
      xmm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + eax)); // 13, 12, 11, 10
      xmm2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + 2 * eax)); // 23, 22, 21, 20
      xmm3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + 3 * eax)); // 33, 32, 31, 30
      xmm4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx)); // 43, 42, 41, 40
      xmm5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + eax)); // 53, 52, 51, 50
      xmm6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + 2 * eax)); // 63, 62, 61, 60
      xmm7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + 3 * eax)); // 73, 72, 71, 70
      xmm0 = _mm_unpacklo_epi16(xmm0, xmm1); // 13, 03, 12, 02, 11, 01, 10, 00
      xmm2 = _mm_unpacklo_epi16(xmm2, xmm3); // 33, 23, 32, 22, 31, 21, 30, 20
      xmm4 = _mm_unpacklo_epi16(xmm4, xmm5); // 53, 43, 52, 42, 51, 41, 50, 40
      xmm6 = _mm_unpacklo_epi16(xmm6, xmm7); // 73, 63, 72, 62, 71, 61, 70, 60
      xmm1 = xmm0;
      xmm5 = xmm4;
      xmm0 = _mm_unpacklo_epi32(xmm0, xmm2); // 31, 21, 11, 01, 30, 20, 10, 00
      xmm1 = _mm_unpackhi_epi32(xmm1, xmm2); // 33, 23, 13, 03, 32, 22, 12, 02
      xmm4 = _mm_unpacklo_epi32(xmm4, xmm6); // 71, 61, 51, 41, 70, 60, 50, 40
      xmm5 = _mm_unpackhi_epi32(xmm5, xmm6); // 73, 63, 53, 43, 72, 62, 52, 42
      xmm2 = xmm0;
      xmm3 = xmm1;
      xmm0 = _mm_unpacklo_epi64(xmm0, xmm4); // 70, 60, 50, 40, 30, 20, 10, 00
      xmm2 = _mm_unpackhi_epi64(xmm2, xmm4); // 71, 61, 51, 41, 31, 21, 11, 01
      xmm1 = _mm_unpacklo_epi64(xmm1, xmm5); // 72, 62, 52, 42, 32, 22, 12, 02
      xmm3 = _mm_unpackhi_epi64(xmm3, xmm5); // 73, 63, 53, 43, 33, 23, 13, 03
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm1);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm3);
      esi += 8;
      edx += 8;
      edi += 64;
    }

      // wavelet transform
    esi = (uint8_t*)work;
    edi = (uint8_t*)dstp;
    esi += 64; // esi = work + 32 (short *)

    xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 32));
    xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 16));
    xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
    xmm2 = _mm_add_epi16(xmm2, xmm1);
    xmm2 = _mm_srai_epi16(xmm2, 1);
    xmm0 = _mm_sub_epi16(xmm0, xmm2);
    _mm_store_si128(reinterpret_cast<__m128i*>(edi - 16), xmm0);

    // wavelet_next8columns:
    for (int horiz = 0; horiz < hloop2; horiz++) {
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 16));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 32));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 48));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 64));
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm3 = _mm_srai_epi16(xmm3, 1);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm4);
      xmm1 = xmm5;
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 80));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 96));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 112));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 128));
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm3 = _mm_srai_epi16(xmm3, 1);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm4);
      xmm1 = xmm5;
      esi += 128;
      edi += 64;
    }

    // horizontal reflection
    if (width % 2 == 0) {
      edi = (uint8_t*)dstp;
      const int eax = width << 3; // eax = width / 2 * 16
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(edi + eax - 32));
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + eax), xmm0);
    }
  }
}

void MosquitoNR::WaveletHorz3(int thread_id)
{
  const int y_start = (height + 15) / 16 * thread_id / threads * 8;
  const int y_end = (height + 15) / 16 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int hloop1 = (width + 4 + 2 + 3) / 4;
  const int hloop2 = (width + 3) / 4;
  short* work = this->work[thread_id];

  for (int y = y_start; y < y_end; y += 8)
  {
    short* srcp = bufy[0] + y * pitch + 4;
    short* dstp1 = bufx[0] + y / 2 * pitch + 8;
    short* dstp2 = bufx[1] + y / 2 * pitch + 8;

    // shuffle
    uint8_t* esi = (uint8_t*)srcp;
    uint8_t* edi = (uint8_t*)work;
    const int eax = pitch * sizeof(short);
    uint8_t* edx = esi + 4 * eax; // edx = srcp + 4 * pitch

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    // shuffle_next4columns:
    for (int horiz = 0; horiz < hloop1; horiz++) {
      xmm0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi)); // 03, 02, 01, 00
      xmm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + eax)); // 13, 12, 11, 10
      xmm2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + 2 * eax)); // 23, 22, 21, 20
      xmm3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(esi + 3 * eax)); // 33, 32, 31, 30
      xmm4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx)); // 43, 42, 41, 40
      xmm5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + eax)); // 53, 52, 51, 50
      xmm6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + 2 * eax)); // 63, 62, 61, 60
      xmm7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(edx + 3 * eax)); // 73, 72, 71, 70
      xmm0 = _mm_unpacklo_epi16(xmm0, xmm1); // 13, 03, 12, 02, 11, 01, 10, 00
      xmm2 = _mm_unpacklo_epi16(xmm2, xmm3); // 33, 23, 32, 22, 31, 21, 30, 20
      xmm4 = _mm_unpacklo_epi16(xmm4, xmm5); // 53, 43, 52, 42, 51, 41, 50, 40
      xmm6 = _mm_unpacklo_epi16(xmm6, xmm7); // 73, 63, 72, 62, 71, 61, 70, 60
      xmm1 = xmm0;
      xmm5 = xmm4;
      xmm0 = _mm_unpacklo_epi32(xmm0, xmm2); // 31, 21, 11, 01, 30, 20, 10, 00
      xmm1 = _mm_unpackhi_epi32(xmm1, xmm2); // 33, 23, 13, 03, 32, 22, 12, 02
      xmm4 = _mm_unpacklo_epi32(xmm4, xmm6); // 71, 61, 51, 41, 70, 60, 50, 40
      xmm5 = _mm_unpackhi_epi32(xmm5, xmm6); // 73, 63, 53, 43, 72, 62, 52, 42
      xmm2 = xmm0;
      xmm3 = xmm1;
      xmm0 = _mm_unpacklo_epi64(xmm0, xmm4); // 70, 60, 50, 40, 30, 20, 10, 00
      xmm2 = _mm_unpackhi_epi64(xmm2, xmm4); // 71, 61, 51, 41, 31, 21, 11, 01
      xmm1 = _mm_unpacklo_epi64(xmm1, xmm5); // 72, 62, 52, 42, 32, 22, 12, 02
      xmm3 = _mm_unpackhi_epi64(xmm3, xmm5); // 73, 63, 53, 43, 33, 23, 13, 03
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm1);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm3);
      esi += 8;
      edx += 8;
      edi += 64;
    }

    // wavelet transform
    esi = (uint8_t*)work;
    edi = (uint8_t*)dstp1;
    edx = (uint8_t*)dstp2;
    esi += 64; // esi = work + 32 (short *)

    xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 32));
    xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 16));
    xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
    xmm2 = _mm_add_epi16(xmm2, xmm1);
    xmm2 = _mm_srai_epi16(xmm2, 1);
    xmm0 = _mm_sub_epi16(xmm0, xmm2);
    _mm_store_si128(reinterpret_cast<__m128i*>(edx - 16), xmm0);

    // wavelet_next4columns:
    for (int horiz = 0; horiz < hloop2; horiz++) {
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 16));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 32));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 48));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 64));
      xmm6 = xmm1;
      xmm7 = xmm3;
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 1);
      xmm3 = _mm_srai_epi16(xmm3, 1);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx + 16), xmm4);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm2 = _mm_add_epi16(xmm2, xmm4);
      xmm0 = _mm_srai_epi16(xmm0, 2);
      xmm2 = _mm_srai_epi16(xmm2, 2);
      xmm6 = _mm_add_epi16(xmm6, xmm0);
      xmm7 = _mm_add_epi16(xmm7, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm7);
      xmm0 = xmm4;
      xmm1 = xmm5;
      esi += 64;
      edi += 32;
      edx += 32;
    }

    // horizontal reflection
    if (width % 2 == 0) {
      edi = (uint8_t*)dstp1;
      edx = (uint8_t*)dstp2;
      const int eax = width << 3; //eax = width / 2 * 16
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(edi + eax - 16));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + eax - 32));
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + eax), xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edx + eax), xmm1);
    }
  }
}

void MosquitoNR::BlendCoef(int thread_id)
{
  const int y_start = ((height + 15) & ~15) / 4 * thread_id / threads;
  const int y_end = ((height + 15) & ~15) / 4 * (thread_id + 1) / threads;
  if (y_start == y_end) return;
  const int pitch = this->pitch;
  const int multiplier = ((128 - restore) << 16) + restore;
  short* dstp = luma[0];
  short* srcp = bufx[0];

  const int eax = pitch * sizeof(short);

  __m128i xmm0, xmm1, xmm2, xmm6, xmm7;

  uint8_t* edi = (uint8_t*)(dstp + y_start * pitch);
  uint8_t* esi = (uint8_t*)(srcp + y_start * pitch);
  int ecx = (y_end - y_start) * pitch / 8;

  xmm6 = _mm_set1_epi32(multiplier); // xmm6 = [128 - restore, restore] * 4

  xmm7 = _mm_set1_epi32(64); // xmm7 = [64] * 4

//  next8pixels:
  for (int horiz = 0; horiz < ecx; horiz++) {
    xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(edi)); // d7, d6, d5, d4, d3, d2, d1, d0
    xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi)); // s7, s6, s5, s4, s3, s2, s1, s0
    xmm1 = xmm0;
    xmm0 = _mm_unpacklo_epi16(xmm0, xmm2); // s3, d3, s2, d2, s1, d1, s0, d0
    xmm1 = _mm_unpackhi_epi16(xmm1, xmm2); // s7, d7, s6, d6, s5, d5, s4, d4
    xmm0 = _mm_madd_epi16(xmm0, xmm6);
    xmm1 = _mm_madd_epi16(xmm1, xmm6);
    xmm0 = _mm_add_epi32(xmm0, xmm7);
    xmm1 = _mm_add_epi32(xmm1, xmm7);
    xmm0 = _mm_srai_epi32(xmm0, 7);
    xmm1 = _mm_srai_epi32(xmm1, 7);
    xmm0 = _mm_packs_epi32(xmm0, xmm1);
    _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);
    edi += 16;
    esi += 16;
  }
}

void MosquitoNR::InvWaveletHorz(int thread_id)
{
  const int y_start = (height + 15) / 16 * thread_id / threads * 8;
  const int y_end = (height + 15) / 16 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int hloop = (width + 3) / 4;
  short* work = this->work[thread_id];

  for (int y = y_start; y < y_end; y += 8)
  {
    short* srcp1 = luma[0] + y / 2 * pitch + 8;
    short* srcp2 = bufx[1] + y / 2 * pitch + 8;
    short* dstp = bufy[0] + y * pitch + 8;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    // wavelet transform
    uint8_t* esi = (uint8_t*)srcp1;
    uint8_t* edx = (uint8_t*)srcp2;
    uint8_t* edi = (uint8_t*)work;

    xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx - 16));
    xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
    xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx));
    xmm2 = _mm_add_epi16(xmm2, xmm1);
    xmm2 = _mm_srai_epi16(xmm2, 2);
    xmm0 = _mm_sub_epi16(xmm0, xmm2);
    _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);

    //wavelet_next4columns:
    for (int horiz = 0; horiz < hloop; horiz++) {
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 16));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + 16));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 32));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + 32));
      xmm6 = xmm1;
      xmm7 = xmm3;
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 2);
      xmm3 = _mm_srai_epi16(xmm3, 2);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 64), xmm4);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm2 = _mm_add_epi16(xmm2, xmm4);
      xmm0 = _mm_srai_epi16(xmm0, 1);
      xmm2 = _mm_srai_epi16(xmm2, 1);
      xmm6 = _mm_add_epi16(xmm6, xmm0);
      xmm7 = _mm_add_epi16(xmm7, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm7);
      xmm0 = xmm4;
      xmm1 = xmm5;
      esi += 32;
      edx += 32;
      edi += 64;
    }

    // shuffle
    esi = (uint8_t*)work;
    edi = (uint8_t*)dstp;
    const int eax = pitch * sizeof(short); // eax = pitch * sizeof(short)
    edx = edi + 4 * eax; // edx = dstp + 4 * pitch

    // shuffle_next4columns:
    for (int horiz = 0; horiz < hloop; horiz++) {
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi)); // 70, 60, 50, 40, 30, 20, 10, 00
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 16)); // 71, 61, 51, 41, 31, 21, 11, 01
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 32)); // 72, 62, 52, 42, 32, 22, 12, 02
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 48)); // 73, 63, 53, 43, 33, 23, 13, 03
      xmm4 = xmm0;
      xmm6 = xmm2;
      xmm0 = _mm_unpacklo_epi16(xmm0, xmm1); // 31, 30, 21, 20, 11, 10, 01, 00
      xmm4 = _mm_unpackhi_epi16(xmm4, xmm1); // 71, 70, 61, 60, 51, 50, 41, 40
      xmm2 = _mm_unpacklo_epi16(xmm2, xmm3); // 33, 32, 23, 22, 13, 12, 03, 02
      xmm6 = _mm_unpackhi_epi16(xmm6, xmm3); // 73, 72, 63, 62, 53, 52, 43, 42
      xmm1 = xmm0;
      xmm5 = xmm4;
      xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);  // 13, 12, 11, 10, 03, 02, 01, 00
      xmm1 = _mm_unpackhi_epi32(xmm1, xmm2); // 33, 32, 31, 30, 23, 22, 21, 20
      xmm4 = _mm_unpacklo_epi32(xmm4, xmm6); // 53, 52, 51, 50, 43, 42, 41, 40
      xmm5 = _mm_unpackhi_epi32(xmm5, xmm6); // 73, 72, 71, 70, 63, 62, 61, 60
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edi), xmm0);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edi + 2 * eax), xmm1);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edx), xmm4);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edx + 2 * eax), xmm5);
      xmm0 = _mm_unpackhi_epi64(xmm0, xmm0);
      xmm1 = _mm_unpackhi_epi64(xmm1, xmm1);
      xmm4 = _mm_unpackhi_epi64(xmm4, xmm4);
      xmm5 = _mm_unpackhi_epi64(xmm5, xmm5);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edi + eax), xmm0);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edi + 3 * eax), xmm1);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edx + eax), xmm4);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(edx + 3 * eax), xmm5);
      esi += 64;
      edi += 8;
      edx += 8;
    }
  }

  // vertical reflection
  if (thread_id == threads - 1 && height % 2 == 0)
    memcpy(bufy[0] + height / 2 * pitch, bufy[0] + (height / 2 - 1) * pitch, pitch * sizeof(short));
}

void MosquitoNR::InvWaveletVert(int thread_id)
{
  const int y_start = (height + 7) / 8 * thread_id / threads * 8;
  const int y_end = (height + 7) / 8 * (thread_id + 1) / threads * 8;
  if (y_start == y_end) return;
  const int pitch = this->pitch;

  __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

  for (int y = y_start; y < y_end; y += 8)
  {
    int hloop = (width + 7) / 8;
    short* srcp1 = bufy[0] + y / 2 * pitch + 8;
    short* srcp2 = bufy[1] + y / 2 * pitch + 8;
    short* dstp = luma[1] + (y + 2) * pitch + 8;

    uint8_t* esi = (uint8_t*)srcp1;
    uint8_t* edx = (uint8_t*)srcp2;
    uint8_t* edi = (uint8_t*)dstp;

    const int eax = pitch * sizeof(short);
    const int ecx = eax * 5; // ecx = pitch * sizeof(short) * 5

    // next8columns:
    for (int horiz = 0; horiz < hloop; horiz++) {
      auto tmp_edi = edi; //  push edi
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx));
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + eax));
      xmm2 = _mm_add_epi16(xmm2, xmm1);
      xmm2 = _mm_srai_epi16(xmm2, 2);
      xmm0 = _mm_sub_epi16(xmm0, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);

      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 1 * eax));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + 2 * eax));
      xmm4 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax));
      xmm5 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + 3 * eax));
      xmm6 = xmm1;
      xmm7 = xmm3;
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm3 = _mm_add_epi16(xmm3, xmm5);
      xmm1 = _mm_srai_epi16(xmm1, 2);
      xmm3 = _mm_srai_epi16(xmm3, 2);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      xmm4 = _mm_sub_epi16(xmm4, xmm3);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 2 * eax), xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 4 * eax), xmm4);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm2 = _mm_add_epi16(xmm2, xmm4);
      xmm0 = _mm_srai_epi16(xmm0, 1);
      xmm2 = _mm_srai_epi16(xmm2, 1);
      xmm6 = _mm_add_epi16(xmm6, xmm0);
      xmm7 = _mm_add_epi16(xmm7, xmm2);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 1 * eax), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 3 * eax), xmm7);

      edi += 4 * eax;

      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 3 * eax));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + 4 * eax));
      xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 4 * eax));
      xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(edx + ecx));
      xmm6 = xmm5;
      xmm7 = xmm1;
      xmm5 = _mm_add_epi16(xmm5, xmm1);
      xmm1 = _mm_add_epi16(xmm1, xmm3);
      xmm5 = _mm_srai_epi16(xmm5, 2);
      xmm1 = _mm_srai_epi16(xmm1, 2);
      xmm0 = _mm_sub_epi16(xmm0, xmm5);
      xmm2 = _mm_sub_epi16(xmm2, xmm1);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 2 * eax), xmm0);
      xmm4 = _mm_add_epi16(xmm4, xmm0);
      xmm0 = _mm_add_epi16(xmm0, xmm2);
      xmm4 = _mm_srai_epi16(xmm4, 1);
      xmm0 = _mm_srai_epi16(xmm0, 1);
      xmm6 = _mm_add_epi16(xmm6, xmm4);
      xmm7 = _mm_add_epi16(xmm7, xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 1 * eax), xmm6);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + 3 * eax), xmm7);

      edi = tmp_edi;
      esi += 16;
      edx += 16;
      edi += 16;
    }
  }
}
