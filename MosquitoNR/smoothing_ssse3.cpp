//------------------------------------------------------------------------------
// smoothing_ssse3.cpp
//------------------------------------------------------------------------------

#include "mosquito_nr.h"
#include <emmintrin.h>
#include <tmmintrin.h>

// direction-aware blur
void MosquitoNR::SmoothingSSSE3(int thread_id)
{
  const int y_start = height * thread_id / threads;
  const int y_end = height * (thread_id + 1) / threads;
  if (y_start == y_end) return;
  const int width = this->width;
  const int pitch = this->pitch;
  const int pitch2 = pitch * sizeof(short);
  __declspec(align(16)) short sad[48];
  short* srcp;
  short* dstp;
  short* sadp = sad;

  const __m128i fours = _mm_set1_epi16(4);
  const __m128i threes = _mm_set1_epi16(3);

  __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

  if (radius == 1)
  {
    const int coef0 = 64 - strength * 2; // own pixel's coefficient (when divisor = 64)
    const int coef1 = 128 - strength * 4; // own pixel's coefficient (when divisor = 128)
    const int coef2 = strength; // other pixel's coefficient

    for (int y = y_start; y < y_end; ++y)
    {
      srcp = luma[0] + (y + 2) * pitch + 8;
      dstp = luma[1] + (y + 2) * pitch + 8;

      for (int x = 0; x < width; x += 8)
      {
        uint8_t* esi = (uint8_t*)srcp;
        uint8_t* edi = (uint8_t*)sadp;
        const int eax = pitch2; // pitch * sizeof(short)

        xmm6 = fours; // [4] * 8
        xmm7 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi)); // (  0,  0 )
        //sub esi, eax // esi = srcp - pitch

        xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2)); // ( -1,  0 )
        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2)); // (  1,  0 )
        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - eax - 2)); // ( -1, -1 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + eax + 2)); // (  1,  1 )

        xmm4 = xmm0;
        xmm5 = xmm1;
        xmm0 = _mm_sub_epi16(xmm0, xmm7);
        xmm1 = _mm_sub_epi16(xmm1, xmm7);
        xmm0 = _mm_abs_epi16(xmm0); // SSSE3
        xmm1 = _mm_abs_epi16(xmm1); // SSSE3
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);

        xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - eax)); // (  0, -1 )
        xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + eax)); // (  0,  1 )

        xmm4 = _mm_add_epi16(xmm4, xmm2);
        xmm5 = _mm_add_epi16(xmm5, xmm3);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);

        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm4 = _mm_abs_epi16(xmm4); // SSSE3
        xmm5 = _mm_abs_epi16(xmm5); // SSSE3
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm4 = _mm_add_epi16(xmm4, xmm6); // add "identification number" to the lower 3 bits (4)
        xmm6 = _mm_sub_epi16(xmm6, threes); // (The lower 3 bits are always zero.
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm4); // If input is 9-bit or more, this hack doesn't work)

        xmm4 = xmm2;
        xmm5 = xmm3;
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm2 = _mm_abs_epi16(xmm2); // SSSE3
        xmm3 = _mm_abs_epi16(xmm3); // SSSE3
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm2 = _mm_add_epi16(xmm2, xmm6);
        xmm6 = _mm_add_epi16(xmm6, fours);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm2);

        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - eax + 2)); // (  1, -1 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + eax - 2)); // ( -1,  1 )

        xmm4 = _mm_add_epi16(xmm4, xmm0);
        xmm5 = _mm_add_epi16(xmm5, xmm1);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);
        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm4 = _mm_abs_epi16(xmm4); // SSSE3
        xmm5 = _mm_abs_epi16(xmm5); // SSSE3
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm4 = _mm_add_epi16(xmm4, xmm6); // (5)
        xmm6 = _mm_sub_epi16(xmm6, threes);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm4);

        xmm4 = xmm0;
        xmm5 = xmm1;
        xmm0 = _mm_sub_epi16(xmm0, xmm7);
        xmm1 = _mm_sub_epi16(xmm1, xmm7);
        xmm0 = _mm_abs_epi16(xmm0); // SSSE3
        xmm1 = _mm_abs_epi16(xmm1); // SSSE3
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        xmm0 = _mm_add_epi16(xmm0, xmm6);
        xmm6 = _mm_add_epi16(xmm6, fours);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 64), xmm0);

        xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2)); // (  1,  0 )
        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2)); // ( -1,  0 )

        xmm4 = _mm_add_epi16(xmm4, xmm2);
        xmm5 = _mm_add_epi16(xmm5, xmm3);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);
        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm4 = _mm_abs_epi16(xmm4); // SSSE3
        xmm5 = _mm_abs_epi16(xmm5); // SSSE3
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm4 = _mm_add_epi16(xmm4, xmm6); // (6)
        xmm6 = _mm_sub_epi16(xmm6, threes);

        xmm4 = _mm_min_epi16(xmm4, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 0)));
        xmm4 = _mm_min_epi16(xmm4, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 16)));
        xmm4 = _mm_min_epi16(xmm4, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 32)));
        xmm4 = _mm_min_epi16(xmm4, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 48)));
        xmm4 = _mm_min_epi16(xmm4, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 64)));

        xmm0 = _mm_add_epi16(xmm0, xmm2);
        xmm1 = _mm_add_epi16(xmm1, xmm3);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_add_epi16(xmm3, xmm7);
        xmm2 = _mm_abs_epi16(xmm2); // SSSE3
        xmm3 = _mm_abs_epi16(xmm3); // SSSE3
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm2 = _mm_add_epi16(xmm2, xmm6); // (3)
        xmm6 = _mm_add_epi16(xmm6, fours);

        _mm_srai_epi16(xmm0, 1);
        _mm_srai_epi16(xmm1, 1);
        _mm_sub_epi16(xmm0, xmm7);
        _mm_sub_epi16(xmm1, xmm7);
        xmm0 = _mm_abs_epi16(xmm0); // SSSE3
        xmm1 = _mm_abs_epi16(xmm1); // SSSE3
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        xmm0 = _mm_add_epi16(xmm0, xmm6); // (7)

        xmm4 = _mm_min_epi16(xmm4, xmm2);
        xmm4 = _mm_min_epi16(xmm4, xmm0);

        _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm4);

        for (int i = 0; i < 8; ++i, ++srcp, ++dstp)
        {
          if ((sad[i] & ~7) == 0) { *dstp = *srcp; continue; }

          switch (sad[i] & 7)
          {
          case 0:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-1] + srcp[1]) + 32) >> 6; break;
          case 1:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch - 1] + srcp[pitch + 1]) + 32) >> 6; break;
          case 2:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch] + srcp[pitch]) + 32) >> 6; break;
          case 3:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch + 1] + srcp[pitch - 1]) + 32) >> 6; break;
          case 4:
            *dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch - 1] + srcp[-1] + srcp[1] + srcp[pitch + 1]) + 64) >> 7; break;
          case 5:
            *dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch - 1] + srcp[-pitch] + srcp[pitch] + srcp[pitch + 1]) + 64) >> 7; break;
          case 6:
            *dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch + 1] + srcp[-pitch] + srcp[pitch] + srcp[pitch - 1]) + 64) >> 7; break;
          case 7:
            *dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch + 1] + srcp[1] + srcp[-1] + srcp[pitch - 1]) + 64) >> 7; break;
          }
        }
      }
    }
  }
  else // radius == 2
  {
    const int coef0 = 128 - strength * 4; // own pixel's coefficient (when divisor = 128)
    const int coef1 = 256 - strength * 8; // own pixel's coefficient (when divisor = 256)
    const int coef2 = strength; // other pixel's coefficient
    const int coef3 = strength * 2; // other pixel's coefficient (doubled)

    for (int y = y_start; y < y_end; ++y)
    {
      srcp = luma[0] + (y + 2) * pitch + 8;
      dstp = luma[1] + (y + 2) * pitch + 8;

      for (int x = 0; x < width; x += 8)
      {
        uint8_t* esi = (uint8_t*)srcp;
        uint8_t* edi = (uint8_t*)sadp;
        const int eax = pitch2; //  eax = pitch * sizeof(short)
        xmm6 = fours; // xmm6 = [4] * 8
        xmm7 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi)); // (  0,  0 )

        xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2)); // ( -1,  0 )
        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2)); // (  1,  0 )
        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 4)); // ( -2,  0 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 4)); // (  2,  0 )
        xmm4 = xmm0;
        xmm5 = xmm1;
        xmm0 = _mm_sub_epi16(xmm0, xmm7);
        xmm1 = _mm_sub_epi16(xmm1, xmm7);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm0 = _mm_abs_epi16(xmm0);
        xmm1 = _mm_abs_epi16(xmm1);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm2);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);

        xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 1 * eax - 2)); // ( -1, -1 )
        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 1 * eax + 2)); // (  1,  1 )
        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 1 * eax - 4)); // ( -2, -1 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 1 * eax + 4)); // (  2,  1 )
        xmm4 = _mm_add_epi16(xmm4, xmm0);
        xmm5 = _mm_add_epi16(xmm5, xmm1);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm4 = _mm_abs_epi16(xmm4);
        xmm5 = _mm_abs_epi16(xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm4);
        xmm2 = _mm_add_epi16(xmm2, xmm6); // add "identification number" to the lower 3 bits (4)
        xmm6 = _mm_sub_epi16(xmm6, threes);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 16), xmm2);

        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2 * eax - 4)); // ( -2, -2 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax + 4)); // (  2,  2 )
        xmm4 = xmm0;
        xmm5 = xmm1;
        xmm0 = _mm_sub_epi16(xmm0, xmm7);
        xmm1 = _mm_sub_epi16(xmm1, xmm7);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm0 = _mm_abs_epi16(xmm0);
        xmm1 = _mm_abs_epi16(xmm1);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm2);
        xmm0 = _mm_add_epi16(xmm0, xmm6); // (1)
        xmm6 = _mm_add_epi16(xmm6, fours);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 32), xmm0);

        xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 1 * eax)); // (  0, -1 )
        xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 1 * eax)); // (  0,  1 )
        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2 * eax - 2)); // ( -1, -2 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax + 2)); // (  1,  2 )
        xmm4 = _mm_add_epi16(xmm4, xmm0);
        xmm5 = _mm_add_epi16(xmm5, xmm1);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm4 = _mm_abs_epi16(xmm4);
        xmm5 = _mm_abs_epi16(xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm4);
        xmm2 = _mm_add_epi16(xmm2, xmm6); // (5)
        xmm6 = _mm_sub_epi16(xmm6, threes);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 48), xmm2);

        xmm2 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi - 2 * eax)); // (  0, -2 )
        xmm3 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax)); // (  0,  2 )
        xmm4 = xmm0;
        xmm5 = xmm1;
        xmm0 = _mm_sub_epi16(xmm0, xmm7);
        xmm1 = _mm_sub_epi16(xmm1, xmm7);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm0 = _mm_abs_epi16(xmm0);
        xmm1 = _mm_abs_epi16(xmm1);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm2);
        xmm0 = _mm_add_epi16(xmm0, xmm6); // (2)
        xmm6 = _mm_add_epi16(xmm6, fours);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 64), xmm0);

        xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 1 * eax + 2)); // (  1, -1 )
        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 1 * eax - 2)); // ( -1,  1 )
        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2 * eax + 2)); // (  1, -2 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax - 2)); // ( -1,  2 )
        xmm4 = _mm_add_epi16(xmm4, xmm0);
        xmm5 = _mm_add_epi16(xmm5, xmm1);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm4 = _mm_abs_epi16(xmm4);
        xmm5 = _mm_abs_epi16(xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm4);
        xmm2 = _mm_add_epi16(xmm2, xmm6); // (6)
        xmm6 = _mm_sub_epi16(xmm6, threes);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi + 80), xmm2);

        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2 * eax + 4)); // (  2, -2 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2 * eax - 4)); // ( -2,  2 )
        xmm4 = xmm0;
        xmm5 = xmm1;
        xmm0 = _mm_sub_epi16(xmm0, xmm7);
        xmm1 = _mm_sub_epi16(xmm1, xmm7);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm0 = _mm_abs_epi16(xmm0);
        xmm1 = _mm_abs_epi16(xmm1);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm1);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm0 = _mm_add_epi16(xmm0, xmm2);
        xmm0 = _mm_add_epi16(xmm0, xmm6); // (3)
        xmm6 = _mm_add_epi16(xmm6, fours);

        xmm0 = _mm_min_epi16(xmm0, _mm_load_si128(reinterpret_cast<const __m128i*>(edi)));
        xmm0 = _mm_min_epi16(xmm0, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 16)));
        xmm0 = _mm_min_epi16(xmm0, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 32)));
        xmm0 = _mm_min_epi16(xmm0, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 48)));
        xmm0 = _mm_min_epi16(xmm0, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 64)));
        xmm0 = _mm_min_epi16(xmm0, _mm_load_si128(reinterpret_cast<const __m128i*>(edi + 80)));

        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 2)); // (  1,  0 )
        xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 1 * eax + 4)); // (  2, -1 )
        xmm3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi + 1 * eax - 4)); // ( -2,  1 )
        xmm4 = _mm_add_epi16(xmm4, xmm1);
        xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(esi - 2)); // ( -1,  0 )
        xmm5 = _mm_add_epi16(xmm5, xmm1);
        xmm4 = _mm_srai_epi16(xmm4, 1);
        xmm5 = _mm_srai_epi16(xmm5, 1);
        xmm2 = _mm_sub_epi16(xmm2, xmm7);
        xmm3 = _mm_sub_epi16(xmm3, xmm7);
        xmm4 = _mm_sub_epi16(xmm4, xmm7);
        xmm5 = _mm_sub_epi16(xmm5, xmm7);
        xmm2 = _mm_abs_epi16(xmm2);
        xmm3 = _mm_abs_epi16(xmm3);
        xmm4 = _mm_abs_epi16(xmm4);
        xmm5 = _mm_abs_epi16(xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm3);
        xmm4 = _mm_add_epi16(xmm4, xmm5);
        xmm2 = _mm_add_epi16(xmm2, xmm4);
        xmm2 = _mm_add_epi16(xmm2, xmm6); // (7)

        xmm0 = _mm_min_epi16(xmm0, xmm2);
        _mm_store_si128(reinterpret_cast<__m128i*>(edi), xmm0);

        for (int i = 0; i < 8; ++i, ++srcp, ++dstp)
        {
          if ((sad[i] & ~7) == 0) { *dstp = *srcp; continue; }

          switch (sad[i] & 7)
          {
          case 0:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-2] + srcp[-1] + srcp[1] + srcp[2]) + 64) >> 7; break;
          case 1:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch2 - 2] + srcp[-pitch - 1] + srcp[pitch + 1] + srcp[pitch2 + 2]) + 64) >> 7; break;
          case 2:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch2] + srcp[-pitch] + srcp[pitch] + srcp[pitch2]) + 64) >> 7; break;
          case 3:
            *dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch2 + 2] + srcp[-pitch + 1] + srcp[pitch - 1] + srcp[pitch2 - 2]) + 64) >> 7; break;
          case 4:
            *dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch - 2] + srcp[pitch + 2]) + coef2 * (srcp[-pitch - 1] + srcp[-1] + srcp[1] + srcp[pitch + 1]) + 128) >> 8; break;
          case 5:
            *dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch2 - 1] + srcp[pitch2 + 1]) + coef2 * (srcp[-pitch - 1] + srcp[-pitch] + srcp[pitch] + srcp[pitch + 1]) + 128) >> 8; break;
          case 6:
            *dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch2 + 1] + srcp[pitch2 - 1]) + coef2 * (srcp[-pitch + 1] + srcp[-pitch] + srcp[pitch] + srcp[pitch - 1]) + 128) >> 8; break;
          case 7:
            *dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch + 2] + srcp[pitch - 2]) + coef2 * (srcp[-pitch + 1] + srcp[1] + srcp[-1] + srcp[pitch - 1]) + 128) >> 8; break;
          }
        }
      }
    } // y
  } // radius 2

  // vertical reflection
  if (y_start <= 1 && 1 < y_end)
    memcpy(luma[1] + pitch, luma[1] + 3 * pitch, pitch * sizeof(short));
  if (y_start <= 2 && 2 < y_end)
    memcpy(luma[1], luma[1] + 4 * pitch, pitch * sizeof(short));
  if (y_start <= height - 3 && height - 3 < y_end)
    memcpy(luma[1] + (height + 3) * pitch, luma[1] + (height - 1) * pitch, pitch * sizeof(short));
  if (y_start <= height - 2 && height - 2 < y_end)
    memcpy(luma[1] + (height + 2) * pitch, luma[1] + height * pitch, pitch * sizeof(short));
}
