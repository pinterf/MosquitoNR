/*
**                      MosquitoNR ver 0.10
**
** Copyright (C) 2012-2013 Wataru Inariba <oinari17@gmail.com>
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// This program is compiled by VC++ 2010 Express.

#include "mosquito_nr.h"
#include <emmintrin.h>

// constructor
MosquitoNR::MosquitoNR(PClip _child, int _strength, int _restore, int _radius, int _threads, IScriptEnvironment* env)
  : GenericVideoFilter(_child), strength(_strength), restore(_restore), radius(_radius), threads(_threads),
  width(vi.width), height(vi.height), pitch(((width + 7) & ~7) + 16)
{
  // Check frame property support
  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  InitBuffer();

  // error checks
  if (!(env->GetCPUFlags() & CPUF_SSSE3))
    env->ThrowError("MosquitoNR: SSSE3 support is required.");
  if (!(vi.IsYUV() && vi.IsPlanar() && vi.BitsPerComponent() == 8))
    env->ThrowError("MosquitoNR: input must be 8-bit Y or YUV format.");
  if (width < 4 || height < 4) env->ThrowError("MosquitoNR: input is too small.");
  if (strength < 0 || 32 < strength) env->ThrowError("MosquitoNR: strength must be 0-32.");
  if (restore < 0 || 128 < restore) env->ThrowError("MosquitoNR: restore must be 0-128.");
  if (radius < 1 || 2 < radius) env->ThrowError("MosquitoNR: radius must be 1 or 2.");
  if (threads < 0 || MAX_THREADS < threads) env->ThrowError("MosquitoNR: threads must be 0(auto) or 1-%d.", MAX_THREADS);

  // detect the number of processors
  if (threads == 0) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    threads = min(si.dwNumberOfProcessors, MAX_THREADS);
  }

  // allocate buffer and create threads
  if (!AllocBuffer())
    env->ThrowError("MosquitoNR: failed to allocate buffer.");
  if (!mt.CreateThreads(threads, this))
    env->ThrowError("MosquitoNR: failed to create threads.");

}

// destructor
MosquitoNR::~MosquitoNR() { FreeBuffer(); }

// filter process
PVideoFrame __stdcall MosquitoNR::GetFrame(int n, IScriptEnvironment* env)
{
  src = child->GetFrame(n, env);
  dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

  // copy chroma
  if (!vi.IsY8() && vi.IsPlanar()) {
    env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U),
      src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
    env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V),
      src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
  }

  if (strength == 0) { // do nothing
    env->BitBlt(dst->GetWritePtr(), dst->GetPitch(), src->GetReadPtr(), src->GetPitch(), src->GetRowSize(), src->GetHeight());
    return dst;
  }

  CopyLumaFrom();
  mt.ExecMTFunc(&MosquitoNR::Smoothing);

  if (restore == 0) { // no restoring
    CopyLumaTo();
    return dst;
  }

  mt.ExecMTFunc(&MosquitoNR::WaveletVert1);
  mt.ExecMTFunc(&MosquitoNR::WaveletHorz1);
  mt.ExecMTFunc(&MosquitoNR::WaveletVert2);

  if (restore == 128) {
    mt.ExecMTFunc(&MosquitoNR::WaveletHorz2);
  }
  else {
    mt.ExecMTFunc(&MosquitoNR::WaveletHorz3);
    mt.ExecMTFunc(&MosquitoNR::BlendCoef);
  }

  mt.ExecMTFunc(&MosquitoNR::InvWaveletHorz);
  mt.ExecMTFunc(&MosquitoNR::InvWaveletVert);
  CopyLumaTo();

  return dst;
}

void MosquitoNR::InitBuffer()
{
  luma[0] = luma[1] = bufy[0] = bufy[1] = bufx[0] = bufx[1] = NULL;
  for (int i = 0; i < MAX_THREADS; ++i) work[i] = NULL;
}

bool MosquitoNR::AllocBuffer()
{
  FreeBuffer();

  luma[0] = (short*)_aligned_malloc((((height + 7) & ~7) + 4) * pitch * sizeof(short), 16);
  luma[1] = (short*)_aligned_malloc((((height + 7) & ~7) + 4) * pitch * sizeof(short), 16);
  bufy[0] = (short*)_aligned_malloc(((((height + 15) & ~15) / 2) + 1) * pitch * sizeof(short), 16);
  bufy[1] = (short*)_aligned_malloc(((((height + 15) & ~15) / 2) + 2) * pitch * sizeof(short), 16);
  bufx[0] = (short*)_aligned_malloc((((height + 15) & ~15) / 4) * pitch * sizeof(short), 16);
  bufx[1] = (short*)_aligned_malloc((((height + 15) & ~15) / 4) * pitch * sizeof(short), 16);

  if (!luma[0] || !luma[1] || !bufy[0] || !bufy[1] || !bufx[0] || !bufx[1]) return false;

  for (int i = 0; i < threads; ++i) {
    work[i] = (short*)_aligned_malloc(8 * pitch * sizeof(short), 16);
    if (!work[i]) return false;
  }

  return true;
}

void MosquitoNR::FreeBuffer()
{
  _aligned_free(luma[0]); _aligned_free(luma[1]);
  _aligned_free(bufy[0]); _aligned_free(bufy[1]);
  _aligned_free(bufx[0]); _aligned_free(bufx[1]);

  for (int i = 0; i < threads; ++i) _aligned_free(work[i]);

  InitBuffer();
}

void MosquitoNR::CopyLumaFrom()
{
  const int src_pitch = src->GetPitch();
  const auto dst_pitch = pitch * sizeof(short);
  const int height = this->height;
  const BYTE* srcp = src->GetReadPtr();
  short* dstp = luma[0];

  const int hloop = (width + 15) / 16;

  __m128i xmm0, xmm1, xmm7;
  uint8_t* edi = (uint8_t*)(dstp + 2 * pitch + 8); // edi = dstp + 2 * pitch + 8

  xmm7 = _mm_setzero_si128();

  for (int y = 0; y < height; y++) {
    //next16pixels_planar :
    for (int x = 0; x < hloop; x++) {
      xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 16)); // movdqu xmm0, [esi]
      xmm1 = xmm0;
      xmm0 = _mm_unpacklo_epi8(xmm0, xmm7);
      xmm1 = _mm_unpackhi_epi8(xmm1, xmm7);
      xmm0 = _mm_slli_epi16(xmm0, 4); // convert to internal 12-bit precision
      xmm1 = _mm_slli_epi16(xmm1, 4); // convert to internal 12-bit precision
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + x * 32), xmm0);
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + x * 32 + 16), xmm1);
    } // sub edx, 1         jnz next16pixels_planar

    srcp += src_pitch; // add esi, eax
    edi += dst_pitch; // add edi, ebx
  }

  // horizontal reflection
  short* p = luma[0] + 2 * pitch + 8;
  for (int y = 0; y < height; ++y, p += pitch)
    p[-2] = p[2], p[-1] = p[1], p[width] = p[width - 2], p[width + 1] = p[width - 3];

  // vertical reflection
  memcpy(luma[0], luma[0] + 4 * pitch, pitch * sizeof(short));
  memcpy(luma[0] + pitch, luma[0] + 3 * pitch, pitch * sizeof(short));
  memcpy(luma[0] + (height + 2) * pitch, luma[0] + height * pitch, pitch * sizeof(short));
  memcpy(luma[0] + (height + 3) * pitch, luma[0] + (height - 1) * pitch, pitch * sizeof(short));
}

void MosquitoNR::CopyLumaTo()
{
  const int src_pitch = pitch * sizeof(short);
  const int dst_pitch = dst->GetPitch();
  const int height = this->height;
  short* srcp = luma[1];
  BYTE* dstp = dst->GetWritePtr();

  const int hloop = (width + 15) / 16;

  __m128i xmm0, xmm1, xmm7;
  uint8_t* esi = (uint8_t*)(srcp + 2 * pitch + 8); // // esi = srcp + 2 * pitch + 8
  uint8_t* edi = (uint8_t*)dstp; //   mov edi, dstp // edi = dstp

  xmm7 = _mm_set1_epi16(8); // xmm7 = [0x0008] * 8 rounder

  // nextrow_planar:
  for (int y = 0; y < height; y++) {
    //next16pixels_planar :
    for (int x = 0; x < hloop; x++) {
      xmm0 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + x * 32));
      xmm1 = _mm_load_si128(reinterpret_cast<const __m128i*>(esi + x * 32 + 16));
      xmm0 = _mm_add_epi16(xmm0, xmm7); // paddw xmm0, xmm7
      xmm1 = _mm_add_epi16(xmm1, xmm7); // paddw xmm0, xmm7
      // paddw xmm1, xmm7
      xmm0 = _mm_srai_epi16(xmm0, 4); //  psraw xmm0, 4
      xmm1 = _mm_srai_epi16(xmm1, 4); //    psraw xmm1, 4

      xmm0 = _mm_packus_epi16(xmm0, xmm1); //  packuswb xmm0, xmm1
      _mm_store_si128(reinterpret_cast<__m128i*>(edi + x * 16), xmm0);
    } // sub edx, 1  jnz next16pixels_planar

    esi += src_pitch; // add esi, eax
    edi += dst_pitch; // add edi, ebx
  }
}

void MosquitoNR::Smoothing(int thread_id)
{
  SmoothingSSSE3(thread_id);
}

AVSValue __cdecl CreateMosquitoNR(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  const VideoInfo& vi_orig = args[0].AsClip()->GetVideoInfo();

  AVSValue new_args[1] = { args[0].AsClip() };
  PClip clip;
  if (vi_orig.IsYUY2()) {
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }
  else {
    clip = args[0].AsClip();
  }

  auto Result = new MosquitoNR(clip, args[1].AsInt(16), args[2].AsInt(128), args[3].AsInt(2), args[4].AsInt(0), env);

  if (vi_orig.IsYUY2()) {
    AVSValue new_args2[1] = { Result };
    return env->Invoke("ConvertToYUY2", AVSValue(new_args2, 1)).AsClip();
  }

  return Result;
}

const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("MosquitoNR", "c[strength]i[restore]i[radius]i[threads]i", CreateMosquitoNR, NULL);
  return "Mosquito noise reduction filter";
}
