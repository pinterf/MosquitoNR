/*
**                      MosquitoNR ver 0.10
**
**	Copyright (C) 2012-2013 Wataru Inariba <oinari17@gmail.com>
**
**	This program is free software; you can redistribute it and/or
**	modify it under the terms of the GNU General Public License
**	as published by the Free Software Foundation; either version 2
**	of the License, or (at your option) any later version.
**
**	This program is distributed in the hope that it will be useful,
**	but WITHOUT ANY WARRANTY; without even the implied warranty of
**	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**	GNU General Public License for more details.
**
**	You should have received a copy of the GNU General Public License
**	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// This program is compiled by VC++ 2010 Express.

#include "mosquito_nr.h"

// constructor
MosquitoNR::MosquitoNR(PClip _child, int _strength, int _restore, int _radius, int _threads, IScriptEnvironment* env)
  : GenericVideoFilter(_child), strength(_strength), restore(_restore), radius(_radius), threads(_threads),
  width(vi.width), height(vi.height), pitch(((width + 7) & ~7) + 16)
{
  InitBuffer();

  // error checks
  if (!(env->GetCPUFlags() & CPUF_SSE2))
    env->ThrowError("MosquitoNR: SSE2 support is required.");
  if (!(vi.IsYUY2() || (vi.IsYUV() && vi.IsPlanar() && vi.BytesFromPixels(1) == 1)))
    env->ThrowError("MosquitoNR: input must be YUY2 or 8-bit YUV planar format.");
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

  CPUCheck();
}

// destructor
MosquitoNR::~MosquitoNR() { FreeBuffer(); }

// filter process
PVideoFrame __stdcall MosquitoNR::GetFrame(int n, IScriptEnvironment* env)
{
  src = child->GetFrame(n, env);
  dst = env->NewVideoFrame(vi);

  // copy chroma
  if (!vi.IsY8() && vi.IsPlanar()) {
    env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetReadPtr(PLANAR_U), src->GetPitch(PLANAR_U),
      src->GetRowSize(PLANAR_U), src->GetHeight(PLANAR_U));
    env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetReadPtr(PLANAR_V), src->GetPitch(PLANAR_V),
      src->GetRowSize(PLANAR_V), src->GetHeight(PLANAR_V));
  }

  if (strength == 0) {	// do nothing
    env->BitBlt(dst->GetWritePtr(), dst->GetPitch(), src->GetReadPtr(), src->GetPitch(), src->GetRowSize(), src->GetHeight());
    return dst;
  }

  CopyLumaFrom();
  mt.ExecMTFunc(&MosquitoNR::Smoothing);

  if (restore == 0) {		// no restoring
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

void MosquitoNR::CPUCheck()
{
  unsigned tmp;

  __asm {
    mov		eax, 1
    cpuid
    mov		tmp, ecx
  }

  ssse3 = (tmp & 0x200) != 0;
}

void MosquitoNR::CopyLumaFrom()
{
  const int src_pitch = src->GetPitch();
  const int dst_pitch = pitch * sizeof(short);
  const int height = this->height;
  const BYTE* srcp = src->GetReadPtr();
  short* dstp = luma[0];

  if (vi.IsYUY2())	// YUY2
  {
    const int hloop = (width + 7) / 8;

    __asm
    {
      mov			esi, srcp			// esi = srcp
      mov			edi, dstp
      mov			eax, src_pitch		// eax = src_pitch
      mov			ebx, dst_pitch		// ebx = pitch * sizeof(short)
      mov			ecx, height			// ecx = height
      lea			edi, [edi + 2 * ebx + 16]	// edi = dstp + 2 * pitch + 8
      mov			edx, 00ff00ffh
      movd		xmm7, edx
      pshufd		xmm7, xmm7, 0		// xmm7 = [0x00ff] * 8

      align 16
      nextrow_yuy2:
      push		esi
        push		edi
        mov			edx, hloop			// edx = hloop

        align 16
        next8pixels_yuy2 :
        movdqu		xmm0, [esi]			// VYUYVYUYVYUYVYUY
        pand		xmm0, xmm7			// -Y-Y-Y-Y-Y-Y-Y-Y
        psllw		xmm0, 4				// convert to internal 12-bit precision
        movdqa[edi], xmm0
        add			esi, 16
        add			edi, 16
        sub			edx, 1
        jnz			next8pixels_yuy2

        pop			edi
        pop			esi
        add			esi, eax
        add			edi, ebx
        sub			ecx, 1
        jnz			nextrow_yuy2
    }
  }
  else	// planar format
  {
    const int hloop = (width + 15) / 16;

    __asm
    {
      mov			esi, srcp			// esi = srcp
      mov			edi, dstp
      mov			eax, src_pitch		// eax = src_pitch
      mov			ebx, dst_pitch		// ebx = pitch * sizeof(short)
      mov			ecx, height			// ecx = height
      lea			edi, [edi + 2 * ebx + 16]	// edi = dstp + 2 * pitch + 8
      pxor		xmm7, xmm7			// xmm7 = [0x00] * 16

      align 16
      nextrow_planar:
      push		esi
        push		edi
        mov			edx, hloop			// edx = hloop

        align 16
        next16pixels_planar :
        movdqu		xmm0, [esi]
        movdqa		xmm1, xmm0
        punpcklbw	xmm0, xmm7
        punpckhbw	xmm1, xmm7
        psllw		xmm0, 4				// convert to internal 12-bit precision
        psllw		xmm1, 4
        movdqa[edi], xmm0
        movdqa[edi + 16], xmm1
        add			esi, 16
        add			edi, 32
        sub			edx, 1
        jnz			next16pixels_planar

        pop			edi
        pop			esi
        add			esi, eax
        add			edi, ebx
        sub			ecx, 1
        jnz			nextrow_planar
    }
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

  if (vi.IsYUY2())	// YUY2
  {
    const int src_pitch2 = src->GetPitch();
    const int hloop = (width + 7) / 8;
    const BYTE* srcp2 = src->GetReadPtr();

    __asm
    {
      mov			esi, srcp
      mov			ebx, srcp2			// ebx = srcp2
      mov			edi, dstp			// edi = dstp
      mov			eax, src_pitch		// eax = pitch * sizeof(short)
      mov			ecx, height			// ecx = height
      lea			esi, [esi + 2 * eax + 16]	// esi = srcp + 2 * pitch + 8
      mov			edx, 000ff00ffh
      movd		xmm5, edx
      mov			edx, 0ff00ff00h
      movd		xmm6, edx
      mov			edx, 000080008h
      movd		xmm7, edx
      pxor		xmm4, xmm4			// xmm4 = [0x0000] * 8
      pshufd		xmm5, xmm5, 0		// xmm5 = [0x00ff] * 8
      pshufd		xmm6, xmm6, 0		// xmm6 = [0xff00] * 8
      pshufd		xmm7, xmm7, 0		// xmm7 = [0x0008] * 8

      align 16
      nextrow_yuy2:
      push		esi
        push		ebx
        push		edi
        mov			edx, hloop			// edx = hloop

        align 16
        next8pixels_yuy2:
      movdqa		xmm0, [esi]
        movdqu		xmm1, [ebx]
        paddw		xmm0, xmm7
        psraw		xmm0, 4
        pmaxsw		xmm0, xmm4
        pminsw		xmm0, xmm5			// -Y-Y-Y-Y-Y-Y-Y-Y
        pand		xmm1, xmm6			// V-U-V-U-V-U-V-U-
        por			xmm0, xmm1			// VYUYVYUYVYUYVYUY
        movdqa[edi], xmm0
        add			esi, 16
        add			ebx, 16
        add			edi, 16
        sub			edx, 1
        jnz			next8pixels_yuy2

        pop			edi
        pop			ebx
        pop			esi
        add			esi, eax
        add			ebx, src_pitch2
        add			edi, dst_pitch
        sub			ecx, 1
        jnz			nextrow_yuy2
    }
  }
  else	// planar format
  {
    const int hloop = (width + 15) / 16;

    __asm
    {
      mov			esi, srcp
      mov			edi, dstp			// edi = dstp
      mov			eax, src_pitch		// eax = pitch * sizeof(short)
      mov			ebx, dst_pitch		// ebx = dst_pitch
      mov			ecx, height			// ecx = height
      lea			esi, [esi + 2 * eax + 16]	// esi = srcp + 2 * pitch + 8
      mov			edx, 00080008h
      movd		xmm7, edx
      pshufd		xmm7, xmm7, 0		// xmm7 = [0x0008] * 8

      align 16
      nextrow_planar:
      push		esi
        push		edi
        mov			edx, hloop			// edx = hloop

        align 16
        next16pixels_planar :
        movdqa		xmm0, [esi]
        movdqa		xmm1, [esi + 16]
        paddw		xmm0, xmm7
        paddw		xmm1, xmm7
        psraw		xmm0, 4
        psraw		xmm1, 4
        packuswb	xmm0, xmm1
        movdqa[edi], xmm0
        add			esi, 32
        add			edi, 16
        sub			edx, 1
        jnz			next16pixels_planar

        pop			edi
        pop			esi
        add			esi, eax
        add			edi, ebx
        sub			ecx, 1
        jnz			nextrow_planar
    }
  }
}

void MosquitoNR::Smoothing(int thread_id)
{
  SmoothingSSSE3(thread_id);
}

AVSValue __cdecl CreateMosquitoNR(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  return new MosquitoNR(args[0].AsClip(), args[1].AsInt(16), args[2].AsInt(128), args[3].AsInt(2), args[4].AsInt(0), env);
}

const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("MosquitoNR", "c[strength]i[restore]i[radius]i[threads]i", CreateMosquitoNR, NULL);
  return "Mosquito noise reduction filter";
}
