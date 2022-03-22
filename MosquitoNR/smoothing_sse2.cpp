//------------------------------------------------------------------------------
//		smoothing_sse2.cpp
//------------------------------------------------------------------------------

#include "mosquito_nr.h"

// direction-aware blur
void MosquitoNR::SmoothingSSE2(int thread_id)
{
	const int y_start = height *  thread_id      / threads;
	const int y_end   = height * (thread_id + 1) / threads;
	if (y_start == y_end) return;
	const int width  = this->width;
	const int pitch  = this->pitch;
	const int pitch2 = pitch * sizeof(short);
	__declspec(align(16)) short sad[48], tmp[24];
	short *srcp, *dstp, *sadp = sad, *tmpp = tmp;
	for (int i =  8; i < 16; ++i) tmp[i] = 4;
	for (int i = 16; i < 24; ++i) tmp[i] = 3;

	if (radius == 1)
	{
		const int coef0 =  64 - strength * 2;	// own pixel's coefficient (when divisor = 64)
		const int coef1 = 128 - strength * 4;	// own pixel's coefficient (when divisor = 128)
		const int coef2 = strength;				// other pixel's coefficient

		for (int y = y_start; y < y_end; ++y)
		{
			srcp = luma[0] + (y + 2) * pitch + 8;
			dstp = luma[1] + (y + 2) * pitch + 8;

			for (int x = 0; x < width; x += 8)
			{
				// to compute an absolute value of xmm0, following code is used:
				//		pxor	xmm1, xmm1
				//		psubw	xmm1, xmm0
				//		pmaxsw	xmm0, xmm1

				__asm
				{
					mov			esi, srcp
					mov			edi, sadp				// edi = sadp
					mov			edx, tmpp				// edx = tmpp
					mov			eax, pitch2				// eax = pitch * sizeof(short)
					movdqa		xmm7, [esi]				// xmm7 = (  0,  0 )
					sub			esi, eax				// esi = srcp - pitch

					movdqu		xmm0, [esi+eax-2]		// xmm0 = ( -1,  0 )
					movdqu		xmm1, [esi+eax+2]		// xmm1 = (  1,  0 )
					movdqu		xmm2, [esi-2]			// xmm2 = ( -1, -1 )
					movdqu		xmm3, [esi+2*eax+2]		// xmm3 = (  1,  1 )

					movdqa		xmm4, xmm0
					movdqa		xmm5, xmm1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm0
					pmaxsw		xmm0, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm1
					pmaxsw		xmm1, xmm6
					paddw		xmm0, xmm1
					movdqa		[edi], xmm0

					movdqa		xmm0, [esi]				// xmm0 = (  0, -1 )
					movdqa		xmm1, [esi+2*eax]		// xmm1 = (  0,  1 )

					paddw		xmm4, xmm2
					paddw		xmm5, xmm3
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm4
					pmaxsw		xmm4, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm5
					pmaxsw		xmm5, xmm6
					movdqa		xmm6, [edx+16]
					paddw		xmm4, xmm5
					paddw		xmm4, xmm6				// add "identification number" to the lower 3 bits (4)
					psubw		xmm6, [edx+32]			// (The lower 3 bits are always zero.
					movdqa		[edi+16], xmm4			//  If input is 9-bit or more, this hack doesn't work)
					movdqa		[edx], xmm6

					movdqa		xmm4, xmm2
					movdqa		xmm5, xmm3
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm2, xmm3
					paddw		xmm2, xmm6				// (1)
					paddw		xmm6, [edx+16]
					movdqa		[edi+32], xmm2
					movdqa		[edx], xmm6

					movdqu		xmm2, [esi+2]			// xmm2 = (  1, -1 )
					movdqu		xmm3, [esi+2*eax-2]		// xmm3 = ( -1,  1 )

					paddw		xmm4, xmm0
					paddw		xmm5, xmm1
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm4
					pmaxsw		xmm4, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm5
					pmaxsw		xmm5, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm4, xmm5
					paddw		xmm4, xmm6				// (5)
					psubw		xmm6, [edx+32]
					movdqa		[edi+48], xmm4
					movdqa		[edx], xmm6

					movdqa		xmm4, xmm0
					movdqa		xmm5, xmm1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm0
					pmaxsw		xmm0, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm1
					pmaxsw		xmm1, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm0, xmm1
					paddw		xmm0, xmm6				// (2)
					paddw		xmm6, [edx+16]
					movdqa		[edi+64], xmm0
					movdqa		[edx], xmm6

					movdqu		xmm0, [esi+eax+2]		// xmm0 = (  1,  0 )
					movdqu		xmm1, [esi+eax-2]		// xmm1 = ( -1,  0 )

					paddw		xmm4, xmm2
					paddw		xmm5, xmm3
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm4
					pmaxsw		xmm4, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm5
					pmaxsw		xmm5, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm4, xmm5
					paddw		xmm4, xmm6				// (6)
					psubw		xmm6, [edx+32]

					pminsw		xmm4, [edi]
					pminsw		xmm4, [edi+16]
					pminsw		xmm4, [edi+32]
					pminsw		xmm4, [edi+48]
					pminsw		xmm4, [edi+64]

					paddw		xmm0, xmm2
					paddw		xmm1, xmm3
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					pxor		xmm5, xmm5
					psubw		xmm5, xmm2
					pmaxsw		xmm2, xmm5
					pxor		xmm5, xmm5
					psubw		xmm5, xmm3
					pmaxsw		xmm3, xmm5
					paddw		xmm2, xmm3
					paddw		xmm2, xmm6				// (3)
					paddw		xmm6, [edx+16]

					psraw		xmm0, 1
					psraw		xmm1, 1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					pxor		xmm3, xmm3
					pxor		xmm5, xmm5
					psubw		xmm3, xmm0
					psubw		xmm5, xmm1
					pmaxsw		xmm0, xmm3
					pmaxsw		xmm1, xmm5
					paddw		xmm0, xmm1
					paddw		xmm0, xmm6				// (7)

					pminsw		xmm4, xmm2
					pminsw		xmm4, xmm0
					movdqa		[edi], xmm4
				}
				
				for (int i = 0; i < 8; ++i, ++srcp, ++dstp)
				{
					if ((sad[i] &~ 7) == 0) { *dstp = *srcp; continue; }

					switch (sad[i] & 7)
					{
						case 0:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-1]       + srcp[1]      ) + 32) >> 6; break;
						case 1:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch-1] + srcp[pitch+1]) + 32) >> 6; break;
						case 2:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch]   + srcp[pitch]  ) + 32) >> 6; break;
						case 3:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch+1] + srcp[pitch-1]) + 32) >> 6; break;
						case 4:
							*dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch-1] + srcp[-1]     + srcp[1]     + srcp[pitch+1]) + 64) >> 7; break;
						case 5:
							*dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch-1] + srcp[-pitch] + srcp[pitch] + srcp[pitch+1]) + 64) >> 7; break;
						case 6:
							*dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch+1] + srcp[-pitch] + srcp[pitch] + srcp[pitch-1]) + 64) >> 7; break;
						case 7:
							*dstp = (coef1 * srcp[0] + coef2 * (srcp[-pitch+1] + srcp[1]      + srcp[-1]    + srcp[pitch-1]) + 64) >> 7; break;
					}
				}
			}
		}
	}
	else	// radius == 2
	{
		const int coef0 = 128 - strength * 4;	// own pixel's coefficient (when divisor = 128)
		const int coef1 = 256 - strength * 8;	// own pixel's coefficient (when divisor = 256)
		const int coef2 = strength;				// other pixel's coefficient
		const int coef3 = strength * 2;			// other pixel's coefficient (doubled)

		for (int y = y_start; y < y_end; ++y)
		{
			srcp = luma[0] + (y + 2) * pitch + 8;
			dstp = luma[1] + (y + 2) * pitch + 8;

			for (int x = 0; x < width; x += 8)
			{
				__asm
				{
					mov			esi, srcp
					mov			edi, sadp				// edi = sadp
					mov			edx, tmpp				// edx = tmpp
					mov			eax, pitch2				// eax = pitch * sizeof(short)
					lea			ecx, [eax+2*eax]		// ecx = pitch * sizeof(short) * 3
					movdqa		xmm7, [esi]				// xmm7 = (  0,  0 )
					sub			esi, eax
					sub			esi, eax				// esi = srcp - 2 * pitch

					movdqu		xmm0, [esi+2*eax-2]		// xmm0 = ( -1,  0 )
					movdqu		xmm1, [esi+2*eax+2]		// xmm1 = (  1,  0 )
					movdqu		xmm2, [esi+2*eax-4]		// xmm2 = ( -2,  0 )
					movdqu		xmm3, [esi+2*eax+4]		// xmm3 = (  2,  0 )
					movdqa		xmm4, xmm0
					movdqa		xmm5, xmm1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm0
					pmaxsw		xmm0, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm1
					pmaxsw		xmm1, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					paddw		xmm0, xmm1
					paddw		xmm2, xmm3
					paddw		xmm0, xmm2
					movdqa		[edi], xmm0

					movdqu		xmm0, [esi+eax-2]		// xmm0 = ( -1, -1 )
					movdqu		xmm1, [esi+ecx+2]		// xmm1 = (  1,  1 )
					movdqu		xmm2, [esi+eax-4]		// xmm2 = ( -2, -1 )
					movdqu		xmm3, [esi+ecx+4]		// xmm3 = (  2,  1 )
					paddw		xmm4, xmm0
					paddw		xmm5, xmm1
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm4
					pmaxsw		xmm4, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm5
					pmaxsw		xmm5, xmm6
					movdqa		xmm6, [edx+16]
					paddw		xmm2, xmm3
					paddw		xmm4, xmm5
					paddw		xmm2, xmm4
					paddw		xmm2, xmm6				// add "identification number" to the lower 3 bits (4)
					psubw		xmm6, [edx+32]
					movdqa		[edi+16], xmm2
					movdqa		[edx], xmm6

					movdqu		xmm2, [esi-4]			// xmm2 = ( -2, -2 )
					movdqu		xmm3, [esi+4*eax+4]		// xmm3 = (  2,  2 )
					movdqa		xmm4, xmm0
					movdqa		xmm5, xmm1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm0
					pmaxsw		xmm0, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm1
					pmaxsw		xmm1, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm0, xmm1
					paddw		xmm2, xmm3
					paddw		xmm0, xmm2
					paddw		xmm0, xmm6				// (1)
					paddw		xmm6, [edx+16]
					movdqa		[edi+32], xmm0
					movdqa		[edx], xmm6

					movdqa		xmm0, [esi+eax]			// xmm0 = (  0, -1 )
					movdqa		xmm1, [esi+ecx]			// xmm1 = (  0,  1 )
					movdqu		xmm2, [esi-2]			// xmm2 = ( -1, -2 )
					movdqu		xmm3, [esi+4*eax+2]		// xmm3 = (  1,  2 )
					paddw		xmm4, xmm0
					paddw		xmm5, xmm1
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm4
					pmaxsw		xmm4, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm5
					pmaxsw		xmm5, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm2, xmm3
					paddw		xmm4, xmm5
					paddw		xmm2, xmm4
					paddw		xmm2, xmm6				// (5)
					psubw		xmm6, [edx+32]
					movdqa		[edi+48], xmm2
					movdqa		[edx], xmm6

					movdqa		xmm2, [esi]				// xmm2 = (  0, -2 )
					movdqa		xmm3, [esi+4*eax]		// xmm3 = (  0,  2 )
					movdqa		xmm4, xmm0
					movdqa		xmm5, xmm1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm0
					pmaxsw		xmm0, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm1
					pmaxsw		xmm1, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm0, xmm1
					paddw		xmm2, xmm3
					paddw		xmm0, xmm2
					paddw		xmm0, xmm6				// (2)
					paddw		xmm6, [edx+16]
					movdqa		[edi+64], xmm0
					movdqa		[edx], xmm6

					movdqu		xmm0, [esi+eax+2]		// xmm0 = (  1, -1 )
					movdqu		xmm1, [esi+ecx-2]		// xmm1 = ( -1,  1 )
					movdqu		xmm2, [esi+2]			// xmm2 = (  1, -2 )
					movdqu		xmm3, [esi+4*eax-2]		// xmm3 = ( -1,  2 )
					paddw		xmm4, xmm0
					paddw		xmm5, xmm1
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm4
					pmaxsw		xmm4, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm5
					pmaxsw		xmm5, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm2, xmm3
					paddw		xmm4, xmm5
					paddw		xmm2, xmm4
					paddw		xmm2, xmm6				// (6)
					psubw		xmm6, [edx+32]
					movdqa		[edi+80], xmm2
					movdqa		[edx], xmm6

					movdqu		xmm2, [esi+4]			// xmm2 = (  2, -2 )
					movdqu		xmm3, [esi+4*eax-4]		// xmm3 = ( -2,  2 )
					movdqa		xmm4, xmm0
					movdqa		xmm5, xmm1
					psubw		xmm0, xmm7
					psubw		xmm1, xmm7
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					pxor		xmm6, xmm6
					psubw		xmm6, xmm0
					pmaxsw		xmm0, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm1
					pmaxsw		xmm1, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm2
					pmaxsw		xmm2, xmm6
					pxor		xmm6, xmm6
					psubw		xmm6, xmm3
					pmaxsw		xmm3, xmm6
					movdqa		xmm6, [edx]
					paddw		xmm0, xmm1
					paddw		xmm2, xmm3
					paddw		xmm0, xmm2
					paddw		xmm0, xmm6				// (3)
					paddw		xmm6, [edx+16]
					movdqa		[edx], xmm6

					pminsw		xmm0, [edi]
					pminsw		xmm0, [edi+16]
					pminsw		xmm0, [edi+32]
					pminsw		xmm0, [edi+48]
					pminsw		xmm0, [edi+64]
					pminsw		xmm0, [edi+80]

					movdqu		xmm6, [esi+2*eax+2]		// xmm6 = (  1,  0 )
					movdqu		xmm1, [esi+2*eax-2]		// xmm1 = ( -1,  0 )
					movdqu		xmm2, [esi+eax+4]		// xmm2 = (  2, -1 )
					movdqu		xmm3, [esi+ecx-4]		// xmm3 = ( -2,  1 )
					paddw		xmm4, xmm6
					paddw		xmm5, xmm1
					psraw		xmm4, 1
					psraw		xmm5, 1
					psubw		xmm2, xmm7
					psubw		xmm3, xmm7
					psubw		xmm4, xmm7
					psubw		xmm5, xmm7
					pxor		xmm6, xmm6
					pxor		xmm7, xmm7
					psubw		xmm6, xmm2
					psubw		xmm7, xmm3
					pmaxsw		xmm2, xmm6
					pmaxsw		xmm3, xmm7
					pxor		xmm6, xmm6
					pxor		xmm7, xmm7
					psubw		xmm6, xmm4
					psubw		xmm7, xmm5
					pmaxsw		xmm4, xmm6
					pmaxsw		xmm5, xmm7
					movdqa		xmm6, [edx]
					paddw		xmm2, xmm3
					paddw		xmm4, xmm5
					paddw		xmm2, xmm4
					paddw		xmm2, xmm6				// (7)

					pminsw		xmm0, xmm2
					movdqa		[edi], xmm0
				}
			
				for (int i = 0; i < 8; ++i, ++srcp, ++dstp)
				{
					if ((sad[i] &~ 7) == 0) { *dstp = *srcp; continue; }

					switch (sad[i] & 7)
					{
						case 0:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-2]        + srcp[-1]       + srcp[1]       + srcp[2]       ) + 64) >> 7; break;
						case 1:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch2-2] + srcp[-pitch-1] + srcp[pitch+1] + srcp[pitch2+2]) + 64) >> 7; break;
						case 2:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch2]   + srcp[-pitch]   + srcp[pitch]   + srcp[pitch2]  ) + 64) >> 7; break;
						case 3:
							*dstp = (coef0 * srcp[0] + coef2 * (srcp[-pitch2+2] + srcp[-pitch+1] + srcp[pitch-1] + srcp[pitch2-2]) + 64) >> 7; break;
						case 4:
							*dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch -2] + srcp[pitch +2]) + coef2 * (srcp[-pitch-1] + srcp[-1]     + srcp[1]     + srcp[pitch+1]) + 128) >> 8; break;
						case 5:
							*dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch2-1] + srcp[pitch2+1]) + coef2 * (srcp[-pitch-1] + srcp[-pitch] + srcp[pitch] + srcp[pitch+1]) + 128) >> 8; break;
						case 6:
							*dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch2+1] + srcp[pitch2-1]) + coef2 * (srcp[-pitch+1] + srcp[-pitch] + srcp[pitch] + srcp[pitch-1]) + 128) >> 8; break;
						case 7:
							*dstp = (coef1 * srcp[0] + coef3 * (srcp[-pitch +2] + srcp[pitch -2]) + coef2 * (srcp[-pitch+1] + srcp[1]      + srcp[-1]    + srcp[pitch-1]) + 128) >> 8; break;
					}
				}
			}
		}
	}

	// vertical reflection
	if (y_start <= 1 && 1 < y_end)
		memcpy(luma[1] + pitch, luma[1] + 3 * pitch, pitch * sizeof(short));
	if (y_start <= 2 && 2 < y_end)
		memcpy(luma[1],         luma[1] + 4 * pitch, pitch * sizeof(short));
	if (y_start <= height - 3 && height - 3 < y_end)
		memcpy(luma[1] + (height + 3) * pitch, luma[1] + (height - 1) * pitch, pitch * sizeof(short));
	if (y_start <= height - 2 && height - 2 < y_end)
		memcpy(luma[1] + (height + 2) * pitch, luma[1] +  height      * pitch, pitch * sizeof(short));
}
