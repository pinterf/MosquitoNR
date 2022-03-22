//------------------------------------------------------------------------------
//		wavelet.cpp
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

void MosquitoNR::WaveletVert1(int thread_id)
{
	const int y_start = (height + 7) / 8 *  thread_id      / threads * 8;
	const int y_end   = (height + 7) / 8 * (thread_id + 1) / threads * 8;
	if (y_start == y_end) return;
	const int width = this->width;
	const int pitch = this->pitch;
	const int hloop = (width + 7) / 8;

	for (int y = y_start; y < y_end; y += 8)
	{
		short* srcp = luma[0] + y * pitch + 8;
		short* dstp = bufy[0] + y / 2 * pitch + 8;

		__asm
		{
			mov			esi, srcp			// esi = srcp
			mov			edi, dstp			// edi = dstp
			mov			eax, pitch
			mov			ecx, hloop			// ecx = hloop
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3

align 16
next8columns:
			push		esi
			movdqa		xmm2, [esi]
			movdqa		xmm0, [esi+eax]
			movdqa		xmm1, [esi+2*eax]
			paddw		xmm2, xmm1
			psraw		xmm2, 1
			psubw		xmm0, xmm2
			add			esi, ebx

			movdqa		xmm2, [esi]
			movdqa		xmm3, [esi+eax]
			movdqa		xmm4, [esi+2*eax]
			movdqa		xmm5, [esi+ebx]
			movdqa		xmm6, xmm1
			movdqa		xmm7, xmm3
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 1
			psraw		xmm3, 1
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			paddw		xmm0, xmm2
			paddw		xmm2, xmm4
			psraw		xmm0, 2
			psraw		xmm2, 2
			paddw		xmm6, xmm0
			paddw		xmm7, xmm2
			movdqa		[edi], xmm6
			movdqa		[edi+eax], xmm7
			lea			esi, [esi+4*eax]

			movdqa		xmm0, [esi]
			movdqa		xmm1, [esi+eax]
			movdqa		xmm2, [esi+2*eax]
			movdqa		xmm3, [esi+ebx]
			movdqa		xmm6, xmm5
			movdqa		xmm7, xmm1
			paddw		xmm5, xmm1
			paddw		xmm1, xmm3
			psraw		xmm5, 1
			psraw		xmm1, 1
			psubw		xmm0, xmm5
			psubw		xmm2, xmm1
			paddw		xmm4, xmm0
			paddw		xmm0, xmm2
			psraw		xmm4, 2
			psraw		xmm0, 2
			paddw		xmm6, xmm4
			paddw		xmm7, xmm0
			movdqa		[edi+2*eax], xmm6
			movdqa		[edi+ebx], xmm7

			pop			esi
			add			esi, 16
			add			edi, 16
			sub			ecx, 1
			jnz			next8columns
		}

		// horizontal reflection
		short* p = dstp;
		for (int i = 0; i < 4; ++i, p += pitch)
			p[-2] = p[2], p[-1] = p[1], p[width] = p[width-2], p[width+1] = p[width-3];
	}
}

void MosquitoNR::WaveletHorz1(int thread_id)
{
	const int y_start = (height + 15) / 16 *  thread_id      / threads * 8;
	const int y_end   = (height + 15) / 16 * (thread_id + 1) / threads * 8;
	if (y_start == y_end) return;
	const int width  = this->width;
	const int pitch  = this->pitch;
	const int hloop1 = (width + 4 + 2 + 3) / 4;
	const int hloop2 = (width + 3) / 4;
	short* work = this->work[thread_id];

	for (int y = y_start; y < y_end; y += 8)
	{
		short* srcp = bufy[0] + y * pitch + 4;
		short* dstp = luma[0] + y / 2 * pitch + 8;

		__asm
		{
			// shuffle
			mov			esi, srcp			// esi = srcp
			mov			edi, work			// edi = work
			mov			eax, pitch
			mov			ecx, hloop1			// ecx = hloop1
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3
			lea			edx, [esi+4*eax]	// edx = srcp + 4 * pitch

align 16
shuffle_next4columns:
			movq		xmm0, qword ptr [esi]				// 03, 02, 01, 00
			movq		xmm1, qword ptr [esi+eax]			// 13, 12, 11, 10
			movq		xmm2, qword ptr [esi+2*eax]			// 23, 22, 21, 20
			movq		xmm3, qword ptr [esi+ebx]			// 33, 32, 31, 30
			movq		xmm4, qword ptr [edx]				// 43, 42, 41, 40
			movq		xmm5, qword ptr [edx+eax]			// 53, 52, 51, 50
			movq		xmm6, qword ptr [edx+2*eax]			// 63, 62, 61, 60
			movq		xmm7, qword ptr [edx+ebx]			// 73, 72, 71, 70
			punpcklwd	xmm0, xmm1			// 13, 03, 12, 02, 11, 01, 10, 00
			punpcklwd	xmm2, xmm3			// 33, 23, 32, 22, 31, 21, 30, 20
			punpcklwd	xmm4, xmm5			// 53, 43, 52, 42, 51, 41, 50, 40
			punpcklwd	xmm6, xmm7			// 73, 63, 72, 62, 71, 61, 70, 60
			movdqa		xmm1, xmm0
			movdqa		xmm5, xmm4
			punpckldq	xmm0, xmm2			// 31, 21, 11, 01, 30, 20, 10, 00
			punpckhdq	xmm1, xmm2			// 33, 23, 13, 03, 32, 22, 12, 02
			punpckldq	xmm4, xmm6			// 71, 61, 51, 41, 70, 60, 50, 40
			punpckhdq	xmm5, xmm6			// 73, 63, 53, 43, 72, 62, 52, 42
			movdqa		xmm2, xmm0
			movdqa		xmm3, xmm1
			punpcklqdq	xmm0, xmm4			// 70, 60, 50, 40, 30, 20, 10, 00
			punpckhqdq	xmm2, xmm4			// 71, 61, 51, 41, 31, 21, 11, 01
			punpcklqdq	xmm1, xmm5			// 72, 62, 52, 42, 32, 22, 12, 02
			punpckhqdq	xmm3, xmm5			// 73, 63, 53, 43, 33, 23, 13, 03
			movdqa		[edi], xmm0
			movdqa		[edi+16], xmm2
			movdqa		[edi+32], xmm1
			movdqa		[edi+48], xmm3
			add			esi, 8
			add			edx, 8
			add			edi, 64
			sub			ecx, 1
			jnz			shuffle_next4columns

			// wavelet transform
			mov			esi, work
			mov			edi, dstp			// edi = dstp
			mov			ecx, hloop2			// ecx = hloop2
			add			esi, 64				// esi = work + 32

			movdqa		xmm2, [esi-32]
			movdqa		xmm0, [esi-16]
			movdqa		xmm1, [esi]
			paddw		xmm2, xmm1
			psraw		xmm2, 1
			psubw		xmm0, xmm2

align 16
wavelet_next4columns:
			movdqa		xmm2, [esi+16]
			movdqa		xmm3, [esi+32]
			movdqa		xmm4, [esi+48]
			movdqa		xmm5, [esi+64]
			movdqa		xmm6, xmm1
			movdqa		xmm7, xmm3
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 1
			psraw		xmm3, 1
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			paddw		xmm0, xmm2
			paddw		xmm2, xmm4
			psraw		xmm0, 2
			psraw		xmm2, 2
			paddw		xmm6, xmm0
			paddw		xmm7, xmm2
			movdqa		[edi], xmm6
			movdqa		[edi+16], xmm7
			movdqa		xmm0, xmm4
			movdqa		xmm1, xmm5
			add			esi, 64
			add			edi, 32
			add			edx, 32
			sub			ecx, 1
			jnz			wavelet_next4columns

			// horizontal reflection
			mov			eax, width
			test		eax, 1
			jnz			no_reflection		// if (width % 2 == 0) {
			mov			edi, dstp			//   edi = dstp
			shl			eax, 3				//   eax = width / 2 * 16
			movdqa		xmm0, [edi+eax-16]
			movdqa		[edi+eax], xmm0
no_reflection:								// }
		}
	}
}

void MosquitoNR::WaveletVert2(int thread_id)
{
	const int y_start = (height + 7) / 8 *  thread_id      / threads * 8;
	const int y_end   = (height + 7) / 8 * (thread_id + 1) / threads * 8;
	if (y_start == y_end) return;
	const int width = this->width;
	const int pitch = this->pitch;
	const int hloop = (width + 7) / 8;

	for (int y = y_start; y < y_end; y += 8)
	{
		short* srcp = luma[1] + y * pitch + 8;
		short* dstp1 = bufy[0] +  y / 2      * pitch + 8;
		short* dstp2 = bufy[1] + (y / 2 + 1) * pitch + 8;

		__asm
		{
			mov			esi, srcp			// esi = srcp
			mov			edi, dstp1			// edi = dstp1
			mov			edx, dstp2			// edx = dstp2
			mov			eax, pitch
			mov			ecx, hloop			// ecx = hloop
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3

align 16
next8columns:
			push		esi
			movdqa		xmm2, [esi]
			movdqa		xmm0, [esi+eax]
			movdqa		xmm1, [esi+2*eax]
			paddw		xmm2, xmm1
			psraw		xmm2, 1
			psubw		xmm0, xmm2
			add			esi, ebx

			movdqa		xmm2, [esi]
			movdqa		xmm3, [esi+eax]
			movdqa		xmm4, [esi+2*eax]
			movdqa		xmm5, [esi+ebx]
			movdqa		xmm6, xmm1
			movdqa		xmm7, xmm3
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 1
			psraw		xmm3, 1
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			movdqa		[edx], xmm2
			movdqa		[edx+eax], xmm4
			paddw		xmm0, xmm2
			paddw		xmm2, xmm4
			psraw		xmm0, 2
			psraw		xmm2, 2
			paddw		xmm6, xmm0
			paddw		xmm7, xmm2
			movdqa		[edi], xmm6
			movdqa		[edi+eax], xmm7
			lea			esi, [esi+4*eax]

			movdqa		xmm0, [esi]
			movdqa		xmm1, [esi+eax]
			movdqa		xmm2, [esi+2*eax]
			movdqa		xmm3, [esi+ebx]
			movdqa		xmm6, xmm5
			movdqa		xmm7, xmm1
			paddw		xmm5, xmm1
			paddw		xmm1, xmm3
			psraw		xmm5, 1
			psraw		xmm1, 1
			psubw		xmm0, xmm5
			psubw		xmm2, xmm1
			movdqa		[edx+2*eax], xmm0
			movdqa		[edx+ebx], xmm2
			paddw		xmm4, xmm0
			paddw		xmm0, xmm2
			psraw		xmm4, 2
			psraw		xmm0, 2
			paddw		xmm6, xmm4
			paddw		xmm7, xmm0
			movdqa		[edi+2*eax], xmm6
			movdqa		[edi+ebx], xmm7

			pop			esi
			add			esi, 16
			add			edi, 16
			add			edx, 16
			sub			ecx, 1
			jnz			next8columns
		}

		// horizontal reflection
		short* p = dstp1;
		for (int i = 0; i < 4; ++i, p += pitch)
			p[-2] = p[2], p[-1] = p[1], p[width] = p[width-2], p[width+1] = p[width-3];
	}

	// vertical reflection
	if (y_start == 0)
		memcpy(bufy[1], bufy[1] + pitch, pitch * sizeof(short));
	if (thread_id == threads - 1 && height % 2 == 0)
		memcpy(bufy[1] + (height / 2 + 1) * pitch, bufy[1] + (height / 2 - 1) * pitch, pitch * sizeof(short));
}

void MosquitoNR::WaveletHorz2(int thread_id)
{
	const int y_start = (height + 15) / 16 *  thread_id      / threads * 8;
	const int y_end   = (height + 15) / 16 * (thread_id + 1) / threads * 8;
	if (y_start == y_end) return;
	const int width  = this->width;
	const int pitch  = this->pitch;
	const int hloop1 = (width + 4 + 2 + 3) / 4;
	const int hloop2 = (width + 7) / 8;
	short* work = this->work[thread_id];

	for (int y = y_start; y < y_end; y += 8)
	{
		short* srcp = bufy[0] + y * pitch + 4;
		short* dstp = bufx[1] + y / 2 * pitch + 8;

		__asm
		{
			// shuffle
			mov			esi, srcp			// esi = srcp
			mov			edi, work			// edi = work
			mov			eax, pitch
			mov			ecx, hloop1			// ecx = hloop1
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3
			lea			edx, [esi+4*eax]	// edx = srcp + 4 * pitch

align 16
shuffle_next4columns:
			movq		xmm0, qword ptr [esi]				// 03, 02, 01, 00
			movq		xmm1, qword ptr [esi+eax]			// 13, 12, 11, 10
			movq		xmm2, qword ptr [esi+2*eax]			// 23, 22, 21, 20
			movq		xmm3, qword ptr [esi+ebx]			// 33, 32, 31, 30
			movq		xmm4, qword ptr [edx]				// 43, 42, 41, 40
			movq		xmm5, qword ptr [edx+eax]			// 53, 52, 51, 50
			movq		xmm6, qword ptr [edx+2*eax]			// 63, 62, 61, 60
			movq		xmm7, qword ptr [edx+ebx]			// 73, 72, 71, 70
			punpcklwd	xmm0, xmm1			// 13, 03, 12, 02, 11, 01, 10, 00
			punpcklwd	xmm2, xmm3			// 33, 23, 32, 22, 31, 21, 30, 20
			punpcklwd	xmm4, xmm5			// 53, 43, 52, 42, 51, 41, 50, 40
			punpcklwd	xmm6, xmm7			// 73, 63, 72, 62, 71, 61, 70, 60
			movdqa		xmm1, xmm0
			movdqa		xmm5, xmm4
			punpckldq	xmm0, xmm2			// 31, 21, 11, 01, 30, 20, 10, 00
			punpckhdq	xmm1, xmm2			// 33, 23, 13, 03, 32, 22, 12, 02
			punpckldq	xmm4, xmm6			// 71, 61, 51, 41, 70, 60, 50, 40
			punpckhdq	xmm5, xmm6			// 73, 63, 53, 43, 72, 62, 52, 42
			movdqa		xmm2, xmm0
			movdqa		xmm3, xmm1
			punpcklqdq	xmm0, xmm4			// 70, 60, 50, 40, 30, 20, 10, 00
			punpckhqdq	xmm2, xmm4			// 71, 61, 51, 41, 31, 21, 11, 01
			punpcklqdq	xmm1, xmm5			// 72, 62, 52, 42, 32, 22, 12, 02
			punpckhqdq	xmm3, xmm5			// 73, 63, 53, 43, 33, 23, 13, 03
			movdqa		[edi], xmm0
			movdqa		[edi+16], xmm2
			movdqa		[edi+32], xmm1
			movdqa		[edi+48], xmm3
			add			esi, 8
			add			edx, 8
			add			edi, 64
			sub			ecx, 1
			jnz			shuffle_next4columns

			// wavelet transform
			mov			esi, work
			mov			edi, dstp			// edi = dstp
			mov			ecx, hloop2			// ecx = hloop2
			add			esi, 64				// esi = work + 32

			movdqa		xmm2, [esi-32]
			movdqa		xmm0, [esi-16]
			movdqa		xmm1, [esi]
			paddw		xmm2, xmm1
			psraw		xmm2, 1
			psubw		xmm0, xmm2
			movdqa		[edi-16], xmm0

align 16
wavelet_next8columns:
			movdqa		xmm2, [esi+16]
			movdqa		xmm3, [esi+32]
			movdqa		xmm4, [esi+48]
			movdqa		xmm5, [esi+64]
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 1
			psraw		xmm3, 1
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			movdqa		[edi], xmm2
			movdqa		[edi+16], xmm4
			movdqa		xmm1, xmm5
			movdqa		xmm2, [esi+80]
			movdqa		xmm3, [esi+96]
			movdqa		xmm4, [esi+112]
			movdqa		xmm5, [esi+128]
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 1
			psraw		xmm3, 1
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			movdqa		[edi+32], xmm2
			movdqa		[edi+48], xmm4
			movdqa		xmm1, xmm5
			add			esi, 128
			add			edi, 64
			sub			ecx, 1
			jnz			wavelet_next8columns

			// horizontal reflection
			mov			eax, width
			test		eax, 1
			jnz			no_reflection		// if (width % 2 == 0) {
			mov			edi, dstp			//   edi = dstp
			shl			eax, 3				//   eax = width / 2 * 16
			movdqa		xmm0, [edi+eax-32]
			movdqa		[edi+eax], xmm0
no_reflection:								// }
		}
	}
}

void MosquitoNR::WaveletHorz3(int thread_id)
{
	const int y_start = (height + 15) / 16 *  thread_id      / threads * 8;
	const int y_end   = (height + 15) / 16 * (thread_id + 1) / threads * 8;
	if (y_start == y_end) return;
	const int width  = this->width;
	const int pitch  = this->pitch;
	const int hloop1 = (width + 4 + 2 + 3) / 4;
	const int hloop2 = (width + 3) / 4;
	short* work = this->work[thread_id];

	for (int y = y_start; y < y_end; y += 8)
	{
		short* srcp = bufy[0] + y * pitch + 4;
		short* dstp1 = bufx[0] + y / 2 * pitch + 8;
		short* dstp2 = bufx[1] + y / 2 * pitch + 8;

		__asm
		{
			// shuffle
			mov			esi, srcp			// esi = srcp
			mov			edi, work			// edi = work
			mov			eax, pitch
			mov			ecx, hloop1			// ecx = hloop1
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3
			lea			edx, [esi+4*eax]	// edx = srcp + 4 * pitch

align 16
shuffle_next4columns:
			movq		xmm0, qword ptr [esi]				// 03, 02, 01, 00
			movq		xmm1, qword ptr [esi+eax]			// 13, 12, 11, 10
			movq		xmm2, qword ptr [esi+2*eax]			// 23, 22, 21, 20
			movq		xmm3, qword ptr [esi+ebx]			// 33, 32, 31, 30
			movq		xmm4, qword ptr [edx]				// 43, 42, 41, 40
			movq		xmm5, qword ptr [edx+eax]			// 53, 52, 51, 50
			movq		xmm6, qword ptr [edx+2*eax]			// 63, 62, 61, 60
			movq		xmm7, qword ptr [edx+ebx]			// 73, 72, 71, 70
			punpcklwd	xmm0, xmm1			// 13, 03, 12, 02, 11, 01, 10, 00
			punpcklwd	xmm2, xmm3			// 33, 23, 32, 22, 31, 21, 30, 20
			punpcklwd	xmm4, xmm5			// 53, 43, 52, 42, 51, 41, 50, 40
			punpcklwd	xmm6, xmm7			// 73, 63, 72, 62, 71, 61, 70, 60
			movdqa		xmm1, xmm0
			movdqa		xmm5, xmm4
			punpckldq	xmm0, xmm2			// 31, 21, 11, 01, 30, 20, 10, 00
			punpckhdq	xmm1, xmm2			// 33, 23, 13, 03, 32, 22, 12, 02
			punpckldq	xmm4, xmm6			// 71, 61, 51, 41, 70, 60, 50, 40
			punpckhdq	xmm5, xmm6			// 73, 63, 53, 43, 72, 62, 52, 42
			movdqa		xmm2, xmm0
			movdqa		xmm3, xmm1
			punpcklqdq	xmm0, xmm4			// 70, 60, 50, 40, 30, 20, 10, 00
			punpckhqdq	xmm2, xmm4			// 71, 61, 51, 41, 31, 21, 11, 01
			punpcklqdq	xmm1, xmm5			// 72, 62, 52, 42, 32, 22, 12, 02
			punpckhqdq	xmm3, xmm5			// 73, 63, 53, 43, 33, 23, 13, 03
			movdqa		[edi], xmm0
			movdqa		[edi+16], xmm2
			movdqa		[edi+32], xmm1
			movdqa		[edi+48], xmm3
			add			esi, 8
			add			edx, 8
			add			edi, 64
			sub			ecx, 1
			jnz			shuffle_next4columns

			// wavelet transform
			mov			esi, work
			mov			edi, dstp1			// edi = dstp1
			mov			edx, dstp2			// edx = dstp2
			mov			ecx, hloop2			// ecx = hloop2
			add			esi, 64				// esi = work + 32

			movdqa		xmm2, [esi-32]
			movdqa		xmm0, [esi-16]
			movdqa		xmm1, [esi]
			paddw		xmm2, xmm1
			psraw		xmm2, 1
			psubw		xmm0, xmm2
			movdqa		[edx-16], xmm0

align 16
wavelet_next4columns:
			movdqa		xmm2, [esi+16]
			movdqa		xmm3, [esi+32]
			movdqa		xmm4, [esi+48]
			movdqa		xmm5, [esi+64]
			movdqa		xmm6, xmm1
			movdqa		xmm7, xmm3
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 1
			psraw		xmm3, 1
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			movdqa		[edx], xmm2
			movdqa		[edx+16], xmm4
			paddw		xmm0, xmm2
			paddw		xmm2, xmm4
			psraw		xmm0, 2
			psraw		xmm2, 2
			paddw		xmm6, xmm0
			paddw		xmm7, xmm2
			movdqa		[edi], xmm6
			movdqa		[edi+16], xmm7
			movdqa		xmm0, xmm4
			movdqa		xmm1, xmm5
			add			esi, 64
			add			edi, 32
			add			edx, 32
			sub			ecx, 1
			jnz			wavelet_next4columns

			// horizontal reflection
			mov			eax, width
			test		eax, 1
			jnz			no_reflection		// if (width % 2 == 0) {
			mov			edi, dstp1			//   edi = dstp1
			mov			edx, dstp2			//   edx = dstp2
			shl			eax, 3				//   eax = width / 2 * 16
			movdqa		xmm0, [edi+eax-16]
			movdqa		xmm1, [edx+eax-32]
			movdqa		[edi+eax], xmm0
			movdqa		[edx+eax], xmm1
no_reflection:								// }
		}
	}
}

void MosquitoNR::BlendCoef(int thread_id)
{
	const int y_start = ((height + 15) &~ 15) / 4 *  thread_id      / threads;
	const int y_end   = ((height + 15) &~ 15) / 4 * (thread_id + 1) / threads;
	if (y_start == y_end) return;
	const int pitch = this->pitch;
	const int multiplier = ((128 - restore) << 16) + restore;
	short *dstp = luma[0], *srcp = bufx[0];

	__asm
	{
		mov			edi, dstp
		mov			esi, srcp
		mov			eax, pitch
		mov			ebx, y_start
		mov			ecx, y_end
		add			eax, eax			// eax = pitch * sizeof(short)
		sub			ecx, ebx
		imul		ebx, eax			// ebx = y_start * pitch * sizeof(short)
		imul		ecx, eax
		add			edi, ebx			// edi = luma[0] + y_start * pitch
		add			esi, ebx			// esi = bufx[0] + y_start * pitch
		shr			ecx, 4				// ecx = (y_end - y_start) * pitch / 8
		movd		xmm6, multiplier
		pshufd		xmm6, xmm6, 0		// xmm6 = [128 - restore, restore] * 4
		mov			edx, 64
		movd		xmm7, edx
		pshufd		xmm7, xmm7, 0		// xmm7 = [64] * 4

align 16
next8pixels:
		movdqa		xmm0, [edi]			// d7, d6, d5, d4, d3, d2, d1, d0
		movdqa		xmm2, [esi]			// s7, s6, s5, s4, s3, s2, s1, s0
		movdqa		xmm1, xmm0
		punpcklwd	xmm0, xmm2			// s3, d3, s2, d2, s1, d1, s0, d0
		punpckhwd	xmm1, xmm2			// s7, d7, s6, d6, s5, d5, s4, d4
		pmaddwd		xmm0, xmm6
		pmaddwd		xmm1, xmm6
		paddd		xmm0, xmm7
		paddd		xmm1, xmm7
		psrad		xmm0, 7
		psrad		xmm1, 7
		packssdw	xmm0, xmm1
		movdqa		[edi], xmm0
		add			edi, 16
		add			esi, 16
		sub			ecx, 1
		jnz			next8pixels
	}
}

void MosquitoNR::InvWaveletHorz(int thread_id)
{
	const int y_start = (height + 15) / 16 *  thread_id      / threads * 8;
	const int y_end   = (height + 15) / 16 * (thread_id + 1) / threads * 8;
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

		__asm
		{
			// wavelet transform
			mov			esi, srcp1			// esi = srcp1
			mov			edx, srcp2			// edx = srcp2
			mov			edi, work			// edi = dstp
			mov			ecx, hloop			// ecx = hloop

			movdqa		xmm2, [edx-16]
			movdqa		xmm0, [esi]
			movdqa		xmm1, [edx]
			paddw		xmm2, xmm1
			psraw		xmm2, 2
			psubw		xmm0, xmm2
			movdqa		[edi], xmm0

align 16
wavelet_next4columns:
			movdqa		xmm2, [esi+16]
			movdqa		xmm3, [edx+16]
			movdqa		xmm4, [esi+32]
			movdqa		xmm5, [edx+32]
			movdqa		xmm6, xmm1
			movdqa		xmm7, xmm3
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 2
			psraw		xmm3, 2
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			movdqa		[edi+32], xmm2
			movdqa		[edi+64], xmm4
			paddw		xmm0, xmm2
			paddw		xmm2, xmm4
			psraw		xmm0, 1
			psraw		xmm2, 1
			paddw		xmm6, xmm0
			paddw		xmm7, xmm2
			movdqa		[edi+16], xmm6
			movdqa		[edi+48], xmm7
			movdqa		xmm0, xmm4
			movdqa		xmm1, xmm5
			add			esi, 32
			add			edx, 32
			add			edi, 64
			sub			ecx, 1
			jnz			wavelet_next4columns

			// shuffle
			mov			esi, work			// esi = work
			mov			edi, dstp			// edi = srcp
			mov			eax, pitch
			mov			ecx, hloop			// ecx = hloop
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3
			lea			edx, [edi+4*eax]	// edx = dstp + 4 * pitch

align 16
shuffle_next4columns:
			movdqa		xmm0, [esi]			// 70, 60, 50, 40, 30, 20, 10, 00
			movdqa		xmm1, [esi+16]		// 71, 61, 51, 41, 31, 21, 11, 01
			movdqa		xmm2, [esi+32]		// 72, 62, 52, 42, 32, 22, 12, 02
			movdqa		xmm3, [esi+48]		// 73, 63, 53, 43, 33, 23, 13, 03
			movdqa		xmm4, xmm0
			movdqa		xmm6, xmm2
			punpcklwd	xmm0, xmm1			// 31, 30, 21, 20, 11, 10, 01, 00
			punpckhwd	xmm4, xmm1			// 71, 70, 61, 60, 51, 50, 41, 40
			punpcklwd	xmm2, xmm3			// 33, 32, 23, 22, 13, 12, 03, 02
			punpckhwd	xmm6, xmm3			// 73, 72, 63, 62, 53, 52, 43, 42
			movdqa		xmm1, xmm0
			movdqa		xmm5, xmm4
			punpckldq	xmm0, xmm2			// 13, 12, 11, 10, 03, 02, 01, 00
			punpckhdq	xmm1, xmm2			// 33, 32, 31, 30, 23, 22, 21, 20
			punpckldq	xmm4, xmm6			// 53, 52, 51, 50, 43, 42, 41, 40
			punpckhdq	xmm5, xmm6			// 73, 72, 71, 70, 63, 62, 61, 60
			movq		qword ptr [edi], xmm0
			movq		qword ptr [edi+2*eax], xmm1
			movq		qword ptr [edx], xmm4
			movq		qword ptr [edx+2*eax], xmm5
			punpckhqdq	xmm0, xmm0
			punpckhqdq	xmm1, xmm1
			punpckhqdq	xmm4, xmm4
			punpckhqdq	xmm5, xmm5
			movq		qword ptr [edi+eax], xmm0
			movq		qword ptr [edi+ebx], xmm1
			movq		qword ptr [edx+eax], xmm4
			movq		qword ptr [edx+ebx], xmm5
			add			esi, 64
			add			edi, 8
			add			edx, 8
			sub			ecx, 1
			jnz			shuffle_next4columns
		}
	}

	// vertical reflection
	if (thread_id == threads - 1 && height % 2 == 0)
		memcpy(bufy[0] + height / 2 * pitch, bufy[0] + (height / 2 - 1) * pitch, pitch * sizeof(short));
}

void MosquitoNR::InvWaveletVert(int thread_id)
{
	const int y_start = (height + 7) / 8 *  thread_id      / threads * 8;
	const int y_end   = (height + 7) / 8 * (thread_id + 1) / threads * 8;
	if (y_start == y_end) return;
	const int pitch  = this->pitch;

	for (int y = y_start; y < y_end; y += 8)
	{
		int hloop = (width + 7) / 8;
		short* srcp1 = bufy[0] + y / 2 * pitch + 8;
		short* srcp2 = bufy[1] + y / 2 * pitch + 8;
		short* dstp = luma[1] + (y + 2) * pitch + 8;

		__asm
		{
			mov			esi, srcp1			// esi = srcp1
			mov			edx, srcp2			// edx = srcp2
			mov			edi, dstp			// edi = dstp
			mov			eax, pitch
			add			eax, eax			// eax = pitch * sizeof(short)
			lea			ebx, [eax+2*eax]	// ebx = pitch * sizeof(short) * 3
			lea			ecx, [eax+4*eax]	// ecx = pitch * sizeof(short) * 5

align 16
next8columns:
			push		edi
			movdqa		xmm2, [edx]
			movdqa		xmm0, [esi]
			movdqa		xmm1, [edx+eax]
			paddw		xmm2, xmm1
			psraw		xmm2, 2
			psubw		xmm0, xmm2
			movdqa		[edi], xmm0

			movdqa		xmm2, [esi+eax]
			movdqa		xmm3, [edx+2*eax]
			movdqa		xmm4, [esi+2*eax]
			movdqa		xmm5, [edx+ebx]
			movdqa		xmm6, xmm1
			movdqa		xmm7, xmm3
			paddw		xmm1, xmm3
			paddw		xmm3, xmm5
			psraw		xmm1, 2
			psraw		xmm3, 2
			psubw		xmm2, xmm1
			psubw		xmm4, xmm3
			movdqa		[edi+2*eax], xmm2
			movdqa		[edi+4*eax], xmm4
			paddw		xmm0, xmm2
			paddw		xmm2, xmm4
			psraw		xmm0, 1
			psraw		xmm2, 1
			paddw		xmm6, xmm0
			paddw		xmm7, xmm2
			movdqa		[edi+eax], xmm6
			movdqa		[edi+ebx], xmm7
			lea			edi, [edi+4*eax]

			movdqa		xmm0, [esi+ebx]
			movdqa		xmm1, [edx+4*eax]
			movdqa		xmm2, [esi+4*eax]
			movdqa		xmm3, [edx+ecx]
			movdqa		xmm6, xmm5
			movdqa		xmm7, xmm1
			paddw		xmm5, xmm1
			paddw		xmm1, xmm3
			psraw		xmm5, 2
			psraw		xmm1, 2
			psubw		xmm0, xmm5
			psubw		xmm2, xmm1
			movdqa		[edi+2*eax], xmm0
			paddw		xmm4, xmm0
			paddw		xmm0, xmm2
			psraw		xmm4, 1
			psraw		xmm0, 1
			paddw		xmm6, xmm4
			paddw		xmm7, xmm0
			movdqa		[edi+eax], xmm6
			movdqa		[edi+ebx], xmm7

			pop			edi
			add			esi, 16
			add			edx, 16
			add			edi, 16
			sub			hloop, 1
			jnz			next8columns
		}
	}
}
