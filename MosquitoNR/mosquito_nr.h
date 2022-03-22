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

#ifndef MOSQUITO_NR_H_
#define MOSQUITO_NR_H_

#include <Windows.h>
#include <process.h>
#include "avisynth.h"

class MosquitoNR;

typedef void (MosquitoNR::*MTFunc)(int thread_id);

const int MAX_THREADS = 32;

struct ThreadInfo
{
	int thread_id;
	bool close;
	MosquitoNR* inst;
	MTFunc mt_func;
	HANDLE job_start, job_finished;
};

class MTInfo
{
private:
	int threads;
	ThreadInfo th[MAX_THREADS];
	HANDLE running[MAX_THREADS];

public:
	MTInfo();
	~MTInfo();
	bool CreateThreads(int _threads, MosquitoNR* inst);
	void ExecMTFunc(MTFunc mt_func);
};

class MosquitoNR : public GenericVideoFilter
{
private:
	const int strength, restore, radius;
	int threads;
	const int width, height;
	const int pitch;			// pitch of following buffers
	short* luma[2];				// original/blurred luma data
	short* bufy[2];				// vertical approximation/detail coefficients
	short* bufx[2];				// shuffled horizontal approximation/detail coefficients of vertical approximation coefficients
	short* work[MAX_THREADS];	// temporal buffer
	bool ssse3;
	MTInfo mt;
	PVideoFrame src, dst;

	void InitBuffer();
	bool AllocBuffer();
	void FreeBuffer();
	void SmoothingSSE2(int thread_id);
	void SmoothingSSSE3(int thread_id);

public:
	MosquitoNR(PClip _child, int _strength, int _restore, int _radius, int _threads, IScriptEnvironment* env);
	~MosquitoNR();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

	void CopyLumaFrom();
	void CopyLumaTo();
	void Smoothing(int thread_id);
	void WaveletVert1(int thread_id);
	void WaveletHorz1(int thread_id);
	void WaveletVert2(int thread_id);
	void WaveletHorz2(int thread_id);
	void WaveletHorz3(int thread_id);
	void BlendCoef(int thread_id);
	void InvWaveletHorz(int thread_id);
	void InvWaveletVert(int thread_id);
};

#endif	// MOSQUITO_NR_H_
