#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<omp.h>
#include<nmmintrin.h>
#include<malloc.h>
#include"Windows.h"
#define ImplN 400	//³¬²ÎÊý£ºÒþº¬²ã 
#define THREADS_NUM 8
#define LEARN_RATE 0.1 //³¬²ÎÊý£ºÑ§Ï°ÂÊ 
//4 Layers
#define STEP 4
#pragma pack(8)
float Layer12[ImplN * 28 * 28], bia2[ImplN];
float Layer23[ImplN * ImplN], bia3[ImplN];
float Layer34[ImplN * 10], bia4[10];
float random01()
{
	return (rand() % 500 / 250.0) - 1;
}
void randW()
{
	for (int i = 0; i < ImplN * 28 * 28; i++)Layer12[i] = random01();
	for (int i = 0; i < ImplN * ImplN; i++)Layer23[i] = random01();
	for (int i = 0; i < ImplN * 10; i++)Layer34[i] = random01();
	for (int i = 0; i < ImplN; i++)bia2[i] = random01();
	for (int i = 0; i < ImplN; i++)bia3[i] = random01();
	for (int i = 0; i < 10; i++)bia4[i] = random01();
}
void Forward784ToImplN(float* input, float* output, float* weight, float* bias)
{
#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)
	{
		__m128 resTmp = _mm_setzero_ps();
		for (int j = 0; j < 784; j += STEP)
		{
			__m128 inp = _mm_load_ps(input + j);
			__m128 wei = _mm_load_ps(weight + i * 784 + j);
			inp = _mm_mul_ps(inp, wei);
			resTmp = _mm_add_ps(resTmp, inp);
			//output[i]+=input[j]*weight[i*784+j];
		}
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		_mm_store_ss(output + i, resTmp);
		output[i] += bias[i];
		output[i] = 1 / (1 + exp(-output[i]));
	}
}
void ForwardImplNToImplN(float* input, float* output, float* weight, float* bias)
{
#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)
	{
		__m128 resTmp = _mm_setzero_ps();
		for (int j = 0; j < ImplN; j += STEP)
		{
			__m128 inp = _mm_load_ps(input + j);
			__m128 wei = _mm_load_ps(weight + i * ImplN + j);
			inp = _mm_mul_ps(inp, wei);
			resTmp = _mm_add_ps(resTmp, inp);
			//output[i]+=input[j]*weight[i*ImplN+j];
		}
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		_mm_store_ss(output + i, resTmp);
		output[i] += bias[i];
		output[i] = 1 / (1 + exp(-output[i]));
	}
}
void ForwardImplNToFinal(float* input, float* output, float* weight, float* bias)
{
	float softmaxSum = 0;
#pragma omp parallel for reduction(+:softmaxSum) num_threads(THREADS_NUM)
	for (int i = 0; i < 10; i++)
	{
		__m128 resTmp = _mm_setzero_ps();
		for (int j = 0; j < ImplN; j += STEP)
		{
			__m128 inp = _mm_load_ps(input + j);
			__m128 wei = _mm_load_ps(weight + i * ImplN + j);
			inp = _mm_mul_ps(inp, wei);
			resTmp = _mm_add_ps(resTmp, inp);
			//output[i]+=input[j]*weight[i*ImplN+j];
		}
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		_mm_store_ss(output + i, resTmp);
		output[i] += bias[i];
		output[i] = exp(output[i]);
		softmaxSum += output[i];
	}
	//#pragma omp //´Ë´¦Ó¦¸ÃSIMD¸ü¼Ó»®µÃÀ´ 
	__m128 divSum = _mm_set1_ps(softmaxSum);
	__m128 out;
	for (int i = 0; i < STEP * (10 / STEP); i += STEP)
	{
		out = _mm_load_ps(output + i);
		out = _mm_div_ps(out, divSum);
		_mm_store_ps(output + i, out);
		//output[i]/=softmaxSum;
	}
	for (int i = STEP * (10 / STEP); i < 10; i++)
	{
		output[i] /= softmaxSum;
	}
}
void Forward(float* BatchData, float*& Impl1, float*& Impl2, float*& predY)//row:28*28,col:batchNum  
//returns PredictResult
//²»»áÊÍ·ÅBatchData 
//Impl1¡¢Impl2¡¢returnÖ¸Õë×ÔÐÐÊÍ·Å 
{
	//Xºó=w * XÇ° 
	float* preLayer = BatchData;
	float* nextLayer = (float*)_mm_malloc(ImplN * sizeof(float), 8);//¿ª±Ù2²ã 
	Forward784ToImplN(preLayer, nextLayer, Layer12, bia2);
	Impl1 = preLayer = nextLayer;//ÏÂÒ»²ã £»preLayerÖ¸Ïò2²ã 
	nextLayer = (float*)_mm_malloc(ImplN * sizeof(float), 8);//nextLayer£º¿ª±Ù3²ã 
	ForwardImplNToImplN(preLayer, nextLayer, Layer23, bia3);
	Impl2 = preLayer = nextLayer;//ÏÂÒ»²ã£»preLayerÖ¸ÏòµÚ3²ã
	nextLayer = (float*)_mm_malloc(10 * sizeof(float), 8);//nextLayer£º¿ª±Ù4²ã 
	ForwardImplNToFinal(preLayer, nextLayer, Layer34, bia4);
	predY = nextLayer;//ÐèÒªÔÚÍâÃæ×ÔÐÐÊÍ·Å 
}
float OneHotCrossEntropy(float* Result, int* RealResult)
{
	//float bat = batchNum;
	float loss = 0;
	//for (int i = 0; i < batchNum; i++)
		loss -= log(Result[RealResult[0]]);
		return loss;
		// bat;
}
void backward(float* BatchData, float* Impl1, float* Impl2, float* yOut, float* yTag, float learningRate)//ÄæÌÝ¶È¸ÄÈ¨ 
{
	float* DeltaAft = (float*)_mm_malloc(10 * sizeof(float), 8);//4²ãµÄdL/dz 
	//ËãdL/dz Layer4 

	//²»¿¼ÂÇbatchNum 
	for (int i = 0; i < 10; i++)
		DeltaAft[i] = yOut[i] - yTag[i];


	float* DeltaBef = (float*)_mm_malloc(ImplN * sizeof(float), 8);//3²ãµÄ 
	//ÏÈËã3²ãDelta 
#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)
	{
		__m128 resTmp = _mm_setzero_ps();
		//DeltaBef[i] = 0;
		for (int j = 0; j < STEP * (10 / STEP); j += STEP)
		{
			__m128 atmp = _mm_load_ps(DeltaAft + j);
			__m128 btmp = _mm_set_ps(Layer34[(j + 3) * ImplN + i], Layer34[(j + 2) * ImplN + i], Layer34[(j + 1) * ImplN + i], Layer34[j * ImplN + i]);
			btmp = _mm_mul_ps(atmp, btmp);
			resTmp = _mm_add_ps(resTmp, btmp);
			//DeltaBef[i] += (DeltaAft[j] * Layer34[j * ImplN + i]);
		}
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		_mm_store_ss(DeltaBef + i, resTmp);
		for (int j = STEP * (10 / STEP); j < 10; j++)
		{
			DeltaBef[i] += (DeltaAft[j] * Layer34[j * ImplN + i]);
		}
		DeltaBef[i] *= ((Impl2[i]) * (1 - Impl2[i]));//BefÎª3²ã 
	}

	//dz/dwij=a,dz/db=1 ºó³ËÇ° 
	//¸ÄLayer34 
#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < 10; i++)
	{
		__m128 mulRate = _mm_set1_ps(learningRate * DeltaAft[i]);
		for (int j = 0; j < ImplN; j += STEP)
		{
			__m128 tmp = _mm_load_ps(Impl2 + j);
			tmp = _mm_mul_ps(mulRate, tmp);
			__m128 Lay = _mm_load_ps(Layer34 + i * ImplN + j);
			Lay = _mm_sub_ps(Lay, tmp);
			_mm_store_ps(Layer34 + i * ImplN + j, Lay);
			//Learning rate?
			//Layer34[i * ImplN + j] -= learningRate *DeltaAft[i] * Impl2[j];
		}
		bia4[i] -= learningRate * DeltaAft[i];
	}
	_mm_free((void*)DeltaAft);
	//delete[]DeltaAft;//Delete 4²ãDelta 
	DeltaAft = DeltaBef;//3²ãDeltaÎªºóÃæ
	DeltaBef = (float*)_mm_malloc(ImplN * sizeof(float), 8);//Ëã2²ãDelta 

#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)
	{
		__m128 resTmp = _mm_setzero_ps();
		//DeltaBef[i] = 0;
		for (int j = 0; j < ImplN; j += STEP)
		{
			//DeltaBef[i] += (DeltaAft[j] * Layer23[j * ImplN + i]);
			__m128 atmp = _mm_load_ps(DeltaAft + j);
			__m128 btmp = _mm_set_ps(Layer23[(j + 3) * ImplN + i], Layer23[(j + 2) * ImplN + i], Layer23[(j + 1) * ImplN + i], Layer23[j * ImplN + i]);
			btmp = _mm_mul_ps(atmp, btmp);
			resTmp = _mm_add_ps(resTmp, btmp);
		}
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		resTmp = _mm_hadd_ps(resTmp, resTmp);
		_mm_store_ss(DeltaBef + i, resTmp);
		DeltaBef[i] *= ((Impl1[i]) * (1 - Impl1[i]));
	}

#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)//¸ÄLayer23
	{
		__m128 mulRate = _mm_set1_ps(learningRate * DeltaAft[i]);
		for (int j = 0; j < ImplN; j += STEP)
		{
			__m128 tmp = _mm_load_ps(Impl1 + j);
			tmp = _mm_mul_ps(mulRate, tmp);
			__m128 Lay = _mm_load_ps(Layer23 + i * ImplN + j);
			Lay = _mm_sub_ps(Lay, tmp);
			_mm_store_ps(Layer23 + i * ImplN + j, Lay);
			//Layer23[i * ImplN + j] -= learningRate*DeltaAft[i] * Impl1[j];
		}
		bia3[i] -= learningRate * DeltaAft[i];
	}
	_mm_free(DeltaAft);
	//delete[]DeltaAft;//Delete 3²ãDelta 
	DeltaAft = DeltaBef;//2²ãDeltaÎªDeltaAft 

#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)//¸ÄLayer12 
	{
		__m128 mulRate = _mm_set1_ps(learningRate * DeltaAft[i]);
		for (int j = 0; j < 784; j += STEP)
		{
			__m128 tmp = _mm_load_ps(BatchData + j);
			tmp = _mm_mul_ps(mulRate, tmp);
			__m128 Lay = _mm_load_ps(Layer12 + i * 784 + j);
			Lay = _mm_sub_ps(Lay, tmp);
			_mm_store_ps(Layer12 + i * 784 + j, Lay);
			//Layer12[i * 784 + j] -= learningRate*DeltaAft[i] * BatchData[j];
		}
		bia2[i] -= learningRate * DeltaAft[i];
	}

	_mm_free((void*)DeltaAft);
	//delete[]DeltaAft;
}
FILE* fx;
FILE* fy;
float cumLoss = 0;
int rightCnt = 0;
void put_one(int idx
	, float& forwardTime, float& backwardTime
)
{
	unsigned char img[784], tag;
	fread(img, 1, 784, fx);
	fread(&tag, 1, 1, fy);
	float* X = (float*)_mm_malloc(784 * sizeof(float), 8);
	float* Y = (float*)_mm_malloc(10 * sizeof(float), 8);
	for (int i = 0; i < 784; i++)X[i] = img[i] / 255.0;
	for (int i = 0; i < 10; i++)Y[i] = (tag == i);
	float* Imp1;
	float* Imp2;
	float* predY;
	long long freq, head, tail;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Forward(X, Imp1, Imp2, predY);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//_mm_free(predY);
	forwardTime = 1000.0 * (tail - head) / freq;
	int guess = 0;
	float predPoss = predY[0];
	for (int i = 1; i < 10; i++)
	{
		if (predY[i] > predPoss)
		{
			predPoss = predY[i];
			guess = i;
		}
	}
	rightCnt += (guess == tag);
	int temp = tag;
	float loss = OneHotCrossEntropy(predY, &temp);
	cumLoss += loss;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	backward(X, Imp1, Imp2, predY, Y, LEARN_RATE);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	backwardTime = 1000.0 * (tail - head) / freq;
	_mm_free(Imp1);
	_mm_free(Imp2);
	_mm_free(X);
	_mm_free(Y);
	_mm_free(predY);
	if (idx % 1000 == 0)
	{
		printf("idx=%d Avg loss=%.3f\n", idx, cumLoss / 1000);
		printf("Acc=%.2f\n", rightCnt / 1000.0);
		cumLoss = 0;
		rightCnt = 0;
	}
}
void testOne(int idx)
{
	unsigned char img[784], tag;
	fread(img, 1, 784, fx);
	fread(&tag, 1, 1, fy);
	float* X = (float*)_mm_malloc(784 * sizeof(float), 8);
	float* Y = (float*)_mm_malloc(10 * sizeof(float), 8);
	for (int i = 0; i < 784; i++)X[i] = img[i] / 255.0;
	for (int i = 0; i < 10; i++)Y[i] = (tag == i);
	float* Imp1;
	float* Imp2;
	float* predY;
	Forward(X, Imp1, Imp2, predY);
	int guess = 0;
	float predPoss = predY[0];
	for (int i = 1; i < 10; i++)
	{
		if (predY[i] > predPoss)
		{
			predPoss = predY[i];
			guess = i;
		}
	}
	_mm_free(Imp1);
	_mm_free(Imp2);
	_mm_free(X);
	_mm_free(Y);
	_mm_free(predY);
	if (guess == tag)
		rightCnt++;
}
int main()
{
	//omp_set_num_threads(4);
	//printf("%d\n", omp_get_num_threads());
	srand((unsigned)time(NULL));
	randW();
	for (int epo = 0; epo < 1; epo++)
	{
		fx = fopen("train-images.idx3-ubyte", "rb");
		fseek(fx, 16, SEEK_SET);
		fy = fopen("train-labels.idx1-ubyte", "rb");
		fseek(fy, 8, SEEK_SET);
		float forwardTotalTime = 0.0, backwardTotalTime = 0.0;
		for (int i = 1; i <= 60000; i++)
		{
			float oneForwardTime, oneBackwardTime;
			put_one(i
				, oneForwardTime, oneBackwardTime
			);
			forwardTotalTime += oneForwardTime;
			backwardTotalTime += oneBackwardTime;
		}
		printf("forward avg elapsed %.3f ms\nbackward avg elapsed %.3f ms", forwardTotalTime / 60000, backwardTotalTime / 60000);
		printf("\n\n");
		fclose(fx);
		fclose(fy);
	}
	fx = fopen("t10k-images.idx3-ubyte", "rb");
	fseek(fx, 16, SEEK_SET);
	fy = fopen("t10k-labels.idx1-ubyte", "rb");
	fseek(fy, 8, SEEK_SET);
	for (int i = 1; i <= 10000; i++)
		testOne(i);
	printf("%.2f\n", rightCnt / 10000.0);
	fclose(fx);
	fclose(fy);
	return 0;
}