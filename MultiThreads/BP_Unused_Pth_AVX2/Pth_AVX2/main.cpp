#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cfloat>
#include<cstdlib>
#include<pthread.h>
//#include<omp.h>
#include<immintrin.h>
#include"Windows.h"
#pragma comment(lib, "pthreadVC2.lib")
#define ImplN 600	//超参数：隐含层 
#define learningRate 0.1 //超参数：学习率 
#define NUM_THREADS 16
#define STEP 8
#pragma pack(8)
//4 Layers
float Layer12[ImplN * 28 * 28], bia2[ImplN];
float Layer23[ImplN * ImplN], bia3[ImplN];
float Layer34[ImplN * 10], bia4[10];
pthread_barrier_t barrier;//barrier
pthread_mutex_t softSumMutex;
float random01() { return (rand() % 500 / 500.0) - 0.5; }
void randW()
{
	for (int i = 0; i < ImplN * 28 * 28; i++)Layer12[i] = random01();
	for (int i = 0; i < ImplN * ImplN; i++)Layer23[i] = random01();
	for (int i = 0; i < ImplN * 10; i++)Layer34[i] = random01();
	for (int i = 0; i < ImplN; i++)bia2[i] = random01();
	for (int i = 0; i < ImplN; i++)bia3[i] = random01();
	for (int i = 0; i < 10; i++)bia4[i] = random01();
}
float BatchDataX[784], Impl1[ImplN], Impl2[ImplN], predY[10], softmax_sum;
float Delta4[10], Delta3[ImplN], Delta2[ImplN];
int BatchDataY;
void getLoHiIdx(int thread_no, int thread_cnt, int dim_sz, int& lo, int& hi)
{
	lo = (thread_no * dim_sz) / thread_cnt;
	hi = ((thread_no + 1) * dim_sz) / thread_cnt;
}
enum { WAITING, FORWARD, BACKWARD, EXIT }state;
void* one_thread_parallel_forward(void* rank)
{
	int thread_num = NUM_THREADS;
	int thread_no = (int)rank;
	float* output;
	float* input;
	float* bias;
	float* weight;
	float softmax_localSum;
	while (1)
	{
		while (state == WAITING);
		if (state == EXIT)break;
		if (state == BACKWARD)goto goto_backward;
		//wait Forward Begin

	goto_forward:
		output = Impl1;
		input = BatchDataX;
		bias = bia2;
		weight = Layer12;
		//calculate upperidx and loweridx
		int lo, hi;
		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)
		{
			__m256 resTmp = _mm256_setzero_ps();
			__m128 s1, s2;
			for (int j = 0; j < 784; j += STEP)
			{
				__m256 inp = _mm256_load_ps(input + j);
				__m256 wei = _mm256_load_ps(weight + i * 784 + j);
				inp = _mm256_mul_ps(inp, wei);
				resTmp = _mm256_add_ps(resTmp, inp);
				//output[i]+=input[j]*weight[i*784+j];
			}
			s1 = _mm256_extractf128_ps(resTmp, 0);  // s1=[a0,a1,a2,a3]
			s2 = _mm256_extractf128_ps(resTmp, 1);	// s2=[a4,a5,a6,a7]
			s1 = _mm_hadd_ps(s1, s2); // s1=[a0+a1,a2+a3,a4+a5,a6+a7]
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3,a4+a5+a6+a7,a0+a1+a2+a3,a4+a5+a6+a7]
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3+a4+a5+a6+a7,...]
			_mm_store_ss(output + i, s1);
			output[i] += bias[i];
			output[i] = 1 / (1 + exp(-output[i]));
		}
		//barrier
		pthread_barrier_wait(&barrier);
		//calculate upper & lower
		output = Impl2;
		input = Impl1;
		bias = bia3;
		weight = Layer23;
		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)
		{
			__m256 resTmp = _mm256_setzero_ps();
			__m128 s1, s2;
			for (int j = 0; j < ImplN; j += STEP)
			{
				__m256 inp = _mm256_load_ps(input + j);
				__m256 wei = _mm256_load_ps(weight + i * ImplN + j);
				inp = _mm256_mul_ps(inp, wei);
				resTmp = _mm256_add_ps(resTmp, inp);
				//output[i]+=input[j]*weight[i*ImplN+j];
			}
			s1 = _mm256_extractf128_ps(resTmp, 0);  // s1=[a0,a1,a2,a3]
			s2 = _mm256_extractf128_ps(resTmp, 1);	// s2=[a4,a5,a6,a7]
			s1 = _mm_hadd_ps(s1, s2); // s1=[a0+a1,a2+a3,a4+a5,a6+a7]
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3,a4+a5+a6+a7,a0+a1+a2+a3,a4+a5+a6+a7]
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3+a4+a5+a6+a7,...]
			_mm_store_ss(output + i, s1);
			output[i] += bias[i];
			output[i] = 1 / (1 + exp(-output[i]));
		}
		//barrier
		pthread_barrier_wait(&barrier);
		//calculate upper&lower
		softmax_localSum = 0;
		output = predY;
		input = Impl2;
		bias = bia4;
		weight = Layer34;
		getLoHiIdx(thread_no, thread_num, 10, lo, hi);
		for (int i = lo; i < hi; i++)
		{
			__m256 resTmp = _mm256_setzero_ps();
			__m128 s1, s2;
			for (int j = 0; j < ImplN; j += STEP)
			{
				__m256 inp = _mm256_load_ps(input + j);
				__m256 wei = _mm256_load_ps(weight + i * ImplN + j);
				inp = _mm256_mul_ps(inp, wei);
				resTmp = _mm256_add_ps(resTmp, inp);
				//output[i]+=input[j]*weight[i*ImplN+j];
			}
			s1 = _mm256_extractf128_ps(resTmp, 0);  // s1=[a0,a1,a2,a3]
			s2 = _mm256_extractf128_ps(resTmp, 1);	// s2=[a4,a5,a6,a7]
			s1 = _mm_hadd_ps(s1, s2); // s1=[a0+a1,a2+a3,a4+a5,a6+a7]
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3,a4+a5+a6+a7,a0+a1+a2+a3,a4+a5+a6+a7]
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3+a4+a5+a6+a7,...]
			_mm_store_ss(output + i, s1);
			output[i] += bias[i];
			output[i] = exp(output[i]);
			softmax_localSum += output[i];
		}
		//barrier
		//pthread_barrier_wait(&barrier);
		pthread_mutex_lock(&softSumMutex);
		softmax_sum += softmax_localSum;
		pthread_mutex_unlock(&softSumMutex);
		pthread_barrier_wait(&barrier);

		//parallel for
		getLoHiIdx(thread_no, thread_num, 10, lo, hi);
		for (int i = lo; i < hi; i++)
		{
			predY[i] /= softmax_sum;
		}
		pthread_barrier_wait(&barrier);
		if (thread_no == 0)state = WAITING;
		pthread_barrier_wait(&barrier);

		//wait backward begin
		while (state == WAITING);
		if (state == EXIT)break;
		if (state == FORWARD)goto goto_forward;

	goto_backward:
		getLoHiIdx(thread_no, thread_num, 10, lo, hi);
		for (int i = lo; i < hi; i++)//i: 0-9
		{
			Delta4[i] = predY[i] - (i == BatchDataY);
		}
		pthread_barrier_wait(&barrier);

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)//i: 0- ImplN-1
		{
			__m128 resTmp = _mm_setzero_ps();
			//Delta3[i] = 0;
			for (int j = 0; j < STEP * (10 / STEP); j += STEP)//j: 0-9
			{
				__m128 atmp = _mm_load_ps(Delta4 + j);
				__m128 btmp = _mm_set_ps(Layer34[(j + 3) * ImplN + i], Layer34[(j + 2) * ImplN + i], Layer34[(j + 1) * ImplN + i], Layer34[j * ImplN + i]);
				btmp = _mm_mul_ps(atmp, btmp);
				resTmp = _mm_add_ps(resTmp, btmp);
				//Delta3[i] += (Delta4[j] * Layer34[j * ImplN + i]);
			}
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			_mm_store_ss(Delta3 + i, resTmp);
			for (int j = STEP * (10 / STEP); j < 10; j++)
			{
				Delta3[i] += (Delta4[j] * Layer34[j * ImplN + i]);
			}
			Delta3[i] *= ((Impl2[i]) * (1 - Impl2[i]));
		}

		getLoHiIdx(thread_no, thread_num, 10, lo, hi);
		for (int i = lo; i < hi; i++)//i: 10
		{
			__m128 mulRate = _mm_set1_ps(learningRate * Delta4[i]);
			for (int j = 0; j < ImplN; j += STEP)//j: ImplN
			{
				__m128 tmp = _mm_load_ps(Impl2 + j);
				tmp = _mm_mul_ps(mulRate, tmp);
				__m128 Lay = _mm_load_ps(Layer34 + i * ImplN + j);
				Lay = _mm_sub_ps(Lay, tmp);
				_mm_store_ps(Layer34 + i * ImplN + j, Lay);
				//Layer34[i * ImplN + j] -= learningRate * Delta4[i] * Impl2[j];
			}
			bia4[i] -= learningRate * Delta4[i];
		}
		pthread_barrier_wait(&barrier);

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)//i: ImplN
		{
			__m128 resTmp = _mm_setzero_ps();
			//Delta2[i] = 0;
			for (int j = 0; j < ImplN; j += STEP)//j: ImplN
			{
				__m128 atmp = _mm_load_ps(Delta3 + j);
				__m128 btmp = _mm_set_ps(Layer23[(j + 3) * ImplN + i], Layer23[(j + 2) * ImplN + i], Layer23[(j + 1) * ImplN + i], Layer23[j * ImplN + i]);
				btmp = _mm_mul_ps(atmp, btmp);
				resTmp = _mm_add_ps(resTmp, btmp);
				//Delta2[i] += (Delta3[j] * Layer23[j * ImplN + i]);
			}
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			_mm_store_ss(Delta2 + i, resTmp);
			Delta2[i] *= ((Impl1[i]) * (1 - Impl1[i]));
		}

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)//i: ImplN
		{
			__m128 mulRate = _mm_set1_ps(learningRate * Delta3[i]);
			for (int j = 0; j < ImplN; j += STEP)//j: ImplN
			{
				__m128 tmp = _mm_load_ps(Impl1 + j);
				tmp = _mm_mul_ps(mulRate, tmp);
				__m128 Lay = _mm_load_ps(Layer23 + i * ImplN + j);
				Lay = _mm_sub_ps(Lay, tmp);
				_mm_store_ps(Layer23 + i * ImplN + j, Lay);
				//Layer23[i * ImplN + j] -= learningRate * Delta3[i] * Impl1[j];
			}
			bia3[i] -= learningRate * Delta3[i];
		}

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)//i: ImplN
		{
			__m128 mulRate = _mm_set1_ps(learningRate * Delta2[i]);
			for (int j = 0; j < 784; j += STEP)//j: 784
			{
				__m128 tmp = _mm_load_ps(BatchDataX + j);
				tmp = _mm_mul_ps(mulRate, tmp);
				__m128 Lay = _mm_load_ps(Layer12 + i * 784 + j);
				Lay = _mm_sub_ps(Lay, tmp);
				_mm_store_ps(Layer12 + i * 784 + j, Lay);
				//Layer12[i * 784 + j] -= learningRate * Delta2[i] * BatchDataX[j];
			}
			bia2[i] -= learningRate * Delta2[i];
		}
		pthread_barrier_wait(&barrier);
		if (thread_no == 0)state = WAITING;
		pthread_barrier_wait(&barrier);
	}
	return NULL;
}
float OneHotCrossEntropy(float* Result, int RealResult)
{
	return -log(Result[RealResult]);
}
FILE* fx;
FILE* fy;
float cumLoss = 0;
int rightCnt = 0;
void ReadOneData()
{
	unsigned char img[784], tag;
	fread(img, 1, 784, fx);
	fread(&tag, 1, 1, fy);
	for (int i = 0; i < 784; i++)BatchDataX[i] = img[i] / 255.0;
	softmax_sum = 0;
	BatchDataY = tag;
}
void put_one(int idx
	, float& forwardTime, float& backwardTime
)
{
	ReadOneData();

	long long freq, head, tail;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	state = FORWARD;//wake forward
	while (state != WAITING);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);

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
	rightCnt += (guess == BatchDataY);
	float loss = OneHotCrossEntropy(predY, BatchDataY);
	cumLoss += loss;

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	state = BACKWARD;//wake backward
	while (state != WAITING);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);

	backwardTime = 1000.0 * (tail - head) / freq;
	if (idx % 1000 == 0)
	{
		printf("idx=%d Avg loss=%.3f\n", idx, cumLoss / 1000.0);
		printf("Acc=%.2f\n", rightCnt / 1000.0);
		cumLoss = 0;
		rightCnt = 0;
	}
}
void testOne(int idx)
{
	ReadOneData();
	state = FORWARD;
	while (state != WAITING);
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
	if (guess == BatchDataY)
		rightCnt++;
}
int main()
{
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);
	pthread_mutex_init(&softSumMutex, NULL);
	state = WAITING;
	pthread_t threads[NUM_THREADS];
	randW();
	for (int i = 0; i < NUM_THREADS; i++)
		pthread_create(threads + i, NULL, one_thread_parallel_forward, (void*)i);
	srand((unsigned)time(NULL));

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
	void* ret;
	state = EXIT;
	for (int i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], &ret);
	pthread_barrier_destroy(&barrier);
	pthread_mutex_destroy(&softSumMutex);
	return 0;
}