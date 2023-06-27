#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cfloat>
#include<cstdlib>
#include<pthread.h>
//#include<omp.h>
#include<nmmintrin.h>
#include"Windows.h"
#include<semaphore.h>
#pragma comment(lib, "pthreadVC2.lib")
#define ImplN 400	//超参数：隐含层 
#define learningRate 0.1 //超参数：学习率 
#define NUM_THREADS 16
#define STEP 4
#pragma pack(8)
//4 Layers
float Layer12[ImplN * 28 * 28], bia2[ImplN];
float Layer23[ImplN * ImplN], bia3[ImplN];
float Layer34[ImplN * 10], bia4[10];
pthread_barrier_t barrier;//barrier
pthread_mutex_t softSumMutex;
//pthread_mutex_t condMutex;
//pthread_cond_t cond,mainCond;
sem_t beginSem,endSem;
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
//inline void getLoHiIdx(int thread_no, int thread_cnt, int dim_sz, int& lo, int& hi)
//{
//	lo = (thread_no * dim_sz) / thread_cnt;
//	hi = ((thread_no + 1) * dim_sz) / thread_cnt;
//}
#define getLoHiIdx(thread_no,thread_cnt,dim_sz,lo,hi) lo = (thread_no * dim_sz) / thread_cnt;hi = ((thread_no + 1) * dim_sz) / thread_cnt;
enum { FORWARD, BACKWARD, EXIT }state;
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
		//pthread_mutex_lock(&condMutex);
		//pthread_cond_wait(&cond,&condMutex);
		//pthread_mutex_unlock(&condMutex);
		sem_wait(&beginSem);
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
			__m128 resTmp = _mm_setzero_ps();
			//output[i] = 0;
			for (int j = 0; j < 784;)
			{
				__m128 atmp = _mm_load_ps(input + j);
				__m128 btmp = _mm_load_ps(weight + i * 784 + j);
				__m128 ctmp = _mm_mul_ps(atmp, btmp);
				j += STEP;
				resTmp = _mm_add_ps(resTmp, ctmp);
				//output[i] += input[j] * weight[i * 784 + j];
			}
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			_mm_store_ss(output + i, resTmp);
			output[i] += bias[i];
			output[i] = 1 / (1 + exp(-output[i]));
		}
		//barrier
		//calculate upper & lower
		output = Impl2;
		input = Impl1;
		bias = bia3;
		weight = Layer23;
		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		pthread_barrier_wait(&barrier);
		for (int i = lo; i < hi; i++)
		{
			__m128 resTmp = _mm_setzero_ps();
			//output[i] = 0;
			for (int j = 0; j < ImplN;)
			{
				__m128 atmp = _mm_load_ps(input + j);
				__m128 btmp = _mm_load_ps(weight + i * ImplN + j);
				__m128 ctmp = _mm_mul_ps(atmp, btmp);
				j += STEP;
				resTmp = _mm_add_ps(resTmp, ctmp);
				//output[i] += input[j] * weight[i * ImplN + j];
			}
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			_mm_store_ss(output + i, resTmp);
			output[i] += bias[i];
			output[i] = 1 / (1 + exp(-output[i]));
		}
		//barrier
		//calculate upper&lower
		softmax_localSum = 0;
		output = predY;
		input = Impl2;
		bias = bia4;
		weight = Layer34;
		getLoHiIdx(thread_no, thread_num, 10, lo, hi);
		pthread_barrier_wait(&barrier);
		for (int i = lo; i < hi; i++)
		{
			__m128 resTmp = _mm_setzero_ps();
			//output[i] = 0;
			for (int j = 0; j < ImplN;)
			{
				__m128 atmp = _mm_load_ps(input + j);
				__m128 btmp = _mm_load_ps(weight + i * ImplN + j);
				__m128 ctmp = _mm_mul_ps(atmp, btmp);
				j += STEP;
				resTmp = _mm_add_ps(resTmp, ctmp);
				//output[i] += input[j] * weight[i * ImplN + j];
			}
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			resTmp = _mm_hadd_ps(resTmp, resTmp);
			_mm_store_ss(output + i, resTmp);
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
		if (thread_no == 0)
		{
			//__m128 divSum = _mm_set1_ps(softmax_sum);
			//__m128 out;
			//out = _mm_load_ps(output);
			//out = _mm_div_ps(out, divSum);
			//_mm_store_ps(output, out);
			//out = _mm_load_ps(output + 4);
			//out = _mm_div_ps(out, divSum);
			//_mm_store_ps(output + 4, out);
			//output[8] /= softmax_sum;
			//output[9] /= softmax_sum;
			sem_post(&endSem);
			//pthread_cond_broadcast(&mainCond);
		}
		//pthread_barrier_wait(&barrier);

		//wait backward begin
		sem_wait(&beginSem);
		//pthread_mutex_lock(&condMutex);
		//pthread_cond_wait(&cond, &condMutex);
		//pthread_mutex_unlock(&condMutex);
		if (state == EXIT)break;
		if (state == FORWARD)goto goto_forward;

	goto_backward:
		getLoHiIdx(thread_no, thread_num, 10, lo, hi);
		for (int i = lo; i < hi; i++)//i: 0-9
			Delta4[i] = predY[i] - (i == BatchDataY);

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		pthread_barrier_wait(&barrier);
		for (int i = lo; i < hi; i++)//i: 0- ImplN-1
		{
			__m128 resTmp = _mm_setzero_ps();
			//Delta3[i] = 0;
			for (int j = 0; j < STEP * (10 / STEP); )//j: 0-9
			{
				__m128 atmp = _mm_load_ps(Delta4 + j);
				__m128 btmp = _mm_set_ps(Layer34[(j + 3) * ImplN + i], Layer34[(j + 2) * ImplN + i], Layer34[(j + 1) * ImplN + i], Layer34[j * ImplN + i]);
				__m128 ctmp = _mm_mul_ps(atmp, btmp);
				j += STEP;
				resTmp = _mm_add_ps(resTmp, ctmp);
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
			for (int j = 0; j < ImplN; )//j: ImplN
			{
				__m128 tmp = _mm_load_ps(Impl2 + j);
				__m128 Lay = _mm_load_ps(Layer34 + i * ImplN + j);
				tmp = _mm_mul_ps(mulRate, tmp);
				j += STEP;
				Lay = _mm_sub_ps(Lay, tmp);
				_mm_store_ps(Layer34 + i * ImplN + j-STEP, Lay);
				//Layer34[i * ImplN + j] -= learningRate * Delta4[i] * Impl2[j];
			}
			bia4[i] -= learningRate * Delta4[i];
		}

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		pthread_barrier_wait(&barrier);
		for (int i = lo; i < hi; i++)//i: ImplN
		{
			__m128 resTmp = _mm_setzero_ps();
			//Delta2[i] = 0;
			for (int j = 0; j < ImplN;)//j: ImplN
			{
				__m128 atmp = _mm_load_ps(Delta3 + j);
				__m128 btmp = _mm_set_ps(Layer23[(j + 3) * ImplN + i], Layer23[(j + 2) * ImplN + i], Layer23[(j + 1) * ImplN + i], Layer23[j * ImplN + i]);
				__m128 ctmp = _mm_mul_ps(atmp, btmp);
				j += STEP;
				resTmp = _mm_add_ps(resTmp, ctmp);
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
			for (int j = 0; j < ImplN;)//j: ImplN
			{
				__m128 tmp = _mm_load_ps(Impl1 + j);
				__m128 Lay = _mm_load_ps(Layer23 + i * ImplN + j);
				tmp = _mm_mul_ps(mulRate, tmp);
				j += STEP;
				Lay = _mm_sub_ps(Lay, tmp);
				_mm_store_ps(Layer23 + i * ImplN + j-STEP, Lay);
				//Layer23[i * ImplN + j] -= learningRate * Delta3[i] * Impl1[j];
			}
			bia3[i] -= learningRate * Delta3[i];
		}

		getLoHiIdx(thread_no, thread_num, ImplN, lo, hi);
		for (int i = lo; i < hi; i++)//i: ImplN
		{
			__m128 mulRate = _mm_set1_ps(learningRate * Delta2[i]);
			for (int j = 0; j < 784;)//j: 784
			{
				__m128 tmp = _mm_load_ps(BatchDataX + j);
				__m128 Lay = _mm_load_ps(Layer12 + i * 784 + j);
				tmp = _mm_mul_ps(mulRate, tmp);
				j += STEP;
				Lay = _mm_sub_ps(Lay, tmp);
				_mm_store_ps(Layer12 + i * 784 + j-STEP, Lay);
				//Layer12[i * 784 + j] -= learningRate * Delta2[i] * BatchDataX[j];
			}
			bia2[i] -= learningRate * Delta2[i];
		}
		pthread_barrier_wait(&barrier);
		if (thread_no == 0)sem_post(&endSem);
		//pthread_barrier_wait(&barrier);
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
	state = FORWARD;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//pthread_cond_broadcast(&cond);
	for (int i=0;i<NUM_THREADS;i++)
		sem_post(&beginSem);
	sem_wait(&endSem);
	//pthread_mutex_lock(&condMutex);
	//pthread_cond_wait(&mainCond, &condMutex);
	//pthread_mutex_unlock(&condMutex);
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

	state = BACKWARD;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//pthread_cond_broadcast(&cond);
	for (int i = 0; i < NUM_THREADS; i++)
		sem_post(&beginSem);
	sem_wait(&endSem);
	//pthread_mutex_lock(&condMutex);
	//pthread_cond_wait(&mainCond, &condMutex);
	//pthread_mutex_unlock(&condMutex);
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
	//pthread_cond_broadcast(&cond);
	for (int i = 0; i < NUM_THREADS; i++)
		sem_post(&beginSem);
	sem_wait(&endSem);
	//pthread_mutex_lock(&condMutex);
	//pthread_cond_wait(&mainCond, &condMutex);
	//pthread_mutex_unlock(&condMutex);
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
	//pthread_cond_init(&cond,NULL);
	//pthread_mutex_init(&condMutex,NULL);
	//pthread_cond_init(&mainCond,NULL);
	sem_init(&beginSem,0,0);
	sem_init(&endSem,0,0);
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
	//pthread_cond_broadcast(&cond);
	for (int i = 0; i < NUM_THREADS; i++)
		sem_post(&beginSem);
	for (int i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], &ret);
	pthread_barrier_destroy(&barrier);
	pthread_mutex_destroy(&softSumMutex);
	//pthread_cond_destroy(&cond);
	//pthread_mutex_destroy(&condMutex);
	//pthread_cond_destroy(&mainCond);
	sem_destroy(&beginSem);
	sem_destroy(&endSem);
	return 0;
}