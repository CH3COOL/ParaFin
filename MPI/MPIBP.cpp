#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<omp.h>
#include<malloc.h>
#include<sys/time.h>
#include<unistd.h>
#include <arm_neon.h>
#include<mpi.h>
#define ImplN 400	//³¬²ÎÊý£ºÒþº¬²ã 
#define THREADS_NUM 16
#define LEARN_RATE 0.1 //³¬²ÎÊý£ºÑ§Ï°ÂÊ 
//4 Layers
#define STEP 4
#pragma pack(8)
#define ALIGN_N 32
float Layer12[ImplN * 28 * 28], bia2[ImplN];
float Layer23[ImplN * ImplN], bia3[ImplN];
float Layer34[ImplN * 10], bia4[10];
float L12[ImplN * 28 * 28], b2[ImplN];
float L23[ImplN * ImplN], b3[ImplN];
float L34[ImplN * 10], b4[10];

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
		float32x4_t resTmp = vdupq_n_f32(0.0);
		for (int j = 0; j < 784; j += STEP)
		{
			float32x4_t inp = vld1q_f32(input + j);
			float32x4_t wei = vld1q_f32(weight + i * 784 + j);
			//inp = _mm_mul_ps(inp, wei);
			resTmp=vmlaq_f32(resTmp,inp,wei);
			//output[i]+=input[j]*weight[i*784+j];
		}
		float32x2_t s1,s2;
		s1=vget_low_f32(resTmp);
		s2=vget_high_f32(resTmp);
		s1=vpadd_f32(s1,s2);
		s1=vpadd_f32(s1,s1);
		vst1_lane_f32(output+i,s1,0);
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//_mm_store_ss(output + i, resTmp);
		output[i] += bias[i];
		output[i] = 1 / (1 + exp(-output[i]));
	}
}
void ForwardImplNToImplN(float* input, float* output, float* weight, float* bias)
{
#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)
	{
		float32x4_t resTmp = vdupq_n_f32(0.0);
		//__m128 resTmp = _mm_setzero_ps();
		for (int j = 0; j < ImplN; j += STEP)
		{
			//__m128 inp = _mm_load_ps(input + j);
			//__m128 wei = _mm_load_ps(weight + i * ImplN + j);
			float32x4_t inp = vld1q_f32(input + j);
			float32x4_t wei = vld1q_f32(weight + i * ImplN + j);
			resTmp=vmlaq_f32(resTmp,inp,wei);
			//inp = _mm_mul_ps(inp, wei);
			//resTmp = _mm_add_ps(resTmp, inp);
			//output[i]+=input[j]*weight[i*ImplN+j];
		}
		float32x2_t s1,s2;
		s1=vget_low_f32(resTmp);
		s2=vget_high_f32(resTmp);
		s1=vpadd_f32(s1,s2);
		s1=vpadd_f32(s1,s1);
		vst1_lane_f32(output+i,s1,0);

		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//_mm_store_ss(output + i, resTmp);
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
		float32x4_t resTmp = vdupq_n_f32(0.0);
		//__m128 resTmp = _mm_setzero_ps();
		for (int j = 0; j < ImplN; j += STEP)
		{
			float32x4_t inp = vld1q_f32(input + j);
			float32x4_t wei = vld1q_f32(weight + i * ImplN + j);
			//__m128 inp = _mm_load_ps(input + j);
			//__m128 wei = _mm_load_ps(weight + i * ImplN + j);
			resTmp=vmlaq_f32(resTmp,inp,wei);
			//inp = _mm_mul_ps(inp, wei);
			//resTmp = _mm_add_ps(resTmp, inp);
			//output[i]+=input[j]*weight[i*ImplN+j];
		}
		float32x2_t s1,s2;
		s1=vget_low_f32(resTmp);
		s2=vget_high_f32(resTmp);
		s1=vpadd_f32(s1,s2);
		s1=vpadd_f32(s1,s1);
		vst1_lane_f32(output+i,s1,0);

		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//_mm_store_ss(output + i, resTmp);
		output[i] += bias[i];
		output[i] = exp(output[i]);
		softmaxSum += output[i];
	}
	//#pragma omp //´Ë´¦Ó¦¸ÃSIMD¸ü¼Ó»®µÃÀ´ 
	float32x4_t divSum=vdupq_n_f32(softmaxSum);
	//__m128 divSum = _mm_set1_ps(softmaxSum);
	float32x4_t out;
	// for (int i = 0; i < STEP * (10 / STEP); i += STEP)
	// {
	// 	out = _mm_load_ps(output + i);
	// 	out = _mm_div_ps(out, divSum);
	// 	_mm_store_ps(output + i, out);
	// 	//output[i]/=softmaxSum;
	// }
	//#pragma omp simd
	float32x4_t sum_simd=vdupq_n_f32(softmaxSum);
	for (int i = 0; i < (10/STEP)*STEP; i+=STEP)
	{
		float32x4_t outt=vld1q_f32(output+i);
		outt=vdivq_f32(outt,sum_simd);
		vst1q_f32(output+i,outt);
		//output[i] /= softmaxSum;
	}
	for (int i = (10/STEP)*STEP; i < 10; i++)
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
	float* nextLayer = (float*)memalign(ALIGN_N,ImplN * sizeof(float));//_mm_malloc(ImplN * sizeof(float), 8);//¿ª±Ù2²ã 
	Forward784ToImplN(preLayer, nextLayer, Layer12, bia2);
	Impl1 = preLayer = nextLayer;//ÏÂÒ»²ã £»preLayerÖ¸Ïò2²ã 
	nextLayer = (float*)memalign(ALIGN_N,ImplN * sizeof(float));//nextLayer£º¿ª±Ù3²ã 
	ForwardImplNToImplN(preLayer, nextLayer, Layer23, bia3);
	Impl2 = preLayer = nextLayer;//ÏÂÒ»²ã£»preLayerÖ¸ÏòµÚ3²ã
	nextLayer = (float*)memalign(ALIGN_N,10*sizeof(float));//_mm_malloc(10 * sizeof(float), 8);//nextLayer£º¿ª±Ù4²ã 
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
	float* DeltaAft = (float*)memalign(ALIGN_N,10*sizeof(float));//_mm_malloc(10 * sizeof(float), 8);//4²ãµÄdL/dz 
	//ËãdL/dz Layer4 

	//²»¿¼ÂÇbatchNum 
	for (int i = 0; i < 10; i++)
		DeltaAft[i] = yOut[i] - yTag[i];


	float* DeltaBef = (float*)memalign(ALIGN_N,ImplN * sizeof(float));//3²ãµÄ 
	//ÏÈËã3²ãDelta 
#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < ImplN; i++)
	{
		float32x4_t resTmp = vdupq_n_f32(0.0);
		//__m128 resTmp = _mm_setzero_ps();
		//DeltaBef[i] = 0;
		for (int j = 0; j < STEP * (10 / STEP); j += STEP)
		{
			float temp[]={Layer34[j * ImplN + i],Layer34[(j + 1) * ImplN + i],Layer34[(j + 2) * ImplN + i],Layer34[(j + 3) * ImplN + i]};

			float32x4_t atmp=vld1q_f32(DeltaAft + j);
			float32x4_t btmp=vld1q_f32(temp);
			//__m128 atmp = _mm_load_ps(DeltaAft + j);
			//__m128 btmp = _mm_set_ps(, , , );
			//btmp = _mm_mul_ps(atmp, btmp);
			resTmp=vmlaq_f32(resTmp,atmp,btmp);
			//resTmp = _mm_add_ps(resTmp, btmp);
			//DeltaBef[i] += (DeltaAft[j] * Layer34[j * ImplN + i]);
		}
		float32x2_t s1,s2;
		s1=vget_low_f32(resTmp);
		s2=vget_high_f32(resTmp);
		s1=vpadd_f32(s1,s2);
		s1=vpadd_f32(s1,s1);
		vst1_lane_f32(DeltaBef+i,s1,0);

		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//_mm_store_ss(DeltaBef + i, resTmp);
		for (int j = STEP * (10 / STEP); j < 10; j++)
		{
			DeltaBef[i] += (DeltaAft[j] * Layer34[j * ImplN + i]);
		}
		DeltaBef[i] *= ((Impl2[i]) * (1 - Impl2[i]));//BefÎª3²ã 
	}

	//dz/dwij=a,dz/db=1 ºó³ËÇ° 
	//¸ÄLayer34 
#pragma omp parallel for num_threads(THREADS_NUM)
#pragma omp simd
	for (int i = 0; i < 10; i++)
	{
		//float mulRate=learningRate * DeltaAft[i];
		//__m128 mulRate = _mm_set1_ps(learningRate * DeltaAft[i]);
		#pragma omp simd
		for (int j = 0; j < ImplN; j ++)//= STEP)
		{
			//__m128 tmp = _mm_load_ps(Impl2 + j);
			//tmp = _mm_mul_ps(mulRate, tmp);
			//__m128 Lay = _mm_load_ps(Layer34 + i * ImplN + j);
			//Lay = _mm_sub_ps(Lay, tmp);
			//_mm_store_ps(Layer34 + i * ImplN + j, Lay);
			//Learning rate?
			Layer34[i * ImplN + j] -= learningRate *DeltaAft[i] * Impl2[j];
		}
		bia4[i] -= learningRate * DeltaAft[i];
	}
	free((void*)DeltaAft);
	//delete[]DeltaAft;//Delete 4²ãDelta 
	DeltaAft = DeltaBef;//3²ãDeltaÎªºóÃæ
	DeltaBef = (float*)memalign(ALIGN_N,ImplN * sizeof(float));//Ëã2²ãDelta 

#pragma omp parallel for num_threads(THREADS_NUM)
#pragma omp simd
	for (int i = 0; i < ImplN; i++)
	{
		//__m128 resTmp = _mm_setzero_ps();
		DeltaBef[i] = 0;
		#pragma omp simd
		for (int j = 0; j < ImplN; j ++)//= STEP)
		{
			DeltaBef[i] += (DeltaAft[j] * Layer23[j * ImplN + i]);
			//__m128 atmp = _mm_load_ps(DeltaAft + j);
			//__m128 btmp = _mm_set_ps(Layer23[(j + 3) * ImplN + i], Layer23[(j + 2) * ImplN + i], Layer23[(j + 1) * ImplN + i], Layer23[j * ImplN + i]);
			//btmp = _mm_mul_ps(atmp, btmp);
			//resTmp = _mm_add_ps(resTmp, btmp);
		}
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//resTmp = _mm_hadd_ps(resTmp, resTmp);
		//_mm_store_ss(DeltaBef + i, resTmp);
		DeltaBef[i] *= ((Impl1[i]) * (1 - Impl1[i]));
	}

#pragma omp parallel for num_threads(THREADS_NUM)
#pragma omp simd
	for (int i = 0; i < ImplN; i++)//¸ÄLayer23
	{
		//float mulRate=learningRate*DeltaAft[i];
		//__m128 mulRate = _mm_set1_ps(learningRate * DeltaAft[i]);
		#pragma omp simd
		for (int j = 0; j < ImplN; j ++)//= STEP)
		{
			//__m128 tmp = _mm_load_ps(Impl1 + j);
			//tmp = _mm_mul_ps(mulRate, tmp);
			//__m128 Lay = _mm_load_ps(Layer23 + i * ImplN + j);
			//Lay = _mm_sub_ps(Lay, tmp);
			//_mm_store_ps(Layer23 + i * ImplN + j, Lay);
			Layer23[i * ImplN + j] -= learningRate*DeltaAft[i] * Impl1[j];
		}
		bia3[i] -= learningRate * DeltaAft[i];
	}
	free(DeltaAft);
	//delete[]DeltaAft;//Delete 3²ãDelta 
	DeltaAft = DeltaBef;//2²ãDeltaÎªDeltaAft 

#pragma omp parallel for num_threads(THREADS_NUM)
#pragma omp simd
	for (int i = 0; i < ImplN; i++)//¸ÄLayer12 
	{
		//__m128 mulRate = _mm_set1_ps(learningRate * DeltaAft[i]);
		#pragma omp simd
		for (int j = 0; j < 784; j ++)//= STEP)
		{
			//__m128 tmp = _mm_load_ps(BatchData + j);
			//tmp = _mm_mul_ps(mulRate, tmp);
			//__m128 Lay = _mm_load_ps(Layer12 + i * 784 + j);
			//Lay = _mm_sub_ps(Lay, tmp);
			//_mm_store_ps(Layer12 + i * 784 + j, Lay);
			Layer12[i * 784 + j] -= learningRate*DeltaAft[i] * BatchData[j];
		}
		bia2[i] -= learningRate * DeltaAft[i];
	}

	free((void*)DeltaAft);
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
	float* X = (float*)memalign(ALIGN_N,784*sizeof(float));//_mm_malloc(784 * sizeof(float), 8);
	float* Y = (float*)memalign(ALIGN_N,10*sizeof(float));//_mm_malloc(10 * sizeof(float), 8);
	for (int i = 0; i < 784; i++)X[i] = img[i] / 255.0;
	for (int i = 0; i < 10; i++)Y[i] = (tag == i);
	float* Imp1;
	float* Imp2;
	float* predY;
    struct timeval beginTime,EndTime;
    gettimeofday(&beginTime,NULL);
	Forward(X, Imp1, Imp2, predY);
    gettimeofday(&EndTime,NULL);
	//_mm_free(predY);
	forwardTime=(EndTime.tv_sec-beginTime.tv_sec)*1000.0+(EndTime.tv_usec-beginTime.tv_usec)/1000.0;
	// int guess = 0;
	// float predPoss = predY[0];
	// for (int i = 1; i < 10; i++)
	// {
	// 	if (predY[i] > predPoss)
	// 	{
	// 		predPoss = predY[i];
	// 		guess = i;
	// 	}
	// }
	// rightCnt += (guess == tag);
	// int temp = tag;
	// float loss = OneHotCrossEntropy(predY, &temp);
	// cumLoss += loss;
    gettimeofday(&beginTime,NULL);
	backward(X, Imp1, Imp2, predY, Y, LEARN_RATE);
    gettimeofday(&EndTime,NULL);
	backwardTime = (EndTime.tv_sec-beginTime.tv_sec)*1000.0+(EndTime.tv_usec-beginTime.tv_usec)/1000.0;
	free(Imp1);
	free(Imp2);
	free(X);
	free(Y);
	free(predY);
	// if (idx % 1000 == 0)
	// {
	// 	printf("idx=%d Avg loss=%.3f\n", idx, cumLoss / 1000);
	// 	printf("Acc=%.2f\n", rightCnt / 1000.0);
	// 	cumLoss = 0;
	// 	rightCnt = 0;
	// }
}
void testOne(int idx)
{
	unsigned char img[784], tag;
	fread(img, 1, 784, fx);
	fread(&tag, 1, 1, fy);
	float* X = (float*)memalign(ALIGN_N,784*sizeof(float));//_mm_malloc(784 * sizeof(float), 8);
	float* Y = (float*)memalign(ALIGN_N,10*sizeof(float));//_mm_malloc(10 * sizeof(float), 8);
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
	free(Imp1);
	free(Imp2);
	free(X);
	free(Y);
	free(predY);
	if (guess == tag)
		rightCnt++;
}
int provided,size,my_rank;
int main()
{
	//omp_set_num_threads(4);
	//printf("%d\n", omp_get_num_threads());
	MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided);
	if(provided<MPI_THREAD_FUNNELED)
		MPI_Abort(MPI_COMM_WORLD,1);
	srand((unsigned)time(NULL));
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	if(my_rank==0){
		randW();
	}
	MPI_Bcast(Layer12,ImplN*28*28,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(bia2,ImplN,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(Layer23,ImplN*ImplN,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(bia3,ImplN,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(Layer34,ImplN*10,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(bia4,10,MPI_FLOAT,0,MPI_COMM_WORLD);
	for (int epo = 0; epo < 1; epo++)
	{
		fx = fopen("train-images.idx3-ubyte", "rb");
		fseek(fx, 16+784*(60000/size)*my_rank, SEEK_SET);
		fy = fopen("train-labels.idx1-ubyte", "rb");
		fseek(fy, 8+(60000/size)*my_rank, SEEK_SET);
		float forwardTotalTime = 0.0, backwardTotalTime = 0.0,Comm_Time=0.0,Cal_Time=0.0;
		double epoBeg=MPI_Wtime();
		for (int i = 1; i <= 60000/size; i++)
		{
			float oneForwardTime, oneBackwardTime;
			put_one(i
				, oneForwardTime, oneBackwardTime
			);
			forwardTotalTime += oneForwardTime;
			backwardTotalTime += oneBackwardTime;
			if(size!=1)
			{
				double Comm_Beg=MPI_Wtime();
				MPI_Allreduce(Layer12,L12,784*ImplN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				MPI_Allreduce(bia2,b2,ImplN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				MPI_Allreduce(Layer23,L23,ImplN*ImplN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				MPI_Allreduce(bia3,b3,ImplN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				MPI_Allreduce(Layer34,L34,ImplN*10,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				MPI_Allreduce(bia4,b4,ImplN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				double Comm_End=MPI_Wtime();
				Comm_Time+=(Comm_End-Comm_Beg);
				//#pragma omp simd
				double avg_Beg=MPI_Wtime();
				#pragma omp parallel for num_threads(THREADS_NUM)
				for(int i=0;i<784*ImplN;i++)
					Layer12[i]=L12[i]/size;
				#pragma omp parallel for num_threads(THREADS_NUM)
				for(int i=0;i<ImplN*ImplN;i++)
					Layer23[i]=L23[i]/size;
				#pragma omp parallel for num_threads(THREADS_NUM)
				for(int i=0;i<ImplN*10;i++)
					Layer34[i]=L34[i]/size;
				#pragma omp parallel for num_threads(THREADS_NUM)
				for(int i=0;i<ImplN;i++)
					bia2[i]=b2[i]/size;
				#pragma omp parallel for num_threads(THREADS_NUM)
				for(int i=0;i<ImplN;i++)
					bia3[i]=b3[i]/size;
				#pragma omp parallel for num_threads(THREADS_NUM)
				for(int i=0;i<10;i++)
					bia4[i]=b4[i]/size;
				double avg_End=MPI_Wtime();
				Cal_Time+=(avg_End-avg_Beg);
			}
		}
		double epoEnd=MPI_Wtime();
		printf("forward elapsed %.2f s\nbackward elapsed %.2f s", forwardTotalTime/1000, backwardTotalTime/1000);
		printf("\n\n");
		fclose(fx);
		fclose(fy);
		double epoElsp=epoEnd-epoBeg;
		double resElsp;
		MPI_Reduce(&epoElsp,&resElsp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
		if(my_rank==0)
		{
			printf("Epoch Elasped %.2lf s\n\n",resElsp);
			printf("Comm Elasped %.2f s \n\n",Comm_Time);
			printf("Avg Elasped %.2f s\n\n",Cal_Time);
		}
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
	MPI_Finalize();
	return 0;
}