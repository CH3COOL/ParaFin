#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include"Windows.h"
#define ImplN 500	//超参数：隐含层 
#define LEARN_RATE 0.1 //超参数：学习率 
//4 Layers
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
void NormalMul(int n, int s, int m, float* a, float* b, float* res)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
		{
			res[m * i + j] = 0;
			for (int k = 0; k < s; k++)
				//res:i行j列 
				//a:i行k列 
				res[m * i + j] += (a[i * s + k] * b[k * m + j]);
		}
}
void sigMoid(int length, float* data)
{
	for (int i = 0; i < length; i++)
		data[i] = 1 / (1 + exp(-data[i]));
}
void addBias(int length, int batchNum, float* res, float* bias)//res：length*batchNum
{
	for (int batchIdx = 0; batchIdx < batchNum; batchIdx++)
		for (int i = 0; i < length; i++)
			res[i * batchNum + batchIdx] += bias[i];
}
void softmax10(int batchNum, float* result)
{
	for (int i = 0; i < batchNum; i++)
	{
		float sum = 0;
		for (int j = 0; j < 10; j++)
			sum += exp(result[10 * i + j]);
		for (int j = 0; j < 10; j++)
			result[10 * i + j] = exp(result[10 * i + j]) / sum;
	}
}
void Forward(int batchNum, float* BatchData, float*& Impl1, float*& Impl2, float*& predY)//row:28*28,col:batchNum  
//returns PredictResult
//不会释放BatchData 
//Impl1、Impl2、return指针自行释放 
{
	//X后=w * X前 
	float* preLayer = BatchData;
	float* nextLayer = new float[ImplN * batchNum];//开辟2层 
	NormalMul(ImplN, 784, batchNum, Layer12, preLayer, nextLayer);//第1->2层计算
	addBias(ImplN, batchNum, nextLayer, bia2);//1->2偏置 
	sigMoid(ImplN * batchNum, nextLayer);//第1->2层激活 
	Impl1 = preLayer = nextLayer;//下一层 ；preLayer指向2层 
	nextLayer = new float[ImplN * batchNum];//nextLayer：开辟3层 
	NormalMul(ImplN, ImplN, batchNum, Layer23, preLayer, nextLayer);//第2->3层计算 
	addBias(ImplN, batchNum, nextLayer, bia3);//2->3偏置 
	//delete[]preLayer;//第2层释放 
	sigMoid(ImplN, nextLayer);//第2->3层激活 
	Impl2 = preLayer = nextLayer;//下一层；preLayer指向第3层
	nextLayer = new float[10 * batchNum];//nextLayer：开辟4层 
	NormalMul(10, ImplN, batchNum, Layer34, preLayer, nextLayer);//第3->4层计算
	addBias(10, batchNum, nextLayer, bia4);//3->4偏置 
	//delete[]preLayer;//第3层释放 
	softmax10(batchNum, nextLayer);
	//return nextLayer;//需要在外面自行释放 
	predY = nextLayer;
}
float OneHotCrossEntropy(int batchNum, float* Result, int* RealResult)
{
	float bat = batchNum;
	float loss = 0;
	for (int i = 0; i < batchNum; i++)
		loss -= log(Result[RealResult[i] * batchNum + i]);
	return loss / bat;
}
void backward(float* BatchData, float* Impl1, float* Impl2, float* yOut, float* yTag,float learningRate
	//,int batchNum=1
)//逆梯度改权 
{
	float* DeltaAft = new float[10
		//*batchNum
	];//4层的dL/dz 
	//算dL/dz Layer4 

	//没有考虑batchNum 
	for (int i = 0; i < 10; i++)DeltaAft[i] = yOut[i] - yTag[i];

	//dz/dwij=a,dz/db=1 后乘前 
	//改Layer34 
	float* Origin34 = new float[ImplN*10];
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < ImplN; j++)
		{
			Origin34[i * ImplN + j] = Layer34[i * ImplN + j];
			//Learning rate?
			Layer34[i * ImplN + j] -= learningRate *DeltaAft[i] * Impl2[j];
		}
		bia4[i] -= learningRate*DeltaAft[i];
	}
	//存Layer34? 

	float* DeltaBef = new float[ImplN
		//*batchNum
	];//3层的 
	for (int i = 0; i < ImplN; i++)
	{
		DeltaBef[i] = 0;
		for (int j = 0; j < 10; j++)
		{
			DeltaBef[i] += (DeltaAft[j] * Origin34[j * ImplN + i]);
		}
		DeltaBef[i] *= ((Impl2[i]) * (1 - Impl2[i]));
	}
	delete[]Origin34;

	delete[]DeltaAft;
	DeltaAft = DeltaBef;//3层的 
	float* Origin23 = new float[ImplN * ImplN];
	for (int i = 0; i < ImplN; i++)
	{
		for (int j = 0; j < ImplN; j++)
		{
			Origin23[i * ImplN + j] = Layer23[i * ImplN + j];
			Layer23[i * ImplN + j] -= learningRate*DeltaAft[i] * Impl1[j];
		}
		bia3[i] -= learningRate*DeltaAft[i];
	}

	DeltaBef = new float[ImplN
		//*batchNum
	];//2层的 
	
	for (int i = 0; i < ImplN; i++)
	{
		DeltaBef[i] = 0;
		for (int j = 0; j < ImplN; j++)
		{
			DeltaBef[i] += (DeltaAft[j] * Origin23[j * ImplN + i]);
		}
		DeltaBef[i] *= ((Impl1[i]) * (1 - Impl1[i]));
	}
	delete[]Origin23;
	delete[]DeltaAft;
	DeltaAft = DeltaBef;//2层的
	for (int i = 0; i < ImplN; i++)
	{
		for (int j = 0; j < 784; j++)
		{
			Layer12[i * 784 + j] -= learningRate*DeltaAft[i] * BatchData[j];
		}
		bia2[i] -= learningRate*DeltaAft[i];
	}
	delete[]DeltaAft;
}
FILE* fx;
FILE* fy;
float cumLoss = 0;
int rightCnt = 0;
void put_one(int idx
	//,float&forwardTime,float&backwardTime
)
{
	unsigned char img[784], tag;
	fread(img, 1, 784, fx);
	fread(&tag, 1, 1, fy);
	float* X = new float[784];
	float* Y = new float[10];
	for (int i = 0; i < 784; i++)X[i] = img[i] / 255.0;
	for (int i = 0; i < 10; i++)Y[i] = (tag == i);
	float* Imp1;
	float* Imp2;
	float* predY;
	//long long freq, head, tail;
	//QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Forward(1, X, Imp1, Imp2, predY);
	//QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//forwardTime = 1000.0*(tail - head) / freq;
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
	float loss = OneHotCrossEntropy(1, predY, &temp);
	cumLoss+= loss;
	//QueryPerformanceCounter((LARGE_INTEGER*)&head);
	backward(X, Imp1, Imp2, predY, Y,LEARN_RATE);
	//QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	//backwardTime = 1000.0 * (tail - head) / freq;
	delete[]Imp1;
	delete[]Imp2;
	delete[]X;
	delete[]Y;
	delete[]predY;
	if (idx % 1000 == 0)
	{
		printf("idx=%d Avg loss=%.3f\n", idx, cumLoss/1000);
		printf("Acc=%.2f\n",rightCnt/1000.0);
		cumLoss = 0;
		rightCnt = 0;
	}
}
void testOne(int idx)
{
	unsigned char img[784], tag;
	fread(img, 1, 784, fx);
	fread(&tag, 1, 1, fy);
	float* X = new float[784];
	float* Y = new float[10];
	for (int i = 0; i < 784; i++)X[i] = img[i] / 255.0;
	for (int i = 0; i < 10; i++)Y[i] = (tag == i);
	float* Imp1;
	float* Imp2;
	float* predY;
	Forward(1, X, Imp1, Imp2, predY);
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
	delete[]Imp1;
	delete[]Imp2;
	delete[]X;
	delete[]Y;
	delete[]predY;
	if (guess == tag)
		rightCnt++;
}
int main()
{
	srand((unsigned)time(NULL));
	randW();
	for (int epo = 0; epo < 3; epo++)
	{
		fx = fopen("train-images.idx3-ubyte", "rb");
		fseek(fx, 16, SEEK_SET);
		fy = fopen("train-labels.idx1-ubyte", "rb");
		fseek(fy, 8, SEEK_SET);
		//float forwardTotalTime = 0.0, backwardTotalTime = 0.0;
		for (int i = 1; i <= 60000; i++)
		{
			//float oneForwardTime, oneBackwardTime;
			put_one(i
				//, oneForwardTime, oneBackwardTime
				);
			//forwardTotalTime += oneForwardTime;
			//backwardTotalTime += oneBackwardTime;
		}
		//printf("forward avg elapsed %.3f ms\nbackward avg elapsed %.3f ms",forwardTotalTime/ 60000,backwardTotalTime/60000);
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
	printf("%.2f\n", rightCnt/10000.0);
	fclose(fx);
	fclose(fy);
	return 0;
}