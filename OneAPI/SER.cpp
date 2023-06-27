#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<chrono>
#define ImplN 800	//�������������� 
#define LEARN_RATE 0.1 //��������ѧϰ�� 
#define BATCH 10
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
				//res:i��j�� 
				//a:i��k�� 
				res[m * i + j] += (a[i * s + k] * b[k * m + j]);
		}
}
void sigMoid(int length, float* data)
{
	for (int i = 0; i < length; i++)
		data[i] = 1 / (1 + exp(-data[i]));
}
void addBias(int length, int batchNum, float* res, float* bias)//res��batchNum*length
{
	for (int batchIdx = 0; batchIdx < batchNum; batchIdx++)
		for (int i = 0; i < length; i++)
			res[batchIdx*length+i] += bias[i];
}
void softmax10(int batchNum, float* result)//res:batchNum*10
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
//�����ͷ�BatchData 
//Impl1��Impl2��returnָ�������ͷ� 
{
	//X��=w * Xǰ 
	float* preLayer = BatchData;//Batch*784
	float* nextLayer = new float[ImplN * batchNum];//����2��
    //L12��784*ImplN
    
	NormalMul(batchNum, 784, ImplN, preLayer,Layer12, nextLayer);//��1->2�����
    //nextLayer��Batch*ImplN
	addBias(ImplN, batchNum, nextLayer, bia2);//1->2ƫ�� 
	sigMoid(ImplN * batchNum, nextLayer);//��1->2�㼤�� 
    
	Impl1 = preLayer = nextLayer;//��һ�� ��preLayerָ��2�� 
    //Batch*ImplN,ImplN*ImplN
	nextLayer = new float[ImplN * batchNum];//nextLayer������3�� 
    
	NormalMul(batchNum, ImplN, ImplN, preLayer,Layer23, nextLayer);//��2->3����� 
	addBias(ImplN, batchNum, nextLayer, bia3);//2->3ƫ�� 
	//delete[]preLayer;//��2���ͷ� 
	sigMoid(ImplN * batchNum, nextLayer);//��2->3�㼤�� 
    
	Impl2 = preLayer = nextLayer;//��һ�㣻preLayerָ���3��
	nextLayer = new float[10 * batchNum];//nextLayer������4�� 
    
    //Layer34:ImplN*10
	NormalMul(batchNum, ImplN, 10,preLayer,Layer34, nextLayer);//��3->4�����
	addBias(10, batchNum, nextLayer, bia4);//3->4ƫ�� 
    
    //Batch*10
	//delete[]preLayer;//��3���ͷ� 
	softmax10(batchNum, nextLayer);
	//return nextLayer;//��Ҫ�����������ͷ� 
	predY = nextLayer;
}
float OneHotCrossEntropy(int batchNum, float* Result, int* RealResult)//Batch*10
{
	float bat = batchNum;
	float loss = 0;
	for (int i = 0; i < batchNum; i++)
		loss -= log(Result[i*10+RealResult[i]]);
	return loss / bat;
}
void backward(float* BatchData, float* Impl1, float* Impl2, float* yOut, float* yTag,float learningRate,int batchNum)
//���ݶȸ�Ȩ 
{
	float* DeltaAft = new float[10*batchNum];//4���dL/dz 
	//��dL/dz Layer4 

	//û�п���batchNum 
	for (int i = 0; i < batchNum*10; i++)
		DeltaAft[i] = (yOut[i] - yTag[i])/batchNum;

	//DeltaAft:Batch*10
	float* DeltaBef = new float[ImplN*batchNum];//3��� 
	//DeltaBef:Batch*10 10*ImplN=Batch*ImplN
	for(int b=0;b<batchNum;b++)
	{
		for (int i = 0; i < ImplN; i++)
		{
			DeltaBef[b*ImplN+i] = 0;//Layer34:ImplN*10
			for (int j = 0; j < 10; j++)
			{//[b][i]=[b][j] * WT[j][i] = [b][j] * W[i][j]
				DeltaBef[b*ImplN+i] += (DeltaAft[b*10+j] * Layer34[i*10+j]);
			}
			DeltaBef[b*ImplN+i] *= ((Impl2[b*ImplN+i]) * (1 - Impl2[b*ImplN+i]));
		}
	}

	//dz/dwij=a,dz/db=1 ���ǰ 
	//��Layer34 
	for(int b=0;b<batchNum;b++)
	{
		for (int i = 0; i < ImplN; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				//Learning rate?
				//Layer34: ImplN*Batch Batch*10
				//ImplN*10 = ImplN * Batch,Batch*10
				//Impl2 T[i][b] * DeltaAft[b][j] = Impl2[b][i]*DeltaAft[b][j]
				Layer34[i * 10 + j] -= learningRate *DeltaAft[b*10+j] * Impl2[b*ImplN+i];//Delta4
			}
		}
		for(int i=0;i<10;i++)
			bia4[i] -= learningRate*DeltaAft[b*10+i];
	}

	delete[]DeltaAft;
	DeltaAft = DeltaBef;//3��� 
	
	DeltaBef = new float[ImplN*batchNum];//2��� 
	
	for(int b=0;b<batchNum;b++)
	{
		for (int i = 0; i < ImplN; i++)
		{
			DeltaBef[b*ImplN+i] = 0;//Layer23:ImplN*ImplN
			for (int j = 0; j < ImplN; j++)
			{//[b][i]=[b][j] * WT[j][i] = [b][j] * W[i][j]
				DeltaBef[b*ImplN+i] += (DeltaAft[b*ImplN+j] * Layer23[i * ImplN + j]);
			}
			DeltaBef[b*ImplN+i] *= ((Impl1[b*ImplN+i]) * (1 - Impl1[b*ImplN+i]));
		}
	}
	for(int b=0;b<batchNum;b++)
	{
		for (int i = 0; i < ImplN; i++)
		{
			for (int j = 0; j < ImplN; j++)
			{
				Layer23[i * ImplN + j] -= learningRate*DeltaAft[b*ImplN+j] * Impl1[b*ImplN+i];//Delta3
			}
			bia3[i] -= learningRate*DeltaAft[b*ImplN+i];
		}
	}
	delete[]DeltaAft;
	DeltaAft = DeltaBef;//2���
	for(int b=0;b<batchNum;b++)
	{
		for (int i = 0; i < 784; i++)
		{
			for (int j = 0; j < ImplN; j++)
			{
				Layer12[i * ImplN + j] -= learningRate*DeltaAft[b*ImplN+j] * BatchData[b*784+i];//Delta2
			}
		}
		for(int i=0;i<ImplN;i++)
			bia2[i] -= learningRate*DeltaAft[b*ImplN+i];
	}
	delete[]DeltaAft;
}
FILE* fx;
FILE* fy;
float cumLoss = 0;
int rightCnt = 0;
void put_one(int idx
	,float&forwardTime,float&backwardTime
)
{
	unsigned char img[784], tag;
    float* X = new float[784*BATCH];
	float* Y = new float[10*BATCH];
    for(int j=0;j<BATCH;j++)
    {
        fread(img, 1, 784, fx);
        fread(&tag, 1, 1, fy);
        for (int i = 0; i < 784; i++)X[784*j+i] = img[i] / 255.0;
        for (int i = 0; i < 10; i++)Y[10*j+i] = (tag == i);//Batch*10
    }
	float* Imp1;
	float* Imp2;
	float* predY;
    std::chrono::high_resolution_clock::time_point s, e;
    s = std::chrono::high_resolution_clock::now();
	Forward(BATCH, X, Imp1, Imp2, predY);
	e = std::chrono::high_resolution_clock::now();
    forwardTime = std::chrono::duration<float, std::milli>(e - s).count();
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
	s = std::chrono::high_resolution_clock::now();
	backward(X, Imp1, Imp2, predY, Y,LEARN_RATE,BATCH);
	e = std::chrono::high_resolution_clock::now();
	backwardTime = std::chrono::duration<float, std::milli>(e - s).count();
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
	for (int epo = 0; epo < 1; epo++)
	{
		fx = fopen("./lab/train-images.idx3-ubyte", "rb");
		fseek(fx, 16, SEEK_SET);
		fy = fopen("./lab/train-labels.idx1-ubyte", "rb");
		fseek(fy, 8, SEEK_SET);
		float forwardTotalTime = 0.0, backwardTotalTime = 0.0;
		for (int i = 1; i <= 60000/BATCH; i++)
		{
			float oneForwardTime, oneBackwardTime;
			put_one(i
				, oneForwardTime, oneBackwardTime
				);
			forwardTotalTime += oneForwardTime;
			backwardTotalTime += oneBackwardTime;
		}
		printf("forward avg elapsed %.3f ms\nbackward avg elapsed %.3f ms",forwardTotalTime/ (60000/BATCH),backwardTotalTime/(60000/BATCH));
		printf("\n\n");
		fclose(fx);
		fclose(fy);
	}
	fx = fopen("./lab/t10k-images.idx3-ubyte", "rb");
	fseek(fx, 16, SEEK_SET);
	fy = fopen("./lab/t10k-labels.idx1-ubyte", "rb");
	fseek(fy, 8, SEEK_SET);
	for (int i = 1; i <= 10000; i++)
		testOne(i);
	printf("%.2f\n", rightCnt/10000.0);
	fclose(fx);
	fclose(fy);
	return 0;
}
