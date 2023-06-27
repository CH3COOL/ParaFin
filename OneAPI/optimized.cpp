#define _CRT_SECURE_NO_WARNINGS
#include<cstdio>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<chrono>
#include<sycl/sycl.hpp>
using namespace sycl;
#define ImplN 800	//超参数：隐含层 
#define LEARN_RATE (float)0.1 //超参数：学习率 
#define LEARN_RATE_INV 10
#define BATCH 10
//4 Layers
float Layer12[ImplN * 28 * 28], bia2[ImplN];
float Layer23[ImplN * ImplN], bia3[ImplN];
float Layer34[ImplN * 10], bia4[10];
queue* q;
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
void gpu_kernel(float* A, float* B, float* C,//C=A*B
	int M, int N, int K,//A M*K B K*N
	int BLOCK, sycl::queue& q) {
	// define the workgroup size and mapping
	buffer buf_A(A, range<2>(M, K));
	buffer buf_B(B, range<2>(K, N));
	buffer buf_C(C, range<2>(M, N));
	auto e = q.submit([&](sycl::handler& h) {
		accessor As(buf_A, h, read_only);
		accessor Bs(buf_B, h, read_only);
		accessor Cs(buf_C, h, write_only);
		h.parallel_for<class k_name_t>(
			range<2>(M, N), [=](sycl::id<2> index) {
				// core computation
				float sum = 0.0f;
				for (int k = 0; k < K; ++k) {
					sum += As[index[0]][k] * Bs[k][index[1]];
				}
				Cs[index] = sum;

			});
		});
	//e.wait();
	return;
}
void NormalMul(int n, int s, int m, float* a, float* b, float* res)
{
	//for (int i = 0; i < n; i++)
	//	for (int j = 0; j < m; j++)
	//	{
	//		res[m * i + j] = 0;
	//		for (int k = 0; k < s; k++)
	//			//res:i行j列 
	//			//a:i行k列 
	//			res[m * i + j] += (a[i * s + k] * b[k * m + j]);
	//	}
	gpu_kernel(a, b, res, n, m, s, 4, *q);
}
void sigMoid(int length, float* data)
{
	//for (int i = 0; i < length; i++)
	//	data[i] = 1 / (1 + exp(-data[i]));
	buffer buf_data(data, range<1>(length));
	q->submit([&](handler& h)
		{
			accessor dataS(buf_data, h, read_write);
			h.parallel_for<class sigMoid>(range<1>(length), [=](id<1>index) {
				dataS[index] = 1 / (1 + exp(-dataS[index]));
				});
		});// .wait();
}
void addBias(int length, int batchNum, float* res, float* bias)//res：batchNum*length
{
	//for (int batchIdx = 0; batchIdx < batchNum; batchIdx++)
	//	for (int i = 0; i < length; i++)
	//		res[batchIdx * length + i] += bias[i];
	buffer buf_bias(bias, range<1>(length));
	buffer buf_res(res, range<1>(length * batchNum));
	q->submit([&](handler& h) {
		accessor bs(buf_bias, h, read_only);
		accessor rs(buf_res, h, read_write);
		h.parallel_for<class bia>(range<1>(length * batchNum), [=](id<1>index) {
			rs[index] += bs[index % length];
			});
		});// .wait();
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
//不会释放BatchData 
//Impl1、Impl2、return指针自行释放 
{
	//X后=w * X前 
	float* preLayer = BatchData;//Batch*784
	float* nextLayer = new float[ImplN * batchNum];//开辟2层
	//L12：784*ImplN

	NormalMul(batchNum, 784, ImplN, preLayer, Layer12, nextLayer);//第1->2层计算
	//nextLayer：Batch*ImplN
	addBias(ImplN, batchNum, nextLayer, bia2);//1->2偏置 
	sigMoid(ImplN * batchNum, nextLayer);//第1->2层激活 

	Impl1 = preLayer = nextLayer;//下一层 ；preLayer指向2层 
	//Batch*ImplN,ImplN*ImplN
	nextLayer = new float[ImplN * batchNum];//nextLayer：开辟3层 

	NormalMul(batchNum, ImplN, ImplN, preLayer, Layer23, nextLayer);//第2->3层计算 
	addBias(ImplN, batchNum, nextLayer, bia3);//2->3偏置 
	//delete[]preLayer;//第2层释放 
	sigMoid(ImplN * batchNum, nextLayer);//第2->3层激活 

	Impl2 = preLayer = nextLayer;//下一层；preLayer指向第3层
	nextLayer = new float[10 * batchNum];//nextLayer：开辟4层 

	//Layer34:ImplN*10
	NormalMul(batchNum, ImplN, 10, preLayer, Layer34, nextLayer);//第3->4层计算
	addBias(10, batchNum, nextLayer, bia4);//3->4偏置 

	//Batch*10
	//delete[]preLayer;//第3层释放 
	softmax10(batchNum, nextLayer);
	//return nextLayer;//需要在外面自行释放 
	predY = nextLayer;
}
float OneHotCrossEntropy(int batchNum, float* Result, int* RealResult)//Batch*10
{
	float bat = batchNum;
	float loss = 0;
	for (int i = 0; i < batchNum; i++)
		loss -= log(Result[i * 10 + RealResult[i]]);
	return loss / bat;
}
void backward(float* BatchData, float* Impl1, float* Impl2, float* yOut, float* yTag, float learningRate, int batchNum)
//逆梯度改权 
{
	float* DeltaAft = new float[10 * batchNum];//4层的dL/dz 
	//算dL/dz Layer4 

	//没有考虑batchNum 
	{
		buffer buf_delta4(DeltaAft, range<1>(batchNum * 10));
		buffer buf_yOut(yOut, range<1>(batchNum * 10));
		buffer buf_yTag(yTag, range<1>(batchNum * 10));
		//buffer buf_batchVar(&batchNum,range<1>(1));
		q->submit([&](handler& h) {
			accessor delta4(buf_delta4, h, write_only);
			accessor yout(buf_yOut, h, read_only);
			accessor ytag(buf_yTag, h, read_only);
			h.parallel_for<class dlt>(range<1>(batchNum * 10), [=](id<1>index) {
				delta4[index] = (yout[index] - ytag[index]) / BATCH;
				});
			});
		//for (int i = 0; i < batchNum * 10; i++)
		//	DeltaAft[i] = (yOut[i] - yTag[i]) / batchNum;
	}
	//DeltaAft:Batch*10
	float* DeltaBef = new float[ImplN * batchNum];//3层的 
	//DeltaBef:Batch*10 10*ImplN=Batch*ImplN
	{
		buffer buf_delta3(DeltaBef, range<1>(batchNum * ImplN));
		buffer buf_delta4(DeltaAft, range<1>(batchNum * 10));
		buffer buf_Layer34(Layer34, range<1>(ImplN * 10));
		buffer buf_Impl2(Impl2, range<1>(batchNum * ImplN));
		q->submit([&](handler& h) {
			accessor delta3(buf_delta3, h, write_only);
			accessor delta4(buf_delta4, h, read_only);
			accessor L34(buf_Layer34, h, read_only);
			accessor impl2(buf_Impl2, h, read_only);
			h.parallel_for(range<1>(batchNum * ImplN), [=](id<1>index) {
				float res = 0.0;
				auto b = index / ImplN;
				auto i = index % ImplN;
				for (int j = 0; j < 10; j++)
				{
					res += (delta4[b * 10 + j] * L34[i * 10 + j]);
				}
				res *= ((impl2[index]) * (1 - impl2[index]));
				delta3[index] = res;
				});
			});
		//for (int b = 0; b < batchNum; b++)
		//{
		//	for (int i = 0; i < ImplN; i++)
		//	{
		//		DeltaBef[b * ImplN + i] = 0;//Layer34:ImplN*10
		//		for (int j = 0; j < 10; j++)
		//		{//[b][i]=[b][j] * WT[j][i] = [b][j] * W[i][j]
		//			DeltaBef[b * ImplN + i] += (DeltaAft[b * 10 + j] * Layer34[i * 10 + j]);
		//		}
		//		DeltaBef[b * ImplN + i] *= ((Impl2[b * ImplN + i]) * (1 - Impl2[b * ImplN + i]));
		//	}
		//}
	}

	//dz/dwij=a,dz/db=1 后乘前 
	//改Layer34 
	{
		buffer buf_delta4(DeltaAft, range<1>(batchNum * 10));
		buffer buf_L34(Layer34, range<1>(10 * ImplN));
		buffer buf_Impl2(Impl2, range<1>(batchNum * ImplN));
		buffer buf_bia4(bia4, range<1>(10));
		q->submit([&](handler& h) {
			accessor delta4(buf_delta4, h, read_only);
			accessor L34(buf_L34, h, read_write);
			accessor impl2(buf_Impl2, h, read_only);
			h.parallel_for(range<1>(10 * ImplN), [=](id<1>index) {
				//auto b = index / ImplN;
				//auto i = index % ImplN;
				//for (int j = 0; j < 10; j++)
				//	L34[i * 10 + j] -= delta4[b * 10 + j] * impl2[b * ImplN + i] * LEARN_RATE;
				auto i = index / 10;
				auto j = index % 10;
				for (int b = 0; b < BATCH; b++)
					L34[index] -= LEARN_RATE * delta4[b * 10 + j] * impl2[b * ImplN + i];
				}); 
			});
		q->submit([&](handler& h) {
			accessor b4(buf_bia4, h, read_write);
			accessor delta4(buf_delta4, h, read_only);
			h.parallel_for(range<1>(10), [=](id<1>index) {
				//b4[index % 10] -= delta4[index] * LEARN_RATE;
				for (int b = 0; b < BATCH; b++)
					b4[index] -= delta4[b * 10 + index] * LEARN_RATE;
				});
			});
	}
	//for (int b = 0; b < batchNum; b++)
	//{
	//	for (int i = 0; i < ImplN; i++)
	//	{
	//		for (int j = 0; j < 10; j++)
	//		{
	//			//Learning rate?
	//			//Layer34: ImplN*Batch Batch*10
	//			//ImplN*10 = ImplN * Batch,Batch*10
	//			//Impl2 T[i][b] * DeltaAft[b][j] = Impl2[b][i]*DeltaAft[b][j]
	//			Layer34[i * 10 + j] -= learningRate * DeltaAft[b * 10 + j] * Impl2[b * ImplN + i];//Delta4
	//		}
	//	}
	//	for (int i = 0; i < 10; i++)
	//		bia4[i] -= learningRate * DeltaAft[b * 10 + i];
	//}

	delete[]DeltaAft;
	DeltaAft = DeltaBef;//3层的 

	DeltaBef = new float[ImplN * batchNum];//2层的 
	{
		buffer buf_delta2(DeltaBef, range<1>(ImplN * batchNum));
		buffer buf_delta3(DeltaAft, range<1>(ImplN * batchNum));
		buffer buf_L23(Layer23, range<1>(ImplN * ImplN));
		buffer buf_Impl1(Impl1, range<1>(ImplN * batchNum));
		q->submit([&](handler& h) {
			accessor delta2(buf_delta2, h, write_only);
			accessor delta3(buf_delta3, h, read_only);
			accessor L23(buf_L23, h, read_only);
			accessor impl1(buf_Impl1, h, read_only);
			h.parallel_for(range<1>(batchNum * ImplN), [=](id<1>index) {
				float r = 0.0;
				auto b = index / ImplN;
				auto i = index % ImplN;
				for (int j = 0; j < ImplN; j++)
				{
					r += (delta3[b * ImplN + j] * L23[i * ImplN + j]);
				}
				r *= ((impl1[index]) * (1 - impl1[index]));
				delta2[index] = r;
				});
			});
		//for (int b = 0; b < batchNum; b++)
		//{
		//	for (int i = 0; i < ImplN; i++)
		//	{
		//		DeltaBef[b * ImplN + i] = 0;//Layer23:ImplN*ImplN
		//		for (int j = 0; j < ImplN; j++)
		//		{//[b][i]=[b][j] * WT[j][i] = [b][j] * W[i][j]
		//			DeltaBef[b * ImplN + i] += (DeltaAft[b * ImplN + j] * Layer23[i * ImplN + j]);
		//		}
		//		DeltaBef[b * ImplN + i] *= ((Impl1[b * ImplN + i]) * (1 - Impl1[b * ImplN + i]));
		//	}
		//}
	}
	{
		buffer buf_L23(Layer23, range<1>(ImplN * ImplN));
		buffer buf_delta3(DeltaAft, range<1>(batchNum * ImplN));
		buffer buf_Impl1(Impl1, range<1>(batchNum * ImplN));
		buffer buf_bia3(bia3, range<1>(ImplN));
		q->submit([&](handler& h) {
			accessor delta3(buf_delta3, h, read_only);
			accessor L23(buf_L23, h, read_write);
			accessor impl1(buf_Impl1, h, read_only);
			h.parallel_for(range<1>(ImplN * ImplN), [=](id<1>index) {
				//auto b = index / ImplN;
				//auto i = index % ImplN;
				//for (int j = 0; j < ImplN; j++)
				//	L23[i * ImplN + j] -= LEARN_RATE * delta3[b * ImplN + j] * impl1[index];
				//b3[i] -= LEARN_RATE * delta3[index];
				auto i = index / ImplN;
				auto j = index % ImplN;
				for (int b = 0; b < BATCH; b++)
					L23[index] -= LEARN_RATE * delta3[b * ImplN + j] * impl1[b * ImplN + i];
				});
			});
		q->submit([&](handler&h) {
			accessor b3(buf_bia3, h, read_write);
			accessor delta3(buf_delta3, h, read_only);
			h.parallel_for(range<1>(ImplN), [=](id<1>index) {
				for(int b=0;b<BATCH;b++)
					b3[index] -= LEARN_RATE * delta3[b*ImplN+index];
				});
			});
		//for (int b = 0; b < batchNum; b++)
		//{
		//	for (int i = 0; i < ImplN; i++)
		//	{
		//		for (int j = 0; j < ImplN; j++)
		//		{
		//			Layer23[i * ImplN + j] -= learningRate * DeltaAft[b * ImplN + j] * Impl1[b * ImplN + i];//Delta3
		//		}
		//		bia3[i] -= learningRate * DeltaAft[b * ImplN + i];
		//	}
		//}
	}
	delete[]DeltaAft;
	DeltaAft = DeltaBef;//2层的
	{
		buffer buf_L12(Layer12, range<1>(784 * ImplN));
		buffer buf_BatchData(BatchData, range<1>(batchNum * 784));
		buffer buf_delta2(DeltaAft, range<1>(batchNum * ImplN));
		buffer buf_bia2(bia2, range<1>(ImplN));
		q->submit([&](handler& h) {
			accessor L12(buf_L12, h, read_write);
			accessor delta2(buf_delta2, h, read_only);
			accessor batchdata(buf_BatchData, h, read_only);
			h.parallel_for(range<1>(ImplN * 784), [=](id<1>index) {
				auto i = index / ImplN;
				auto j = index % ImplN;
				for (int b = 0; b < BATCH; b++)
					L12[index] -= LEARN_RATE * delta2[b * ImplN + j] * batchdata[b * 784 + i];
				});
			});
		q->submit([&](handler& h) {
			accessor b2(buf_bia2, h, read_write);
			accessor delta2(buf_delta2, h, read_only);
			h.parallel_for(range<1>(ImplN), [=](id<1>index) {
				//auto i = index % ImplN;
				//b2[i] -= LEARN_RATE * delta2[index];
				for (int b = 0; b < BATCH; b++)
					b2[index] -= LEARN_RATE * delta2[b*ImplN+index];
				});
			});
		//for (int b = 0; b < batchNum; b++)
		//{
		//	for (int i = 0; i < 784; i++)
		//	{
		//		for (int j = 0; j < ImplN; j++)
		//		{
		//			Layer12[i * ImplN + j] -= learningRate * DeltaAft[b * ImplN + j] * BatchData[b * 784 + i];//Delta2
		//		}
		//	}
		//	for (int i = 0; i < ImplN; i++)
		//		bia2[i] -= learningRate * DeltaAft[b * ImplN + i];
		//}
	}
	delete[]DeltaAft;
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
	float* X = new float[784 * BATCH];
	float* Y = new float[10 * BATCH];
	for (int j = 0; j < BATCH; j++)
	{
		fread(img, 1, 784, fx);
		fread(&tag, 1, 1, fy);
		for (int i = 0; i < 784; i++)X[784 * j + i] = img[i] / 255.0;
		for (int i = 0; i < 10; i++)Y[10 * j + i] = (tag == i);//Batch*10
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
	cumLoss += loss;
	s = std::chrono::high_resolution_clock::now();
	backward(X, Imp1, Imp2, predY, Y, LEARN_RATE, BATCH);
	e = std::chrono::high_resolution_clock::now();
	backwardTime = std::chrono::duration<float, std::milli>(e - s).count();
	delete[]Imp1;
	delete[]Imp2;
	delete[]X;
	delete[]Y;
	delete[]predY;
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
	q = new queue();
	for (int epo = 0; epo < 1; epo++)
	{
		fx = fopen("./lab/train-images.idx3-ubyte", "rb");
		fseek(fx, 16, SEEK_SET);
		fy = fopen("./lab/train-labels.idx1-ubyte", "rb");
		fseek(fy, 8, SEEK_SET);
		float forwardTotalTime = 0.0, backwardTotalTime = 0.0;
		for (int i = 1; i <= 60000 / BATCH; i++)
		{
			float oneForwardTime, oneBackwardTime;
			put_one(i
				, oneForwardTime, oneBackwardTime
			);
			forwardTotalTime += oneForwardTime;
			backwardTotalTime += oneBackwardTime;
		}
		printf("forward avg elapsed %.3f ms\nbackward avg elapsed %.3f ms", forwardTotalTime / (60000 / BATCH), backwardTotalTime / (60000 / BATCH));
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
	printf("%.2f\n", rightCnt / 10000.0);
	fclose(fx);
	fclose(fy);
	delete q;
	return 0;
}