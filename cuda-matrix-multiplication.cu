#include <stdio.h>
#include <math.h>
#define T threadIdx
#define B blockIdx
#define T_W 2		//T_W => Tile Width that lowers the burden on GPU for computation

/*matrix multiplication kernels*/
// shared
__global__ void MatrixMulSh( float *Md , float *Nd , float *Pd , const int W )
{

	//These shared variables are present in shared memory that is common to all threads within a block.
	__shared__ float Mds [T_W][T_W] ;
	__shared__ float Nds [T_W][T_W] ;

	// calculate thread id
	unsigned int col = T_W*B.x + T.x;
	unsigned int row = T_W*B.y + T.y;

	//printf("---------COL OF [%d]{%d} is: %d ------- \n", B.x, T.x, col);
	//printf("---------ROW OF [%d]{%d} is: %d ------- \n", B.y, T.y, row);
	float Pvalue = 0;
	// m indicate number of phase
	for (int m = 0 ; m < W/T_W ; m++ ) {

		//printf("***** FOR M : %d ******\n", m);
		
		Mds[T.y][T.x] =  Md[row*W + (m*T_W + T.x)];

		//printf("\nMds[%d][%d] = Md [%d]\n", T.y, T.x, (row*W+(m*T_W + T.x)));

		Nds[T.y][T.x] =  Nd[ col+( m*T_W + T.y) * W ] ;

		//printf("\nNds[%d][%d] = Nd [%d]\n", T.y, T.x, (col+(m*T_W + T.y)*W));

		__syncthreads() ; //for synchronizing the threads


		for (int k = 0; k < T_W; ++k) {
			Pvalue += Mds[T.x][k] * Nds[k][T.y];
			//printf("\nPvalue += Mds[%d][%d] * Nds[%d][%d]\n", T.x, k, k, T.y);             
		}
		__syncthreads();
	}

	Pd[row*W + col] = Pvalue;
	//printf("\n~~~Pd[%d] = %d~~~\n", row*W+col, Pvalue);
}

int main () {
	const int W = 6;
	float array1_h[W][W],array2_h[W][W],M_result_array_h[W][W];
	float *array1_d,*array2_d ,*M_result_array_d ; // device array  *result_array_d
	int i , j;
	//input in host array
	//hardcoding 1 in all slots of 1st array and 2 in all slots of 2nd array
	for ( i = 0 ; i<W ; i++ ) {
		for (j = 0 ; j<W ; j++ ) {
			array1_h[i][j] = 1;
			array2_h[i][j] = 2;
		}
	}

	//create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;

	cudaMalloc((void **) &array1_d , W*W*sizeof (int) ) ;

	cudaMalloc((void **) &array2_d , W*W*sizeof (int) ) ;



	//copy host array to device array; cudaMemcpy ( dest , source , W , direction )

	cudaMemcpy ( array1_d , array1_h , W*W*sizeof (int) , cudaMemcpyHostToDevice ) ;

	cudaMemcpy ( array2_d , array2_h , W*W*sizeof (int) , cudaMemcpyHostToDevice ) ;



	//allocating memory for resultent device array

	cudaMalloc((void **) &M_result_array_d , W*W*sizeof (int) );

	//calling kernal

	dim3 dimBlock ( W/T_W , W/T_W ,1 ) ;

	dim3 dimThread ( T_W, T_W, 1 ) ;

#if 1

MatrixMulSh<<<dimBlock,dimThread>>> ( array1_d , array2_d ,M_result_array_d , W) ;

#endif

	// all gpu function blocked till kernel is working
	//copy back result_array_d to result_array_h

	cudaMemcpy(M_result_array_h , M_result_array_d , W*W*sizeof(int),cudaMemcpyDeviceToHost) ;

	cudaFree(array1_d);
	cudaFree(array2_d);
	cudaFree(M_result_array_d);

	//printf the result array
	for ( i = 0 ; i<W ; i++ ) {
		for ( j = 0 ; j < W ; j++ ) {
			printf ("%f   ",M_result_array_h[i][j] ) ;
		}
		printf ("\n") ;
	}
	cudaFree(M_result_array_h);
}
