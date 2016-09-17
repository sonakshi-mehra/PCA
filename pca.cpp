/*
* Name : Sonakshi Mehra
* Enrollment no. : 12103473
* Batch : B3
* Email : sonakshimehra.sona@gmail.com
*/

/* --NOTE--
* C++ used
* Working code for n-dimentional data and thoroughly tested
* I have used 'eigen' library in LINUX to compute eigenvalues and eigenvectors
* Steps to install 'eigen' library.....type the below in the terminal:
* $ sudo apt-get install libeigen3-dev  
*
*
* STEPS TO COMPILE AND RUN :
* $ g++ pca.cpp -o sona
* $ ./sona
*/

#include <stdio.h>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

int main()
{
	int n,total,i,j,z;
	cout<<"Enter the number of dimentions : ";
	cin>>n;
	cout<<"Enter total number of points   : ";
	cin>>total;
	float *mean;                                                   /* array with rows=dimentions of the point and each row will store the mean corresponding to that dimention */
	MatrixXd point(total,n),pointAdjust(total,n),cov(n,n);         /*point is a matrix which stores the origional points fed in by the user*/
																   /*pointAdjust stores the mean adjusted points */
	mean=new float[n];                                             /*dynamic allocation of array */
	for(j=0;j<n;j++)
		mean[j]=0;                                                 /*initially set mean of all dimentions to 0, will be changed later on */

	/******* Input points and calculation of mean for each dimention ********/
	for(i=0;i<total;i++)  
	{
		cout<<"Enter coordinates of point #"<<i+1<<" seperated by spaces : ";
		for(j=0;j<n;j++)
		{
			cin>>point(i,j);
			mean[j]+=point(i,j);
		}

	}
	for(j=0;j<n;j++)
		mean[j]/=total;
	


	/******** Points adjusted by subtracting mean from them *******************/
	for(i=0;i<total;i++)
	{
		for(j=0;j<n;j++)
			pointAdjust(i,j)=point(i,j)-mean[j];
	}

	/******** Calculation of covariance matrix *******************/
	float temp;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			temp=0;
			for(z=0;z<total;z++)
			{
				temp+=pointAdjust(z,i)*pointAdjust(z,j);
			}
			temp/=(total-1);
			cov(i,j)=temp;
		}
	}


	SelfAdjointEigenSolver<MatrixXd> es;					       /* SelfAdjointEigenSolver is a class in Eigen library which computes eigenvalues and eigenvectors for a given matrix */
	es.compute(cov);                                               /* compute is a public member function of SelfAdjointEigenSolver */
	MatrixXd eval=es.eigenvalues();                                /* returns eigenvalues in the form of a column vector sorted in a non-decreasing order */
	MatrixXd evec=es.eigenvectors();                               /* returns eigenvectors corresponding to the eigenvalues where one column is equal to one eigenvector */
	cout<<"Enter the value of k (<=dimentions i.e. "<<n<<") : ";
	int k;                                                         /* number of dimentions you want to keep in your decomposition */
	cin>>k;
	MatrixXd featureValues(k,1),featureVector(n,k);                /* featureValues is a row matrix which will store eigen values in each column in a non-increasing order */
																   /* featureVector will contain eigen vectors correcponding to the order of eigenvalues in featureValues */
	
	/*********** Computation of featureVector and featureValues ****************/
	for(i=n-1;i>=n-k;i--)
	{
		featureVector.col(n-i-1)=evec.col(i);
		featureValues(n-i-1,0)=eval(i);
	}

	/*********** Computation of transformed points ***********************/
	MatrixXd finalData(total,n);
	finalData=featureVector.transpose()*pointAdjust.transpose();    /* Calculation of new points */
	cout<<"_______________________________________\nTRANSFORMED POINTS : \n";
	cout<<finalData.transpose()<<endl;
	cout<<"_______________________________________\n";


	/************* Getting back the origional points  ********************/
	MatrixXd rowOriginalData(n,total);
	rowOriginalData=featureVector*finalData;
	for(i=0;i<total;i++)
	{
		for(j=0;j<n;j++)
		{
			rowOriginalData(j,i)+=mean[j];
		}
	}
	cout<<"\n_______________________________________\nRECONSTRUCTED POINTS (precision could be lost if k<n)"<<endl;
	cout<<rowOriginalData.transpose()<<endl;
	cout<<"_______________________________________\n";
}