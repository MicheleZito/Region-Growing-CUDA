#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Funzione che ha come input un oggetto della classe Mat di OpenCV e tre matrici di unsigned char, una per ogni canale BGR, che sono anche di output.
// Riempie le matrici dei singoli canali con i corrispettivi valori dagli elementi di tipo Vec3b dell'immagine Mat
void from_Mat_to_Char(Mat img, unsigned char* out_channel_b, unsigned char* out_channel_g, unsigned char* out_channel_r, int rows, int cols)
{

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            out_channel_b[i*cols+j] = img.at<Vec3b>(i,j)[0];
            out_channel_g[i*cols+j] = img.at<Vec3b>(i,j)[1];
            out_channel_r[i*cols+j] = img.at<Vec3b>(i,j)[2];
        }
    }

}

// Funzione che, a partire da tre matrici di unsigned char che rappresentano i canali BGR separati di una immagine
// inserisce nelle corrispettive posizioni della immagine Mat di input/output un nuovo oggetto Vec3b con i corrispondenti valori dai tre canali
void from_Char_to_Mat(Mat &img, unsigned char* out_channel_b, unsigned char* out_channel_g, unsigned char* out_channel_r, int rows, int cols)
{

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
	{
            img.at<Vec3b>(i,j) = Vec3b(out_channel_b[i*cols+j], out_channel_g[i*cols+j], out_channel_r[i*cols+j]);
        }
    }

}

// Funzione che calcola la distanza euclidea tra due colori rappresentati dalle due triple di unsigned char
int dist_euclid(unsigned char first_b, unsigned char first_g, unsigned char first_r, unsigned char second_b, unsigned char second_g, unsigned char second_r)
{
	int b = first_b - second_b;
	int g = first_g - second_g;
	int r = first_r - second_r;

	int dist = (int)sqrt((float)(b*b + g*g + r*r));
    	return dist;
}
