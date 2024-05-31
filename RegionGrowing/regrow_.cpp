#include "utility.h"
#include "regionGrowGPU.cuh"

// Funzione per l'accrescimento delle regioni.
// Gli ultimi due parametri rappresentano il punto da cui si accrescono le regioni considerandone l'intorno 3x3, oltre che allo stesso punto.
// Dato il punto posto a queste due coordinate si controlla il suo intorno e si pongono gli elementi della maschera ad 1 se posti a una distanza
// minore della soglia rispetto al colore dell'elemento Seed. Tutti i punti che soddisfano questa condizione vengono inseriti nello stack
// consentendo poi con considerare man mano gli intorni 3x3 di tutti i punti inseriti, accrescendo così le regioni.

//Viene ripetuto il processo di estrazione dell'elemento in cima allo stack
// e controllarne il valore insieme a quello degli
// Il valore BGR del pixel Seed dato dalle coordinate (i,j)
void grow_region(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r, int rows, int cols, int point_i, int point_j , int soglia, int k, int l)
{
	// Le coordinate dei punti che durante la ricerca degli intorni risultano essere a una distanza minore della soglia dal Seed
	// vengono inserite all'interno dei due stack per gli indici.
	// Questi due stack vengono definiti unsigned int, diversamente da unsigned char, perchè i valori degli indici di una immagine
	// possono facilmente superare il valore 255. 
	// Viene definita la dimensione di questi due vettori pari al numero totale di pixel dell'immagine, poichè non è conosciuto a priori
	// il numero di pixel che durante la ricerca verranno visitati e che rispettano la soglia (pensiamo idealmente a una immagine monocromatica
	// il cui colore è proprio quello del seed)
	unsigned int *stack_r = (unsigned int*)malloc(rows*cols*sizeof(unsigned int));
    	unsigned int *stack_c = (unsigned int*)malloc(rows*cols*sizeof(unsigned int));
    
	//Vengono inizializzati i due indici per la gestione dei due stack e vengono inserite al suo interno le coordinate del punto dato in input
    	int index_r = 0;
    	int index_c = 0;
	stack_r[index_r] = k;
    	stack_c[index_c] = l;

	//Fintanto che i due stack non si svuoteranno del tutto.
	// Durante la ricerca dei pixel delle regioni è molto plausibile che arrivati ad un certo punto
	// O tutti i punti dell'immagine sono stati visitati e il corrispettivo valore della maschera è stato posto ad 1 (nel caso di immagine monocromatica)
	// oppure sono stati visitati e marcati ad 1 tutti i pixel della regione circoscritta che si sta esaminando, non trovando ulteriori vicini
	// che rispettino la condizione di distanza.
	while(index_r > -1 && index_c > -1)
	{
		// viene "estratto" l'elemento in cima allo stack
		// e viene decrementato l'indice sui due array corrispondenti
        	int temp_r = stack_r[index_r];
        	int temp_c = stack_c[index_c];
        	index_r--;
        	index_c--;


		//Per considerare l'intorno 3x3 del pixel estratto dallo stack aggiungiamo a queste coordinate valori sulle righe
		// e sulle colonne che vanno da -1 ad 1.
		for(short i = -1; i <= 1; i++)
		{
			for(short j = -1; j <= 1; j++)
			{
				// Controlliamo che le coordinate del pixel nel vicinato non eccedano i limiti dell'immagine
				if(temp_r + i >= 0 && temp_c + j >= 0 && temp_r + i < rows && temp_c + j < cols)
				{   
					//calcoliamo che il valore della maschera in posizione (i,j) non sia già stato considerato in precedenza e posto ad uno
                    			int mat_ind_temp = (temp_r + i)*cols + (temp_c + j);
					if(matrix[mat_ind_temp] == 0)
					{
						// calcoliamo la distanza euclidea tra il valore del seed e quello del pixel nella posizione in attuale considerazione
						int dst = dist_euclid(source_channel_b[mat_ind_temp], source_channel_g[mat_ind_temp], source_channel_r[mat_ind_temp], source_channel_b[point_i*cols + point_j], source_channel_g[point_i*cols + point_j], source_channel_r[point_i*cols + point_j]);
						
						// Se la distanza soddisfa la soglia abbiamo trovato un nuovo pixel da aggiungere alla regione
						if( dst <= soglia)
						{
							// impostiamo il corrispettivo valore nella maschera ad 1
							matrix[mat_ind_temp] = 1;
							
							//incrementiamo gli indici sui due stack e inseriamo le coordinate di questo nuovo punto
							// per poter poi controllane l'intorno 3x3 nella prossima iterazione del while.
                            				index_r++;
                            				index_c++;
                            				stack_r[index_r] = temp_r + i;
                            				stack_c[index_c] = temp_c + j;

							// Anche se venissero inseriti piu volte gli stessi indici in iterazioni diverse, controllandone il corrispettivo
							// valore nella maschera che sarà già stato posto ad 1, non verranno inseriti ulteriormente ma solo estratti dagli
							// stack per procedere subito alla iterazione successiva del while.
						}
					}
				}
			}
		}
	}

	//Libero la memoria allocata dinamicamente per i due stack
    free(stack_r);
    free(stack_c);
}

// Funzione che implementa l'algoritmo Region Growing. Accresce le regioni intorno ai punti (i,j) richiamando con uno step di step_size
// la funzione grow_region
void region_growing_CPU(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r, int rows, int cols, int step_size, int point_x, int point_y , int soglia)
{
	for(int i = 0; i < rows; i+=step_size)
	{
		for(int j = 0; j < cols; j+=step_size)
		{
            		grow_region(matrix, source_channel_b, source_channel_g, source_channel_r, rows, cols, point_x, point_y, soglia, i, j);
		}
	}
}

//Funzione per la ricolorazione dell'immagine data la matrice della maschera delle regioni
// Viene clonata l'immagine Mat di input e successivamente tutti i pixel i cui valori della maschera sono
// posti a zero (ovvero non appartengono alle regioni trovate), il corripettivo colore viene posto a Nero
Mat ricolorazione(Mat source, unsigned char* matrix)
{
	Mat temp = source.clone();

	for(short i = 0; i < source.rows; i++)
	{
		for(short j = 0; j < source.cols; j++)
		{
			if(matrix[i*source.cols+j] != 1)
			{
				temp.at<Vec3b>(i,j) = Vec3b(0,0,0);
			}
		}
	}
	return temp;
}

int main(int argc, char* argv[])
{
	//Aprimamo l'immagine il cui path è stato dato in input, tramite OpenCV
	Mat img = imread(argv[1], IMREAD_COLOR);

	if(!img.data || argc != 5)
	{
		cout<<"Error on input or on image opening"<<endl;
        	cout<<"Usage ./regionGrow <Path Immagine> <soglia> <coordinata i del seed> <coordinata j del seed>"<<endl;
		exit(-1);
	}

	int soglia = atoi(argv[2]);

    	int cord_i = atoi(argv[3]);
    	int cord_j = atoi(argv[4]);

	if(cord_i > img.rows || cord_j > img.cols)
	{
		cout << "Seed coordinates out of image bound"<<endl;
		exit(-2);
	}

	//Creo per l'immagine di input e per l'immagine di output prodotta dalla funzione su GPU
	// tre matrici di unsigned char, non uso gli interi perchè i valori dei pixel per ogni canale vanno da 0 a 255
    	unsigned char *img_channel_b = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
    	unsigned char *img_channel_g = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
    	unsigned char *img_channel_r = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));

	unsigned char *out_channel_b = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
    	unsigned char *out_channel_g = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
    	unsigned char *out_channel_r = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));

	//Creo le due matrici che rappresentano la maschera per le regioni trovate
    	unsigned char *matrix = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));
	unsigned char *matrix_cpu = (unsigned char*)malloc(img.rows*img.cols*sizeof(unsigned char));

	//Riempio le tre matrici dei singoli canali dall'immagine Mat
    	from_Mat_to_Char(img, img_channel_b, img_channel_g, img_channel_r, img.rows, img.cols);

    	//inizializzo a zero le matrici che rappresentano le maschere per le regioni
	for(int i = 0; i < img.rows; i++)
	{
		for(int j = 0; j < img.cols; j++)
		{
			matrix[i*img.cols+j] = 0;
			matrix_cpu[i*img.cols+j] = 0;
		}
	}

	//Variabili CUDA per determinare il tempo di esecuzione su CPU in millisecondi
	cudaEvent_t start_cpu, stop_cpu;
	float time_cpu;
	cudaEventCreate(&start_cpu);
	cudaEventCreate(&stop_cpu);
    	cudaEventRecord(start_cpu, 0);

	//Effettuo una chiamata alla funzione di RegionGrowing sequenziale
	// Oltre alle matrici per la maschera e i tre canali BGR separati, do in input le dimensioni dell'immagine
	// lo step con cui richiamare la funzione di crescita delle regioni per coprire tutta l'immagine, la posizione
	// dell'elemento seed e la soglia utilizzata per aggiungere i pixel alle regioni simili al seed
    	region_growing_CPU(matrix_cpu, img_channel_b, img_channel_g, img_channel_r, img.rows, img.cols, 3, cord_i, cord_j, soglia);
	
	//Ottenuta la maschera di output, richiamo la funzione di ricolorazione per avere una immagine di output nera
	// dove i pixel colorati sono solo quelli appartenenti alle regioni trovate dall'algoritmo.
    	Mat ris_cpu = ricolorazione(img, matrix_cpu);

	cudaEventRecord(stop_cpu,0);
	cudaEventSynchronize(stop_cpu);
	cudaEventElapsedTime(&time_cpu, start_cpu, stop_cpu);
	cout << endl << "Tempo di esecuzione sull'host (CPU) in millisecondi: " << time_cpu << endl << endl;

    	imwrite("RegionGrowing_CPU.jpg", ris_cpu);
    

	//Variabili CUDA per determinare il tempo di esecuzione su GPU in millisecondi
	cudaEvent_t start_gpu, stop_gpu;
	float time_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
    	cudaEventRecord(start_gpu, 0);

	// Richiamo la funzione per eseguire l'algoritmo su GPU
	// Dando in input anche le matrici che rappresenteranno i tre canali della immagine di output
	regionGrowingGPU(matrix, img_channel_b, img_channel_g, img_channel_r, out_channel_b, out_channel_g, out_channel_r, img.rows, img.cols, 3, cord_i, cord_j, soglia);

	cudaEventRecord(stop_gpu,0);
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&time_gpu, start_gpu, stop_gpu);

	cout << endl << "Tempo di esecuzione sul device (GPU) in millisecondi: " << time_gpu << endl << endl;

	// ottenuti i tre canali della matrice di output li convertiamo in un oggetto Mat di OpenCV per poter
	// poi salvare l'immagine tramite la imwrite.
	Mat ris_gpu = img.clone();
	from_Char_to_Mat(ris_gpu, out_channel_b, out_channel_g, out_channel_r, img.rows, img.cols);

    	imwrite("RegionGrowing_GPU.jpg", ris_gpu);

	//Libero la memoria da tutte le matrici allocate dinamicamente con la malloc
    	free(matrix);
    	free(img_channel_b);
    	free(img_channel_g);
    	free(img_channel_r);
	free(out_channel_b);
    	free(out_channel_g);
    	free(out_channel_r);
	return 0;
}
