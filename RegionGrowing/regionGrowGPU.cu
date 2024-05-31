#include "regionGrowGPU.cuh"

// Definisco il numero di pixel per ogni blocco, il numero di pixel per ogni thread all'interno di un blocco
// Poichè i blocchi e i thread considereranno regioni quadrate dell'immagine, viene definito un solo valore.
// un blocco considererà quindi una regione di dimensione [ NUM_PIXEL_PER_BLOCK x NUM_PIXEL_PER_BLOCK ] (analogamente per i thread).
// Definiamo anche il numero di thread in ogni blocco, sia sulle righe che sulle colonne.
#define NUM_PIXEL_PER_BLOCK 32
#define NUM_PIXEL_PER_THREAD 16
#define NUM_THREAD_PER_BLOCK_X 2
#define NUM_THREAD_PER_BLOCK_Y 2


// Funzione per l'accrescimento delle regioni, utilizzando anche la Shared Memory sul Device
// Le porzioni di codice commentate riguardano dettagli implementativi per l'esecuzione du Device CUDA.
// Il funzionamento dell'algoritmo segue la stessa struttura generica del corrispettivo sequenziale.
__global__ void growRegionsGPU_SHM(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r,  int rows, int cols, int step_size, int point_i, int point_j , int soglia)
{
    // Per evitare continui accessi ai valori BGR del pixel Seed dalla Global Memory
    // salvo localmente la tripla corrispondente.
    unsigned char point_b, point_g, point_r;
    point_b = source_channel_b[point_i*cols + point_j];
    point_g = source_channel_g[point_i*cols + point_j];
    point_r = source_channel_r[point_i*cols + point_j];

    // Calcolo per ogni blocco l'indice iniziale sia per le righe che per le colonne
    // Il valore corrispondente è ottenuto dividendo il numero di righe (o colonne) per il numero di blocchi sulle righe (o colonne) della griglia dei blocchi
    // Prendendone il ceil() e moltiplicandolo per il corrispettivo indice del blocco sulle righe (o sulle colonne)
    // Es. consideriamo una matrice [211 x 471] la corrispettiva griglia dei blocchi è di [ 7 x 15]
    // il valore di Starting_row e Starting_col per l'ultimo blocco con indice [6,14] sarà
    // St_r = ceil(211.0/7.0)*6 = 31*6 = 186  e St_c = ceil(471.0/15.0)*14 = 32*14 = 448
    // Come è facilmente intuibile da questo esempio, questo blocco avrà un range di indici che eccede quelli dell'immagine stessa
    int starting_row = (int)ceil((float)rows/(float)gridDim.y)*blockIdx.y;
    int starting_col = (int)ceil((float)cols/(float)gridDim.x)*blockIdx.x;
    
    // Calcoliamo gli indici iniziali e finali di competenza di ciascun thread a partire dai valori calcolati in precedenza per il proprio blocco.
    // Essendo i thread, per come abbiamo definito, due sulle righe e due sulle colonne i loro indici varieranno da 0 ad 1.
    // Questi valori sono calcolati seguendo un ragionamento simile a quello fatto in precedenza.
    // Poichè sono stati definiti due Thread sulle righe e due sulle colonne, il rapporto tra NUM_PIXEL_PER_BLOCK e NUM_THREAD_PER_BLOCK_*
    // Da come valore NUM_PIXEL_PER_THREAD, ma poichè teoricamente questa configurazione potrebbe anche essere cambiata, ho lasciato
    // questa definizione.
    // Seguendo l'esempio precedente, per il thread [0,0] nel blocco, i valori di queste variabili saranno
    // th_st_r = 16*0 + 186      th_end_r = 16*0 + 186 + 16 - 1     th_st_c = 16*0 + 448   th_end_c = 16*0 + 448 + 16 - 1
    // e quindi considererà gli elementi dall'indice 186 a 201 sulle righe e dall'indice 448 a 463
    // Se consideriamo per questo esempio il Thread [1,1] i valori saranno
    // th_st_r = 16*1 + 186      th_end_r = 16*1 + 186 + 16 - 1     th_st_c = 16*1 + 448   th_end_c = 16*1 + 448 + 16 - 1
    // e quindi considererà gli elementi dall'indice 202 a 217 sulle righe e dall'indice 464 a 479
    // eccedendo i limiti dell'immagine. Questo problema viene risolto in seguito modificando i valori finali con gli ultimi indici reali dell'immagine (rows-1 e cols-1) 
    int thread_start_row = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y))*threadIdx.y + starting_row;
    int thread_end_row = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y))*threadIdx.y + starting_row + ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y)) - 1;
    int thread_start_col = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X))*threadIdx.x + starting_col;
    int thread_end_col = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X))*threadIdx.x + starting_col + ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X)) - 1;


    // Creazione delle porzioni di Shared Memory per questo blocco
    // Poichè un blocco considera delle porzioni [NUM_PIXEL_PER_BLOCK x NUM_PIXEL_PER_BLOCK]
    // allo stesso modo queste porzioni di memoria avranno le stesse dimensioni
    __shared__ unsigned char shared_B[NUM_PIXEL_PER_BLOCK][NUM_PIXEL_PER_BLOCK];
    __shared__ unsigned char shared_G[NUM_PIXEL_PER_BLOCK][NUM_PIXEL_PER_BLOCK];
    __shared__ unsigned char shared_R[NUM_PIXEL_PER_BLOCK][NUM_PIXEL_PER_BLOCK];
    __shared__ unsigned char shared_mask[NUM_PIXEL_PER_BLOCK][NUM_PIXEL_PER_BLOCK];

    // Poichè le immagini di input non vengono ridimensionate a dimensioni utili per consentire una
    // Suddivisione precisa con il numero di blocchi, potrebbe capitare che qualche thread di qualche blocco
    // Riceva indici iniziali e finali che già eccedono le dimensioni reali della immagine.
    // Per questi thread non è prevista alcuna elaborazione, per cui solo i thread che hanno almeno degli elementi
    // all'interno dell'immagine procederanno con l'elaborazione.
    if(thread_start_col < cols && thread_start_row < rows)
    {
        // Come detto in precedenza negli esempi, un thread potrebbe aver calcolato degli indici
        // finali che eccedono le dimensioni effettive dell'immagine, questi valori vengono modificati
        // affinchè non causino degli errori a runtime.
        if(thread_end_col >= cols)
        {
            thread_end_col = cols -1;
        }
        if(thread_end_row >= rows)
        {
            thread_end_row = rows -1;
        }

        // Faccio riempire la SM solo al thread [0,0] di ogni blocco
        // riempie la memoria shared da quella globale
        //if(threadIdx.x == 0 && threadIdx.y == 0)
        //{
        //    for(int i = 0; i < NUM_PIXEL_PER_BLOCK; i++)
        //    {
        //        for(int j = 0; j < NUM_PIXEL_PER_BLOCK; j++)
        //        {
        //            int row_index = starting_row + i;
        //            int column_index = starting_col + j;
        //            shared_B[i][j] = source_channel_b[row_index*cols + column_index];
        //            shared_G[i][j] = source_channel_g[row_index*cols + column_index];
        //            shared_R[i][j] = source_channel_r[row_index*cols + column_index];
        //            shared_mask[i][j] = matrix[row_index*cols + column_index]; //dovrebbe essere zero
        //        }
        //    }
        //}
        //__syncthreads();

        // Tutti i thread del blocco riempiono la propria porzione di SM dalle matrici nella memoria globale
        for(int i = thread_start_row; i <= thread_end_row; i++)
        {
            for(int j = thread_start_col; j <= thread_end_col; j++)
            {
                // Effettuando un operazione di divisione in modulo tra l'indice sulla matrice ed il valore
                // NUM_PIXEL_PER_BLOCK otteniamo sempre dei valori in [ 0 , NUM_PIXEL_PER_BLOCK -1] che 
                // corrispondono ai range di indici sulle porzioni di SM
                shared_B[i%NUM_PIXEL_PER_BLOCK][j%NUM_PIXEL_PER_BLOCK] = source_channel_b[i*cols + j];
                shared_G[i%NUM_PIXEL_PER_BLOCK][j%NUM_PIXEL_PER_BLOCK] = source_channel_g[i*cols +j];
                shared_R[i%NUM_PIXEL_PER_BLOCK][j%NUM_PIXEL_PER_BLOCK] = source_channel_r[i*cols + j];
                shared_mask[i%NUM_PIXEL_PER_BLOCK][j%NUM_PIXEL_PER_BLOCK] = matrix[i*cols + j];
            }
        }
        __syncthreads();

        //Poichè un Thread potrà visitare al massimo tutte le posizioni all'interno della sua porzione di immagine
        // I suoi stack per le coordinate avranno una dimensione pari a [NUM_PIXEL_PER_THREAD x NUM_PIXEL_PER_THREAD]
        int stack_r[NUM_PIXEL_PER_THREAD*NUM_PIXEL_PER_THREAD];
        int stack_c[NUM_PIXEL_PER_THREAD*NUM_PIXEL_PER_THREAD];

        // Analogamente alla funzione sequenziale, effettuiamo l'accrescimento delle regioni utilizzando uno step_size
        // in modo tale da poter aggiungere anche regioni che non sono necessariamente connesse alle prime che vengono trovate
        // e che quindi sono lontane tra loro
        for(int k = thread_start_row; k <= thread_end_row; k+=step_size)
        {
            for(int l = thread_start_col; l <= thread_end_col; l+=step_size)
            {   
                
                int index_r = 0;
                int index_c = 0;

                stack_r[index_r] = k;
                stack_c[index_c] = l;

                while(index_r > -1 && index_c > -1)
	            {
                    int temp_r = stack_r[index_r];
                    int temp_c = stack_c[index_c];
                    index_r--;
                    index_c--;

		            for(short i = -1; i <= 1; i++)
		            {
    			        for(short j = -1; j <= 1; j++)
			            {
    				        if(temp_r + i >= thread_start_row && temp_c + j >= thread_start_col && temp_r + i <= thread_end_row && temp_c + j <= thread_end_col)
				            {   
                                // Otteniamo i corrispettivi indici da utilizzare all'interno delle porzioni di SM 
                                int converted_i = (temp_r + i)%NUM_PIXEL_PER_BLOCK;
                                int converted_j = (temp_c + j)%NUM_PIXEL_PER_BLOCK;

					            if(shared_mask[converted_i][converted_j] == 0)
					            {
                                    // Poichè richiamare funzioni esterne all'interno di un kernel è complesso,
                                    // Calcoliamo "manualmente" la distanza euclidea
                                    int b_diff = shared_B[converted_i][converted_j] - point_b;
	                                int g_diff = shared_G[converted_i][converted_j] - point_g;
	                                int r_diff = shared_R[converted_i][converted_j] - point_r;

	                                int dist_euclid = (int)sqrt((float)(b_diff*b_diff + g_diff*g_diff + r_diff*r_diff));
    
						            if( dist_euclid <= soglia)
						            {
    							        shared_mask[converted_i][converted_j] = 1;
                                        index_r++;
                                        index_c++;
                                        stack_r[index_r] = temp_r + i;
                                        stack_c[index_c] = temp_c + j;
						            }
					            }
				            }
			            }
		            }
	            }
            }
        }

        __syncthreads();

        // Dalla SM andiamo ad inserire i valori ottenuti all'interno della matrice maschera che si trova nella 
        // Global memory sul Device.
        
        for(int k = thread_start_row; k <= thread_end_row; k++)
        {
            for(int l = thread_start_col; l <= thread_end_col; l++)
            { 
                matrix[k*cols + l] = shared_mask[k%NUM_PIXEL_PER_BLOCK][l%NUM_PIXEL_PER_BLOCK];
            }
        }

        __syncthreads();
    }
}

// Versione del Kernel per l'accrescimento delle regioni che non fa uso della Shared Memory.
// Segue lo stesso schema precedente, con la differenza che vengono fatti accessi diretti alla memoria globale sul Device
__global__ void growRegionsGPU(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r,  int rows, int cols, int step_size, int point_i, int point_j , int soglia)
{
    unsigned char point_b, point_g, point_r;
    point_b = source_channel_b[point_i*cols + point_j];
    point_g = source_channel_g[point_i*cols + point_j];
    point_r = source_channel_r[point_i*cols + point_j];

    int starting_row = (int)ceil((float)rows/(float)gridDim.y)*blockIdx.y;
    int starting_col = (int)ceil((float)cols/(float)gridDim.x)*blockIdx.x;
    
    int thread_start_row = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y))*threadIdx.y + starting_row;
    int thread_end_row = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y))*threadIdx.y + starting_row + ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y)) - 1;
    int thread_start_col = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X))*threadIdx.x + starting_col;
    int thread_end_col = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X))*threadIdx.x + starting_col + ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X)) - 1;

    if(thread_start_col < cols && thread_start_row < rows)
    {
        if(thread_end_col >= cols)
        {
            thread_end_col = cols -1;
        }
        if(thread_end_row >= rows)
        {
            thread_end_row = rows -1;
        }

        int stack_r[NUM_PIXEL_PER_THREAD*NUM_PIXEL_PER_THREAD];
        int stack_c[NUM_PIXEL_PER_THREAD*NUM_PIXEL_PER_THREAD];

        for(int k = thread_start_row; k <= thread_end_row; k+=step_size)
        {
            for(int l = thread_start_col; l <= thread_end_col; l+=step_size)
            {   
                
                int index_r = 0;
                int index_c = 0;

                stack_r[index_r] = k;
                stack_c[index_c] = l;

                while(index_r > -1 && index_c > -1)
	            {
                    int temp_r = stack_r[index_r];
                    int temp_c = stack_c[index_c];
                    index_r--;
                    index_c--;

		            for(short i = -1; i <= 1; i++)
		            {
    			        for(short j = -1; j <= 1; j++)
			            {
    				        if(temp_r + i >= thread_start_row && temp_c + j >= thread_start_col && temp_r + i <= thread_end_row && temp_c + j <= thread_end_col)
				            {   
                                int mat_ind_temp = (temp_r + i)*cols + (temp_c + j);

					            if(matrix[mat_ind_temp] == 0)
					            {
                                    int b_diff = source_channel_b[mat_ind_temp] - point_b;
	                                int g_diff = source_channel_g[mat_ind_temp] - point_g;
	                                int r_diff = source_channel_r[mat_ind_temp] - point_r;

	                                int dist_euclid = (int)sqrt((float)(b_diff*b_diff + g_diff*g_diff + r_diff*r_diff));
    
						            if( dist_euclid <= soglia)
						            {
    							        matrix[mat_ind_temp] = 1;
                                        index_r++;
                                        index_c++;
                                        stack_r[index_r] = temp_r + i;
                                        stack_c[index_c] = temp_c + j;
						            }
					            }
				            }
			            }
		            }
	            }
            }
        }

        __syncthreads();
    }
}

// Funzione per la ricolorazione dei canali BGR delle matrici di output a partire dalla Maschera e dai canali BGR dell'immagine di input.
__global__ void ricolorazioneGPU(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r, unsigned char* out_channel_b, unsigned char* out_channel_g, unsigned char* out_channel_r, int rows, int cols)
{
    // Per ogni thread vengono calcolati i range di indici di sua competenza
    // seguendo lo stesso schema incontrato nelle funzioni precedenti

    int starting_row = (int)ceil((float)rows/(float)gridDim.y)*blockIdx.y;
    int starting_col = (int)ceil((float)cols/(float)gridDim.x)*blockIdx.x;
    
    int thread_start_row = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y))*threadIdx.y + starting_row;
    int thread_end_row = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y))*threadIdx.y + starting_row + ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_Y)) - 1;
    int thread_start_col = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X))*threadIdx.x + starting_col;
    int thread_end_col = ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X))*threadIdx.x + starting_col + ((int)(NUM_PIXEL_PER_BLOCK/NUM_THREAD_PER_BLOCK_X)) - 1;

    if(thread_start_col < cols && thread_start_row < rows)
    {
        if(thread_end_col >= cols)
        {
            thread_end_col = cols -1;
        }
        if(thread_end_row >= rows)
        {
            thread_end_row = rows -1;
        }

        for(int i = thread_start_row; i <= thread_end_row; i++)
        {
            for(int j = thread_start_col; j <= thread_end_col; j++)
            {   
                // Se il corrispettivo valore nella matrice maschera è posto ad 1
                // vuol dire che il pixel appartiene alle regioni trovate e quindi
                // il valore del pixel nelle matrici dei canali di outpu viene posto
                // al corrispettivo valore dell'immagine di input
                if(matrix[i*cols + j] == 1)
                {
                    out_channel_b[i*cols + j] = source_channel_b[i*cols + j];
                    out_channel_g[i*cols + j] = source_channel_g[i*cols + j];
                    out_channel_r[i*cols + j] = source_channel_r[i*cols + j];
                }
                // Altrimenti il pixel viene posto al colore Nero
                else
                {
                    out_channel_b[i*cols + j] = 0;
                    out_channel_g[i*cols + j] = 0;
                    out_channel_r[i*cols + j] = 0;
                }
            }
        }
        __syncthreads();
    }
}


// Funzione per l'esecuzione dell'algoritmo Region Growing su GPU.
// Diversamente dalla funzione sequenziale eseguita sulla CPU ci sono tre matrici ulteriori che corrisponderanno ai canali BGR del''immagine di output
// ottenuta con la ricolorazione.
void regionGrowingGPU(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r, unsigned char* out_channel_b, unsigned char* out_channel_g, unsigned char* out_channel_r, int rows, int cols, int step_size, int point_i, int point_j , int soglia)
{
    unsigned char *source_channel_b_device, *source_channel_g_device, *source_channel_r_device, *out_channel_b_device, *out_channel_g_device, *out_channel_r_device, *matrix_device;
    
    // Alloco sul Device gli spazi di memoria necessari per poter contenere tutte le matrici che verranno copiate dall'Host
    // oppure su cui si andrà a scrivere durante l'esecuzione dei thread
    // Vengono allocate matrici per i tre canali BGR di input, i tre canali BGR di output e la matrice che rappresenta la maschera

    cudaMalloc((void**)&source_channel_b_device, rows*cols*sizeof(unsigned char));
    cudaMalloc((void**)&source_channel_g_device, rows*cols*sizeof(unsigned char));    
    cudaMalloc((void**)&source_channel_r_device, rows*cols*sizeof(unsigned char));
    cudaMalloc((void**)&out_channel_b_device, rows*cols*sizeof(unsigned char));
    cudaMalloc((void**)&out_channel_g_device, rows*cols*sizeof(unsigned char));    
    cudaMalloc((void**)&out_channel_r_device, rows*cols*sizeof(unsigned char));
    cudaMalloc((void**)&matrix_device, rows*cols*sizeof(unsigned char));

    // Copiamo dall'Host al Device le matrici date in input in tutte le matrici allocate in precedenza.

    cudaMemcpy(source_channel_b_device, source_channel_b , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(source_channel_g_device, source_channel_g , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(source_channel_r_device, source_channel_r , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(out_channel_b_device, out_channel_b , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(out_channel_g_device, out_channel_g , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(out_channel_r_device, out_channel_r , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_device, matrix , rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    //Dichiaro le variabili che serviranno per richiamare il Kernel sulla GPU, utilizzando nBlocks blocchi e nThreadsPerBlock thread all'interno di ciascun blocco.
    dim3 nBlocks, nThreadsPerBlock;
    nThreadsPerBlock.x = NUM_THREAD_PER_BLOCK_X;
    nThreadsPerBlock.y = NUM_THREAD_PER_BLOCK_Y;
	
    //Il numero di blocchi da utilizzare sulle righe e sulle colonne della griglia
    // viene calcolato come il minimo più grande valore intero maggiore
    // del rapporto tra il numero di righe/colonne diviso il numero di pixel
    // Viene fatto questo per poter gestire immagini di qualsiasi dimensione, 
    // che non sia obbligatoriamente un multiplo di 32 o una potenza di due
    nBlocks.x = (int)ceil((float)cols/(float)NUM_PIXEL_PER_BLOCK);
	nBlocks.y = (int)ceil((float)rows/(float)NUM_PIXEL_PER_BLOCK);

    //cudaEvent_t start, stop;
	//float time;

	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
    //cudaEventRecord(start, 0);

    // Richiamo del Kernel da eseguire sulla GPU, dando come input le matrici allocate sul Device, oltre che ai valori analoghi alla versione sequenziale
    // Viene richiamato il Kernel che fa utilizzo della Shared Memory sul Device

    growRegionsGPU<<<nBlocks, nThreadsPerBlock>>>(matrix_device, source_channel_b_device, source_channel_g_device, source_channel_r_device, rows, cols, step_size, point_i, point_j, soglia);
    
    // Ci assicuriamo che tutti i thread abbiano terminato l'esecuzione della propria parte del kernel e che quindi si sincronizzino.
    cudaDeviceSynchronize();

    // Richiamo del Kernel per la colorazione dei tre canali BGR, già allocati sul Device, dando in input
    // La maschera ottenuta dal Kernel precedente e le matrici per i tre canali sia dell'immagine data in input da cui prendere i valori, sia i tre canali dell'immagine
    // che verrà data in output. 
    ricolorazioneGPU<<<nBlocks, nThreadsPerBlock>>>(matrix_device, source_channel_b_device, source_channel_g_device, source_channel_r_device, out_channel_b_device, out_channel_g_device, out_channel_r_device, rows, cols);
    cudaDeviceSynchronize();

    // Ottenuti i tre canali di output dell'immagine risultato li ricopiamo dal Device all'Host, per poterli poi convertire in un oggetto Mat e salvare l'immagine
    // come file
    cudaMemcpy(out_channel_b, out_channel_b_device, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_channel_g, out_channel_g_device, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_channel_r, out_channel_r_device, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    //// memorizzo il tempo di fine
	////cudaEventRecord(stop,0);
	////cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&time, start, stop);
	//printf("Tempo di esecuzione sul device (GPU) in millisecondi: %f \n", time);


    // Libero la memoria allocata in precedenza sul Device
    cudaFree(source_channel_b_device);
    cudaFree(source_channel_g_device);
    cudaFree(source_channel_r_device);
    cudaFree(out_channel_b_device);
    cudaFree(out_channel_g_device);
    cudaFree(out_channel_r_device);
    cudaFree(matrix_device);
}

