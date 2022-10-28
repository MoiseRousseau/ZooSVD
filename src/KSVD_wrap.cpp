#include <mpi.h>
#include <string>


extern "C" {
    void wrap_KSVD(double*, const long int[2], double*, double*, double*, char*);
}


/*
Eigen Two-sided Jacobi SVD for double type
*/
void wrap_KSVD(
    double* mat_val, const long int shape[2], double* U, double* s, double* VT, char* executable
) {
    //get number of processes requested by the user
    int n_processes = 1;
    char *val = std::getenv("ZOOSVD_MPI_PROCESSES");
    if (val != NULL) n_processes = std::atoi(val);
    std::cout << "Call KSVD with " << n_processes << " MPI processes" << std::endl;

    //if there is more than 1 processes required, treat SVD in parallel
    if (n_processes > 1)
    {
        //prepare spawn
        int retcode;
        MPI_Init(NULL, NULL);
        MPI_Comm child_comm;
        int error[10];
        MPI_Info mpi_info;
        MPI_Info_create(&mpi_info);
        //spawn MPI processes
        retcode = MPI_Comm_spawn(executable, nullptr, n_processes-1, mpi_info, 0, MPI_COMM_SELF, &child_comm, error);
        if (retcode != MPI_SUCCESS) {
            std::cout << "Error when spawning the child MPI processes" << std::endl;
            std::cout << "MPI error code: " << error << std::endl;
        }
        //Send array to MPI processes
        
    }
    //do part of the job on this process too
    std::cout << "I'm the parent" << std::endl;
    
    //collect results and finish
    if (n_processes > 1)
    {
        MPI_Finalize();
    }
}
