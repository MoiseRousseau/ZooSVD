#include <mpi.h>

void child_KSVD() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "I'm the child, rank " << rank << std::endl;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    child_KSVD();
    MPI_Finalize();
    return 0;
}
