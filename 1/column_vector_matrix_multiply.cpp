#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "mpi.h"

int ProcNum = 8;      // Кількість доступних процесів
int ProcRank = 0;     // id поточного процесу


// Функція для випадкового визначення матричних та векторних елементів
void RandomDataInitialization(double *pMatrix, double *pVector, int Size) {
    int i, j;
    srand(unsigned(clock()));
    for (i = 0; i < Size; i++) {
        pVector[i] = rand() / double(1000);
        for (j = 0; j < Size; j++)
            pMatrix[j * Size + i] = rand() / double(1000);
    }
}

// Функція виділення пам'яті та ініціалізації даних
void ProcessInitialization(double *&pMatrix, double *&pVector, double *&pProcVector,
                           double *&pResult, double *&pProcColumns, double *&pProcResult,
                           int &Size, int &ColumnNum) {
    int RestColumns; // Кількість стовпців, які ще не розповсюджені
    int i;

    setvbuf(stdout, 0, _IONBF, 0);
    if (ProcRank == 0) {
        do {
            printf("\nEnter size of the initial objects: ");
            scanf("%d", &Size);
            if (Size < ProcNum) {
                printf("Size of the objects must be greater than number of processes! \n ");
            }
        } while (Size < ProcNum);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Визначаємо кількість рядків матриці, що зберігаються в кожному процесі
    RestColumns = Size;
    for (i = 0; i < ProcRank; i++)
        RestColumns = RestColumns - RestColumns / (ProcNum - i);
    ColumnNum = RestColumns / (ProcNum - ProcRank);

    // Виділення пам'яті
    pProcVector = new double[ColumnNum];
    pProcColumns = new double[ColumnNum * Size];
    pProcResult = new double[Size];

    // Отримати значення елементів початкових об'єктів
    if (ProcRank == 0) {
        // Початкова матриця існує тільки головному процесі
        pMatrix = new double[Size * Size];
        // Початковий вектор існує тільки в головному процесі
        pVector = new double[Size];
        pResult = new double[Size];
        // Значення елементів визначаються тільки в головному процесі
        RandomDataInitialization(pMatrix, pVector, Size);
    }
}

// Функція розподілу початкових об'єктів між процесами
void DataDistribution(double *pMatrix, double *pProcColumns, double *pVector, double *pProcVector,
                      int Size, int ColumnNum) {
    int *pSendNum; // Кількість елементів, надісланих до процесу
    int *pSendInd; // Індекс першого елемента даних, надісланого процесу
    int RestColumns = Size; // Кількість рядків, які ще не розподілені
    int *pSendVecNum; // Кількість елементів вектора, надісланих до процесу
    int *pSendVecInd; // Індекс першого елемента у векторі, надісланого процесу

    // Виділяє пам'ять для тимчасових об'єктів
    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];
    pSendVecInd = new int[ProcNum];
    pSendVecNum = new int[ProcNum];

    // Визначаємо розташування рядків матриці для поточного процесу
    ColumnNum = (Size / ProcNum);
    pSendNum[0] = ColumnNum * Size;
    pSendInd[0] = 0;
    pSendVecNum[0] = ColumnNum;
    pSendVecInd[0] = 0;
    for (int i = 1; i < ProcNum; i++) {
        RestColumns -= ColumnNum;
        ColumnNum = RestColumns / (ProcNum - i);
        pSendNum[i] = ColumnNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        pSendVecNum[i] = ColumnNum;
        pSendVecInd[i] = pSendVecInd[i - 1] + pSendVecNum[i - 1];
    }

    // Розкидаємо часткові вектори
    MPI_Scatterv(pVector, pSendVecNum, pSendVecInd, MPI_DOUBLE, pProcVector, pSendVecNum[ProcRank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    // Розкидаємо стовпчики
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcColumns,
                 pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Звільнимо пам'ять
    delete[] pSendNum;
    delete[] pSendInd;
    delete[] pSendVecNum;
    delete[] pSendVecInd;
}

// Реплікація вектора результату
void ResultReplication(double *pProcResult, double *pResult, int Size) {

    // Підсумуємо всі вектори procResult в один вектор pResult
    MPI_Reduce(pProcResult, pResult, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

}

// Функція послідовного множення матриці на вектор
void SerialResultCalculation(double *pMatrix, double *pVector, double *pResult, int Size) {
    int i, j;
    for (i = 0; i < Size; i++)
        pResult[i] = 0;
    for (j = 0; j < Size; j++) {
        for (i = 0; i < Size; i++)
            pResult[i] += pMatrix[j * Size + i] * pVector[j];
    }
}

// Обробити рядки та векторне множення
void ParallelResultCalculation(double *pProcColumns, double *pProcVector, double *pProcResult, int Size, int ColumnNum) {
    int i, j;
    for (i = 0; i < Size; i++)
        pProcResult[i] = 0;
    for (j = 0; j < ColumnNum; j++) {
        for (i = 0; i < Size; i++)
            pProcResult[i] += pProcColumns[j * Size + i] * pProcVector[j];
    }
}


// Перевірка результату паралельного множення матриці на вектор
void TestResult(double *pMatrix, double *pVector, double *pResult,
                int Size) {
    // Буфер для зберігання результату послідовного множення матриці на вектор
    double *pSerialResult;
    // Прапорець, який показує, ідентичні вектори чи ні
    int equal = 0;
    int i;

    if (ProcRank == 0) {
        pSerialResult = new double[Size];
        SerialResultCalculation(pMatrix, pVector, pSerialResult, Size);
        //PrintVector(pResult, Size);
        //printf("\n");
        //PrintVector(pSerialResult, Size);
        //printf("\n");
        for (i = 0; i < Size; i++) {
            if (fabs(pResult[i] - pSerialResult[i]) > 1e-6)
                equal = 1;
        }
        if (equal == 1)
            printf("The results of serial and parallel algorithms are NOT identical. Check your code.");
        else
            printf("The results of serial and parallel algorithms are identical.");

        delete[] pSerialResult;
    }
}

// Функція завершення обчислювального процесу
void ProcessTermination(double *pMatrix, double *pVector, double *pProcVector, double *pResult,
                        double *pProcColumns, double *pProcResult) {
    if (ProcRank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
    }
    delete[] pProcVector;
    delete[] pProcColumns;
    delete[] pProcResult;
}

int main(int argc, char *argv[]) {
    double *pMatrix;  // Перший аргумент - початкова матриця
    double *pVector;  // Другий аргумент - початковий вектор
    double *pProcVector; // Частковий вектор поточного процесу
    double *pResult;  // Вектор результату для множення матриці на вектор
    int Size;        // Розміри початкової матриці та вектора
    double *pProcColumns;   // Смуга матриці на поточному процесі
    double *pProcResult; // Блок вектора результату поточного процесу
    int ColumnNum;          // Кількість стовпців у смузі матриці
    double Duration, Start, Finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        printf("Parallel matrix-vector multiplication program\n");
    }

    // Виділення пам'яті та ініціалізація даних
    ProcessInitialization(pMatrix, pVector, pProcVector, pResult, pProcColumns, pProcResult,
                          Size, ColumnNum);

    Start = MPI_Wtime();


    // Розподіл початкових об'єктів між процесами
    DataDistribution(pMatrix, pProcColumns, pVector, pProcVector, Size, ColumnNum);

    //TestDistribution(pMatrix, pVector, pProcVector, pProcColumns, Size, ColumnNum);

    // Обробити рядки та векторне множення
    ParallelResultCalculation(pProcColumns, pProcVector, pProcResult, Size, ColumnNum);

    // Реплікація результату
    ResultReplication(pProcResult, pResult, Size);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    //TestPartialResults(pProcResult, Size);
    TestResult(pMatrix, pVector, pResult, Size);

    if (ProcRank == 0) {
        printf("\nTime of execution = %f s\n", Duration);
    }

    /*
    Size = 10000;
    pMatrix = new double[Size * Size];
    pVector = new double[Size];
    pResult = new double[Size];
    RandomDataInitialization(pMatrix, pVector, Size);
    time_t start, finish;

    start = clock();
    SerialResultCalculation(pMatrix, pVector, pResult, Size);
    finish = clock();

    double duration = (finish - start) / double(CLOCKS_PER_SEC);
    printf("\n Time of execution: %f", duration);
    */

    // Припинення процесу
    ProcessTermination(pMatrix, pVector, pProcVector, pResult, pProcColumns, pProcResult);

    MPI_Finalize();
}
