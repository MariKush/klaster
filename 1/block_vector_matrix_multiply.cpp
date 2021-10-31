#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

int ProcNum = 8;      // Кількість доступних процесів
int ProcRank = 0;     // id поточного процесу

void PrintVector(double *pVector, int Size);

void PrintMatrix(double *pMatrix, int RowNum, int ColumnNum);

int determineMaxWidth(int Size) {
    for (int i = sqrt(Size); i >= 1; i--)
        if (Size % i == 0)
            return i;
}

// Функція простого визначення матричних та векторних елементів
void DummyDataInitialization(double *original, double *pMatrix, double *pVector, int Size, int h, int w) {
    int i, j;

    for (i = 0; i < Size; i++) {
        pVector[i] = 1;
        for (j = 0; j < Size; j++) {
            int ind = (i - (i % h)) * Size + (j - (j % w)) * h + (i % h) * w + (j % w);
            pMatrix[ind] = i;
            original[i * Size + j] = i;

        }
    }
    PrintMatrix(original, Size, Size); // виводить вхідні матрицю
    PrintMatrix(pMatrix, Size, Size); // показує перетворену матрицю
}

// Функція для випадкового визначення матричних та векторних елементів
void RandomDataInitialization(double *original, double *pMatrix, double *pVector, int Size, int h, int w) {
    int i, j;
    srand(unsigned(clock()));
    for (i = 0; i < Size; i++) {
        pVector[i] = rand() / double(1000);
        for (j = 0; j < Size; j++) {
            double randVal = rand() / double(1000);
            int ind = (i - (i % h)) * Size + (j - (j % w)) * h + (i % h) * w + (j % w);
            pMatrix[ind] = randVal;
            original[i * Size + j] = randVal;
        }
    }
}

// Функція виділення пам'яті та ініціалізації даних
void ProcessInitialization(double *&original, double *&pMatrix, double *&pVector, double *&pProcVector,
                           double *&pResult, double *&pProcMatrix, double *&pProcResult,
                           int &Size, int &RowNum, int &ColumnNum) {
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
    int s = determineMaxWidth(ProcNum), q = ProcNum / s;
    RowNum = Size / q, ColumnNum = Size / s;

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&RowNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ColumnNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Виділення пам'яті
    pProcVector = new double[ColumnNum];
    pProcMatrix = new double[RowNum * ColumnNum];
    pProcResult = new double[RowNum];


    // Отримати значення елементів початкових об'єктів
    if (ProcRank == 0) {
        // Початкова вихідна матриця
        original = new double[Size * Size];
        // Початкова матриця існує тільки в головному процесі
        pMatrix = new double[Size * Size];
        // Початковий вектор існує тільки в головному процесі
        pVector = new double[Size];
        pResult = new double[Size];
        // Значення елементів визначаються тільки в гоовному процесі
        RandomDataInitialization(original, pMatrix, pVector, Size, RowNum, ColumnNum);
    }
}

// Функція розподілу початкових об'єктів між процесами
void DataDistribution(double *pMatrix, double *pProcMatrix, double *pVector, double *pProcVector,
                      int Size, int RowNum, int ColumnNum) {
    int *pSendNum; // Кількість елементів, надісланих до процесу
    int *pSendInd; // Індекс першого елемента даних, надісланого процесу
    int *pSendVecNum; // Кількість елементів вектора, надісланих до процесу
    int *pSendVecInd; // Індекс першого елемента у векторі, надісланого процесу

    // Виділяє пам'ять для тимчасових об'єктів
    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];
    pSendVecInd = new int[ProcNum];
    pSendVecNum = new int[ProcNum];

    // Визначаємо розташування матриці та вектора для поточного процесу
    pSendNum[0] = RowNum * ColumnNum;
    pSendInd[0] = 0;
    pSendVecNum[0] = ColumnNum;
    pSendVecInd[0] = 0;
    for (int i = 1; i < ProcNum; i++) {

        pSendNum[i] = RowNum * ColumnNum;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        pSendVecNum[i] = ColumnNum;
        pSendVecInd[i] = (pSendVecInd[i - 1] + pSendVecNum[i - 1]) % Size;
    }

    //MPI_Scatter()
    // Розкидаємо часткові вектори
    MPI_Scatterv(pVector, pSendVecNum, pSendVecInd, MPI_DOUBLE, pProcVector, pSendVecNum[ProcRank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    // Розкидаємо стовпчики
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcMatrix,
                 pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Звільнимо пам'ть
    delete[] pSendNum;
    delete[] pSendInd;
    delete[] pSendVecNum;
    delete[] pSendVecInd;
}

// Реплікація вектора результату
void ResultReplication(double *pProcResult, double *pResult, int Size, int RowNum, int ColumnNum) {

    int posInRes = (ProcRank / (Size / ColumnNum)) * RowNum;
    double *allProcResult = new double[Size];
    for (int i = 0; i < Size; i++)
        allProcResult[i] = 0;
    for (int i = posInRes; i < posInRes + RowNum; i++) {
        allProcResult[i] = pProcResult[i - posInRes];
    }

    // Підсумуємо всі вектори procResult в один вектор pResult
    MPI_Reduce(allProcResult, pResult, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

// Функція послідовного множення матриці на вектор
void SerialResultCalculation(double *original, double *pVector, double *pResult, int Size) {
    int i, j;  // Loop variables
    for (i = 0; i < Size; i++) {
        pResult[i] = 0;
        for (j = 0; j < Size; j++)
            pResult[i] += original[i * Size + j] * pVector[j];
    }
}

void TestSerialResult() {
    int Size = 5;
    double *original = new double[Size * Size];
    double *pMatrix = new double[Size * Size];
    double *pVector = new double[Size];
    double *pResult = new double[Size];
    int w = determineMaxWidth(Size), h = Size / w;

    RandomDataInitialization(original, pMatrix, pVector, Size, h, w);

    printf("Matrix: \n");
    PrintMatrix(original, Size, Size);
    printf("Vector: \n");
    PrintVector(pVector, Size);

    SerialResultCalculation(original, pVector, pResult, Size);

    PrintVector(pResult, Size);

}

// Обробити рядки та векторне множення
void ParallelResultCalculation(double *pProcMatrix, double *pProcVector, double *pProcResult, int RowNum, int ColumnNum) {
    int i, j;
    for (i = 0; i < RowNum; i++)
        pProcResult[i] = 0;
    for (i = 0; i < RowNum; i++) {
        for (j = 0; j < ColumnNum; j++)
            pProcResult[i] += pProcMatrix[i * ColumnNum + j] * pProcVector[j];
    }
}

// Функція для виведення форматованої матриці
void PrintMatrix(double *pMatrix, int RowCount, int ColumnCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColumnCount; j++)
            printf("%7.4f ", pMatrix[i * RowCount + j]);
        printf("\n");
    }
}

// Функція форматованого векторного виводу
void PrintVector(double *pVector, int Size) {
    int i;
    for (i = 0; i < Size; i++)
        printf("%0.6f ", pVector[i]);
    printf("\n");
}

void TestDistribution(double *original, double *pVector, double *pProcVector, double *pProcMatrix,
                      int Size, int RowNum, int ColumnNum) {
    if (ProcRank == 0) {
        printf("Initial Matrix: \n");
        PrintMatrix(original, Size, Size);
        printf("Initial Vector: \n");
        PrintVector(pVector, Size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < ProcNum; i++) {
        if (ProcRank == i) {
            printf("\nProcRank = %d \n", ProcRank);
            printf(" Matrix Stripe:\n");
            PrintMatrix(pProcMatrix, RowNum, ColumnNum);
            printf(" Vector: \n");
            PrintVector(pProcVector, ColumnNum);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Функція перевірки результатів множення смуги матриці на вектор
void TestPartialResults(double *pProcResult, int RowNum) {
    int i;    // Loop variables
    for (i = 0; i < ProcNum; i++) {
        if (ProcRank == i) {
            printf("\nProcRank = %d \n Part of result vector: \n", ProcRank);
            PrintVector(pProcResult, RowNum);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Перевірка результату паралельного множення матриці на вектор
void TestResult(double *original, double *pVector, double *pResult,
                int Size) {
    // Буфер для зберігання результату послідовного множення матриці на вектор
    double *pSerialResult;
    // Прапорець, який показує, ідентичні вектори чи ні
    int equal = 0;
    int i;

    if (ProcRank == 0) {
        pSerialResult = new double[Size];
        SerialResultCalculation(original, pVector, pSerialResult, Size);
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
void ProcessTermination(double *original, double *pMatrix, double *pVector, double *pProcVector, double *pResult,
                        double *pProcMatrix, double *pProcResult) {
    if (ProcRank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
        delete[] original;
    }
    delete[] pProcVector;
    delete[] pProcResult;
    delete[] pProcMatrix;
}

int main(int argc, char *argv[]) {
    double *originalMatrix; // Перший аргумент - початкова матриця
    double *pMatrix;  // Перетворена матриця
    double *pVector;  // Другий аргумент - початковий вектор
    double *pProcVector; // Частковий вектор поточного процесу
    double *pResult;  // Вектор результату для множення матриці на вектор
    int Size;            // Розміри початкової матриці та вектора
    double *pProcMatrix;   // Смуга матриці на поточному процесі
    double *pProcResult; // Блок вектора результату поточного процесу
    int RowNum;          // Кількість рядків у блоці матриці
    int ColumnNum;          // Кількість стовпців у блоці матриці
    double Duration, Start, Finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);


    if (ProcRank == 0) {
        printf("Parallel matrix-vector multiplication program\n");
    }

    // Memory allocation and data initialization
    ProcessInitialization(originalMatrix, pMatrix, pVector, pProcVector, pResult, pProcMatrix, pProcResult,
                          Size, RowNum, ColumnNum);

    Start = MPI_Wtime();

    // Розподіл початкових об'єктів між процесами
    DataDistribution(pMatrix, pProcMatrix, pVector, pProcVector, Size, RowNum, ColumnNum);

    //TestDistribution(originalMatrix, pVector, pProcVector, pProcMatrix, Size, RowNum, ColumnNum);
    // Обробити рядки та векторне множення
    ParallelResultCalculation(pProcMatrix, pProcVector, pProcResult, RowNum, ColumnNum);

    // Реплікація результату
    ResultReplication(pProcResult, pResult, Size, RowNum, ColumnNum);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    //TestPartialResults(pProcResult, RowNum);
    TestResult(originalMatrix, pVector, pResult, Size);

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

    //TestSerialResult();
    // Припинення процесу
    ProcessTermination(originalMatrix, pMatrix, pVector, pProcVector, pResult, pProcMatrix, pProcResult);

    MPI_Finalize();
}