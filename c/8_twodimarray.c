#include <stdio.h>



int main(){
    // 2차원 배열의 메모리도 1차원으로 존재함
    // 그래서 int형이니까 빈 공간은(예: 1행 2열) 0으로 초기화됨.
    int arr[3][3] = {
        {1}, 
        {4, 5}, 
        {7, 8, 9}
    };
    // works
    int arr1[3][3] = {
        1,2,3,4,5,6,7,8,9
    };
    // works
    int arr2[][2] = {
        1,2,3,4,5,6,7,8
    };
    // error 배열의 세로길이만 생략 가능
    // int arr2[2][] = {
    //     1,2,3,4,5,6,7,8
    // };
    // error
    // int arr3[][] = {
    //     1,2,3,4,5,6,7,8,9
    // };


    
    // problem2
    int arr4[2][4] = {
        1,2,3,4,5,6,7,8
    };
    int arr5[4][2];
    // 오 책에는 없지만 이렇게 해야 하네 별 하나만으로는 안 됨. 
    **arr5 = **arr4;
    printf("%p %d\n", arr4, **arr4);
    printf("%p %d\n", arr4, *(arr4[1]));   
    printf("%p %d\n", arr5, **arr5);
    printf("%p\n", *arr4); // 2-dim array는 포인터로, 그 값은 1d array(포인터).

    
    // double pointer. 그 값은 포인터 변수의 위치(주소)가 됨
    // 즉 포인터가 변수의 위치를 받는 변수라고 한다면,
    // 즉 **는 포인터의 위치를 받는 변수임. 
    int **dptr; 
    int *ptr;
    int var = 0;
    
    ptr = &var;
    // * 하나는 ptr주소가 되어야 함. 
    *dptr = ptr;
    // *dptr = var; // error


    // the below two are conceptually different
    // length 2 array which get (2) int pointer
    // [포인터 2 개를 받는 array](포인터)이므로 더블 포인터임.
    int* arr6[2]; 
    // a pointer whose unit is int*2
    int (*arr7)[2]; // equal to "int arr7[][2]"



    // 다음 행이 출력됨. 즉 열의 개수에 따라서 포인터형(혹은 단위 바이트 사이즈)이 달라짐
    // 2차원 배열의 포인터 형은 배열의 크기에 dependent하다. 
    printf("%p %p\n", arr4, arr4+1); 
    int (*ptr1) [2]; // int*2가 한 단위인 포인터 변수 선언
    // int ptr[][2]; // 동일한 코드
    
    // 2차원 배열을 인자로 받을 떄 
    void func(int **args){}; // wrong
    void func2(int * arts[2]){}; // correct. 단위 바이트를 규정해 줘야 함
    void func4(int *args){}; // 포인터 배열을 받을 때의 인자. 단위 바이트는 int * 1

    // TFAE
    arr5[2][1] = 4;
    (*(arr5+2))[1] = 4;
    *(arr[2]+1) = 4;
    *(*(arr+2)+1)=4;




}