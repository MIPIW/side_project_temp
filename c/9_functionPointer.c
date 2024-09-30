#include <stdio.h>

void main(){
    // 함수도 포인터이다
    // 함수 포인터는 input 타입과 output 타입을 모두 표기. 
    int func6(int val){return 0;};
    int (*fptr) (int); // (int) argument 하나이고 int를 리턴하는 함수 포인터
    int (*fptr1) (int, int); // (int) argument 두 개이고 int를 리턴하는 함수 포인터
    fptr = func6; // correct
    // fptr = &func; // wrong
    
    // 그래서 함수를 감싸는 함수를 선언하려면
    void func7(int val, int (*funcs)(int)); // 값을 val에 넣고, 함수를 funcs에 넣는 함수

    // void pointer
    int num = 20;
    void * ptr;

    ptr = &num;
    // ptr++ unavailable as '+1' is not deceided such as array is
    // *ptr=20 unavailable as the data type is not defined.  


}
    