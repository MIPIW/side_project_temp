#include <stdio.h>

int main(){
    // 구 c에서는 배열 길이를 상수로 지정하는 게 표준. 
    int arr[4];
    // 할당되지 않은 메모리 공간에 선언. 경우에 따라 문제가 발견되지 않을 수 있다. 
    arr[5] = 1;

    long int arr1[3] = {1,2}; // 세 번째 값은 0(전역변수 초기화처럼)

    printf("%d\n", sizeof arr1);

    // char str = "good morning"; // error char gets only one character
    // printf("%c", str[2]); // error

    // 자동으로 배열 길이를 계산. 문장 끝 \0(null char, ascii: 0)까지 더해서 계산
    // 아 내 컴파일러는 자동으로 길이 계산이 안 된다 야 -> 어 되네? 
    // \0이 있는 데에서 잘라서 출력 
    char str1[] = "Good morn\0ing!";
    printf("%s\n", str1); 

    char str2[50];
    int int2;
    // 입력하는 게 귀찮아서 일단 스킵
    // scanf는 띄어쓰기 기준으로 입력을 나눔. 문장을 입력받기 적절하지 않음.
    // scanf("%s", str2);
    // scanf("%d", &int2); // 변수는 & 붙여줘야 하는데 string list는 안 붙임


}