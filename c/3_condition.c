#include <stdio.h>

int main(){
    int num1 = 10, num2 = 3;

    if(num1 > num2){
        printf("num1(%d) is larger than num2(%d)\n", num1, num2);
    }else{
        printf("num2(%d) is larger than num1(%d)\n", num2, num1);
    }


    (num1 > num2) ? printf("num1, %d\n", num1) : printf("num2, %d\n", num2) ;
    int num3 = (num1 > num2) ? num1 : num2 ;
    printf("%d \n", num3);


    int num4 = 2;
    switch(num3){ // case에 없으면 아무것도 실행하지 않음
        case 1: printf("the num is case 1: %d", num4); break; // 해당 코드 이후의 코드를 모두 실행
        case 2: printf("the num is case 2: %d", num4); break;
        case 3: printf("the num is case 3: %d", num4); break;
        case 4: printf("the num is case 4: %d", num4); break;

    }

    char val1 = "m";
    // error: case label does not reduce to an integer constant
    // 교과서에 있지만 오류
    // switch(val1){
    //     case "M": 
    //     case "m": printf("good morning!"); break;
    //     case "E": 
    //     case "e": printf("good evening!"); break;
    // }
}