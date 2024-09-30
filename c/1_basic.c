#include <stdio.h>

// this is comment

/* 

this is multi-
line comment 

*/ 

int main() {
//    int type: char, short, int(long), long long vs float type: float, double, long double
    int num1 = 0, num2 = 0, num4 = 0; 
    num1 = 2;
    num2 = 0xA; // 16진수
    num4 = 067; // 8진수
    char num5 = 0b00000010; // 2진수로 2
    unsigned char num6 = 0b10000000; // MSB까지 숫자 표기에 이용됨
    char num7 = 65; // %c로 프린트하면 문자가 나옴
    const int MAX=100; // 이름을 가진 상수. 변경 불가
    
    int num8 = 3, num9 = 4;
    // int num3 = 2; // working but cautious
    // scanf("%d %d", &num1, &num2); // 절차지향적 특징(순서가 중요함)
    // num3 += num1; 

//    후위증가/후위감소는 다음 문장으로 넘어가야만 증감이 처리됨
//    곱셈과 나눗셈은 먼저 등장하는 순서대로 처리됨

//    double num3 = 3; // 두 번 정의되면 안 됨 
//    num3 += num1; // 아 double * int는 뭔가 이상하게 처리되는구나

//    이름 없이 저장되는 0 같은 객체를 리터럴이라고 함
//    정수 리터럴은 int, 실수 리터럴은 double로 자동으로 저장됨. 
//    바꿔주려면 float num8 = 5.789f; 이렇게.
    
    printf("%f \n", num8 / num9); // 정수형으로 계산된 다음에 float형으로 형변환
    printf("%f \n", (double)num8 / num9); // double로 형변환하니까 num9도 형변환됨. 그치만 명시해주는 게 좋음.
    printf("Hello, World!\n");
    printf("%-8s %14s %5s", "장효형", "언어학과", "열 살");
    printf("\n");
    printf("%d \n", ~--num5); // 1을 뺴고 반전하나 반전해서 1을 더하나 같음
    printf("%c %d \n", num7, num6<<1); // 1을 뺴고 반전하나 반전해서 1을 더하나 같음
    printf("%d", sizeof(num6));

    
    return 0;
}
