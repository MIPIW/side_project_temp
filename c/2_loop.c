#include <stdio.h>

int main() {

    int num1 = 0;
    do{
        printf("%d\n", num1++);
    } while(num1<3);

    while(num1 < 15){
        printf("-----%d\n", num1++);
    }

    printf("%d\n", num1);

    /////////// equal
    int num2 = 0, sum = 0;
    while(num2<100){
        num2 += 2;
        sum += num2;
        if(num2>50){
            break; // break and continue
        }
    }
    printf("sum1 %d\n", sum);

    num2 = 0, sum = 0;
    while(num2<100){
        num2 += 2;
        sum += num2;

    }
    printf("sum2 %d\n", sum);

    
    for(int num = 0; num <3; num++){
        printf("-----------%d\n", num);
    }

}
