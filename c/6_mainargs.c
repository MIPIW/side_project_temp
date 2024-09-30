#include <stdio.h>

// argc: number of arg, argv: array of character pointers.
int main(int argc, char *argv[]){
    for(int i=0; i < argc; i++){
        printf("%d's arg is: %s \n", i, argv[i]);
    }

    // 이렇게도 표현할 수 있음
    // 리스트 맨 뒤에 NULL문자가 들어감. 
    int i=0;
    while(argv[i] != NULL){
        printf("%d's arg is: %s \n", i, argv[i++]);

    }
}