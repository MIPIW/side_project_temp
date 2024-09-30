#include <stdio.h>

// call by value
int func(int val);
// call by reference
// 함수는 인자를 복사(값의 주소를 복사), 
// 인자로 포인터를 넣으면 포인터의 위치를 가리키는 포인터가 생성된다. 
int func1(int * arr); // 배열을 인자로 넣고 싶을 때에는 배열의 포인터값을 넣는다
int func2(int arr[]); //배개변수의 선언에 한해 둘이 같은 코드. 자바스크립트랑 비슷한거네

int main(){
    int num = 7; // 일반 변수 정의
    int *point_num; // 포인터 변수 정의. 
    // 64비트 시스템은 주소값을 64비트로 저장하므로 포인터 변수 크기도 64비트(8byte)가 됨. 
    // int* pnum, int * pnum, int *pnum 다 가능

    point_num = &num; // num의 주소값(&)을 저장

    printf("%p\n", point_num);
    printf("%p\n", &num); // 동일한 값



    // 포인터 변수 앞의 자료형은 해당 공간부터 몇 바이트를 읽을 것인지를 알려주는 정보.
    // double *point_num1 = &num; // warning으로 작동하긴 하나 포인터 연산할 때 문제가 생김



    // 포인터 변수는 그 값이 메모리 공간의 주소임
    // 이렇게 출력하면 해당 메모리 공간의 값을 출력함
    int *point_num2 = &num; 
    printf("%d\n", --(*point_num2)); 
    printf("%p\n", point_num2);



    // 쓰레기 (주소)값으로 초기화하는데, 해당 주소의 값에 200을 넣음
    // danger!
    // int *pt = 200; 
    // int *pt; *pt = 200; 도 위험 
    // printf("%d, %d\n", pt, *pt);
    // 이렇게 초기화하는 것이 바람직
    int *pt = NULL; 



    // 배열의 변수이름은 배열의 첫 항의 주소를 가리키는 포인터이다. 
    // 그리고 해당 주소 값은 변경이 불가능함
    int arr[3] = {0,1,2};
    printf("%p = %p, %p\n", arr, &arr[0], &arr[1]);
    // 작동한다!
    int *pt1 = arr;
    printf("%d\n", ++*pt1);



    // 아래 네 개는 모두 동일!
    printf("%d %d\n", *(pt1+0), *(pt1+2));
    printf("%d %d\n", *(arr+0), *(arr+2));
    printf("%d %d\n", pt1[0], pt1[2]);
    printf("%d %d\n", arr[0], arr[2]);
    // 하지만 다음은 다름
    printf("%d\n", sizeof arr);
    printf("%d\n", sizeof arr[0]);
    printf("%p %p\n", arr, arr+1); // int이니까 4바이트 뒤에 있는 애가 주소로 출력됨




    // 상수 형태의 문자열(값 변경 불가, 포인터는 첫 번쨰 char만 가리킴)
    char *pt_str1 = "my string";
    // 변수 형태의 문자열(값 변경 가능)
    char str2[] = "your string";
    // str2[3] = "t"; // 가능
    // pt_str1[3] = "t"; // 컴파일은 되나 변경불가능



    // 포인터 배열
    // 포인터 각 항에는 첫 글자의 주소가 부여
    // str는 그 자체로 배열의 주소값
    char *pt_strArr[3] = {"my", "word", "three"}; 


    // const는 실수로 해당 값을 수정하는 것을 막아준다. 
    // *ptr1가 고정, num이라는 변수로는 접근 가능
    const int * ptr1 = &num;
    // *ptr1 = 30 // error!
    // ptr2가 고정, *ptr는 상관없음
    int * const ptr2 = &num;
    // ptr2 = &num2 // error!


    
}

