#include <stdio.h>
#include <stdlib.h>

void main(){


    // 힙 영역: 로컬 함수가 호출될 때마다 할당되고, 함수가 사라져고 유지되는 메모리.
    // 정확히는, 할당과 소멸을 프로그래머가 직접 조절할 수 있음. 
    // 포인터를 이용해서 접근할 수밖에 없음. 
    void * ptr1 = malloc(4); // 힙 영역에 4 byte 할당
    void free(void* ptr); // 4 byte 해제
    
    // 실질적으로 쓰려면 할당하고 형변환해서 사용해야 함
    int * ptr2 = (int *)malloc(4);    
    if(ptr2 ==NULL){ // 메모리 할당에 실패했다면 NULL값이 부여됨
        //script
    }

    // void * calloc(size_t elt_ount, size_t elt_size) elt_size(블록 크기) * elt_count(블록 개수)를 할당.
    // void * realloc(void * head_ptf, size_t size) 메모리 사이즈 재생성. 


}