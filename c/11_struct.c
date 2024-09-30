#include <stdio.h>
#include <math.h>

// double getDistance(struct point a, struct point b);

// 밖에 선언해야 밖에서 함수 만들 때 참조할 수 있음
struct point{
    int posx;
    int posy;
};


void main(){

    // struct data type defining
    // 'point' 가 하나의 자료형이 됨(user defined datatype)


    // array도 가능
    struct person{
        char name[20];
        char phoneNum[20];
        int age;
    };

    // struct instance defining
    struct point mouse1 = {1,2};
    // 혹은
    struct point mouse2;
    mouse2.posx = 3;
    mouse2.posy = 5;

    // struct array
    struct point mouses[3];
    // mouses[0] = {1,2}; //not possible
    // mouses[1] = {3,4};
    // mouses[2] = {5,6};

    mouses[0].posx = 1;
    mouses[0].posy = 2;
    mouses[1].posx = 3;
    mouses[1].posy = 4;
    mouses[2].posx = 5;
    mouses[2].posy = 6;
    
    struct point * pptr = &mouses[0];

    // TFAE
   (*pptr).posx = 10;
    pptr -> posx = 10;
    printf("%p \n %d, %d \n", pptr, (*pptr).posx, pptr -> posy);



    // 가능
    struct trianglePoint{
        int posx;
        int posy;
        struct trianglePoint * nextPoint; // 가능!!
    };

    struct trianglePoint p1 = {1,2};
    struct trianglePoint p2 = {3,4};
    struct trianglePoint p3 = {5,6};
    p1.nextPoint = &p2;
    p2.nextPoint = &p3;
    p3.nextPoint = &p1;
    printf("(%d, %d), (%d, %d), (%d, %d)\n", 
        p1.posx, p1.posy,
        (*p1.nextPoint).posx, (*p1.nextPoint).posy,
        (*(*p1.nextPoint).nextPoint).posx, (*(*p1.nextPoint).nextPoint).posy
    );


    // printf("%f", getDistance(mouses[0], mouses[1]))

}

// double getDistance(struct point a, struct point b){
//     int xdelta = (a.posx - b.posx);
//     int ydelta = (a.posy - b.posy);

//     double distance = sqrt((double)(xdelta^2 + ydelta^2));

//     return distance;
// }