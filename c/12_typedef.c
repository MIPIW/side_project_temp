#include <stdio.h>




void main(){
    typedef int INT;
    INT num = 3;

    struct point{
        int posx;
        int posy;
    };
    typedef struct point Point;

    // both possible
    Point pos1 = {1,2};
    struct point pos2;

    // these are also possible
    typedef struct {
        int pos2x;
        int pos2y;
    } Point2;

    Point2 pos3;

    //application
    int func1(Point x){
        return x.posx;
    };
    printf("%d\n", func1(pos1));
    
    //application2
    int func2(Point * x){
        return (x -> posx) = (x -> posx) * -1;
    };
    printf("%d\n", func2(&pos1));
    printf("%d\n", pos1.posx);

    //구조체 변수의 연산은 직접 정의해야 한다. 
    Point vectorSubtraction(Point a, Point b){
        Point c;
        c.posx = a.posx - b.posx;
        c.posy = a.posy - b.posy;
        return c;
    };
    Point c = vectorSubtraction(pos1, pos1);
    printf("%d, %d\n", c.posx, c.posy);


    //[공용체]는 union
    typedef struct {
        unsigned short upper;
        unsigned short lower;
    } Dbshort;
    
    // 가장 긴 자료형인 ibuf를 기준으로 메모리를 할당하되, 
    // ibuf와 4 * char 길이의 bbuf, sbuf의 두 자료형이 같은 메모리를 공유해서 덮어씀
    // 동시에 부르지 않아도 될 떄, multimodal 자료형으로 활용 가능. 
    typedef union {
        int ibuf;
        char bbuf[4];
        Dbshort sbuf;
    } rdbuf;



    //[열거형]은 enum. 이름이랑 자연수랑 매치. 이름 있는 상수를 정의한다는 점에서 유용.
    typedef enum {DO, RE, MI, FA} syllable; 
    // typedef enum {DO=0, RE=1, MI=2, FA=3} syllable; // equivalent. 책에서는 1부터라고 되어 있는데 여기서는 0부터네 
    void func3(syllable var){
        switch(var){
            case DO: printf("%d\n", 10); break;
            case RE: printf("%d\n", 11); break;
            case MI: printf("%d\n", 12); break;
            case FA: printf("%d\n", 1234); break;
        };
    };
    //이런 것도 가능
    syllable a;
    for(a=DO; a<=FA; a++){
        func3(a);
    };

    typedef enum {DO1=1, RE1=4, MI1, FA1=8} syllable1; 
    // typedef enum {DO1=1, RE1=4, MI1=5, FA1=8} syllable1; 와 동일(1 증가)

}