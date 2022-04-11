#include <stdio.h>

void Shift(int in_channels, int input_size, float* parameter, float* input){ //parameter = (1, in_channels, 1 , 1)
    int i,j;

    for(i=0;i<in_channels;i++){
        for(j=0;j<input_size*input_size;j++)
            *(input+i*input_size*input_size+j) = *(input+i*input_size*input_size+j)+*(parameter+i);
    }
}

void Sign(int in_channels, int input_size, float* input){
    int i,j;

    for(i=0;i<in_channels;i++)
        for(j=0;j<input_size*input_size;j++){
            if (*(input+i*input_size*input_size+j) >= 0){
                *(input+i*input_size*input_size+j)=1;
            }
            else{
                *(input+i*input_size*input_size+j)=0;
            }
        }
}

void PReLu(int in_channels, int input_size, float* input, float* parameter){
    int i,j;

    for(i=0;i<in_channels;i++)
        for(j=0;j<input_size*input_size;j++){
            if (*(input+i*input_size*input_size+j) < 0){
                *(input+i*input_size*input_size+j)=*(parameter+i)*(*(input+i*input_size*input_size+j));
            }
        }
}

void RSign(int in_channels, int input_size, float* input, float* shift_parameter){
    Shift(in_channels, input_size, shift_parameter, input);
    Sign(in_channels, input_size, input);
}

void RPReLu(int in_channels, int input_size, float* input, float* relu_parameter, float* gamma_parameter, float* zeta_parameter){
    Shift(in_channels, input_size, input, gamma_parameter);
    PReLu(in_channels, input_size, input, relu_parameter);
    Shift(in_channels, input_size, input, zeta_parameter);
}

