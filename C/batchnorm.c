#include <stdio.h>
#include <math.h>

void batchnorm(int in_channels, int input_size, float* gamma_parameter, float* beta_parameter, float* input){ //parameter = (1, in_channels, 1 , 1)
    int i,j;

    float mean=0;
    float var=0;
    float eps=0.0001;

    for(i=0;i<in_channels;i++){
        for(j=0;j<input_size*input_size;j++)
            mean += *(input+i*input_size*input_size+j);
            var += (*(input+i*input_size*input_size+j))*(*(input+i*input_size*input_size+j));
    }

    mean=mean/(in_channels*input_size*input_size);
    var=var/(in_channels*input_size*input_size);

    var-=mean;

    for(i=0;i<in_channels;i++){
        for(j=0;j<input_size*input_size;j++)
            *(input+i*input_size*input_size+j)=(*(input+i*input_size*input_size+j))-mean;
            *(input+i*input_size*input_size+j)=(*(input+i*input_size*input_size+j))/sqrt(var+eps);
            *(input+i*input_size*input_size+j)=(*(input+i*input_size*input_size+j))*(*(gamma_parameter+i))+(*(beta_parameter+i));

    }
}