#include "nn.h"

int main(){
    Topology t;
    MultiLayerPerceptron NeuralNet(t,e_ReLU);
    vector<double> input,output,results;
    //feed forward
    NeuralNet.feedForward(input);
    //back prop
    NeuralNet.backProp(output);
    // resutls
}