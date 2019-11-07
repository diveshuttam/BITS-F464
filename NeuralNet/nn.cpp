#include<vector>
#include<cstdlib>
#include<cassert>
#include<cmath>
#include "nn.h"
using namespace std;

/* connection */
double Connection::randomWeight(){
    return rand()/double(RAND_MAX);
}
Connection::Connection(){
    weight = randomWeight();
    deltaWeight=0;
}
/* end of connection*/


/* perceptron */
double Perceptron::f_ReLU(double x){
    return (x<0)?0:x;
}
double Perceptron::f_sigmoid(double x){
    return exp(x)/(exp(x)+1);
}
double Perceptron::f_tanh(double x){
    return tanh(x);
}

// derivatives
double Perceptron::d_ReLU(double x){
    return max(0.0,x);
}

double Perceptron::d_sigmoid(double x){
    return f_sigmoid(x)*(1-f_sigmoid(x));
}
double Perceptron::d_tanh(double x){
    return 1/(cosh(x)*cosh(x));
}
double Perceptron::sumDNext(Layer &nextLayer){
    double sum=0.0;
    for(unsigned long int i=0;i<nextLayer.size()-1;i++){
        sum+=outputConnections[i].weight * nextLayer[i].gradient;
    }
    return sum;
}
Perceptron::Perceptron(int pNum, int layerNum, const int numOutput, activationFunc f){
    this->perceptronNum=pNum;
    this->layerNum=layerNum;
    outputConnections = vector<Connection> (numOutput, Connection());
    this->f = f;
}

void Perceptron::setOutput(double outputVal){
    this->outputVal=outputVal;
}
double Perceptron::getOutputVal(){
    return outputVal;
}
double Perceptron::getWeight(int i){
    return outputConnections[i].weight;
}
double Perceptron::f_activate(double s){
    switch(f){
        case e_ReLU:
            return this->f_ReLU(s);
        case e_sigmoid:
            return this->f_sigmoid(s);
        case e_tanh:
            return this->f_tanh(s);
        default:
            //invalid activation function
            assert(false);
    }
}
double Perceptron::d_activate(double s){
    switch(f){
        case e_ReLU:
            return this->d_ReLU(s);
        case e_sigmoid:
            return this->d_sigmoid(s);
        case e_tanh:
            return this->d_sigmoid(s);
        default:
            //invalid activation function
            assert(false);
    }
    }
void Perceptron::setActivationFunc(activationFunc f){
    this->f=f;
}
void Perceptron::feedForward(Layer &prevLayer){
    double sum=0;
    for(unsigned long int i=0;i<prevLayer.size();i++){
        sum+=prevLayer[i].getWeight(perceptronNum)*prevLayer[i].getOutputVal();
    }
    this->outputVal=f_activate(sum);
}
void Perceptron::calcOutputGradient(double targetVal){
    double delta = targetVal-outputVal;
    gradient = delta*d_activate(outputVal);
}

void Perceptron::calcHiddenGradient(Layer &nextLayer){
    double sumOfDNext = sumDNext(nextLayer);
    gradient=sumOfDNext*d_activate(outputVal);
}
// eta is the learning rate
// alpha is the momentum
void Perceptron::updateWeights(Layer &prevLayer, double eta, double alpha){
    for(unsigned long int i=0;i<prevLayer.size();++i){
        Perceptron &prevPerceptron=prevLayer[i];
        double prevDelta = prevPerceptron.outputConnections[this->perceptronNum].deltaWeight;
        double newDelta = 
            eta
            *prevPerceptron.getOutputVal()
            *gradient
            + alpha
            * prevDelta;
        prevPerceptron.outputConnections[perceptronNum].deltaWeight=newDelta;
        prevPerceptron.outputConnections[perceptronNum].weight=newDelta;
    }
}
/* end of perceptron */


/* layer */
Layer::Layer(const int layerNum, const int numPerceptrons,const int numOutput, activationFunc f){
    this->layerNum=layerNum;
    // last one is the bias neuron
    for(int p = 0; p <= numPerceptrons; p++){
        perceptrons.push_back(Perceptron(p,layerNum, numOutput,f));
    }
}
unsigned long int Layer::size(){
    return perceptrons.size();
}
Perceptron &Layer::operator [] (int idx){
    return perceptrons[idx];
}
Perceptron Layer::back(){
    return perceptrons.back();
}
Perceptron Layer::front(){
    return perceptrons.front();
}
void Layer::feedForward(Layer &prevLayer){
    //initialize sum to 0

    //dont feed forward bias neuron
    for(long unsigned int i=0;i<this->perceptrons.size()-1;i++){
        perceptrons[i].feedForward(prevLayer);
    }
}
void Layer::calcHiddenGradient(Layer &nextLayer){
    for(long unsigned int i=0;i<perceptrons.size();i++){
        perceptrons[i].calcHiddenGradient(nextLayer);
    }
}
void Layer::updateWeights(Layer &prevLayer, double eta, double alpha){
    for(long unsigned int i=0;i<perceptrons.size()-1;i++){
        perceptrons[i].updateWeights(prevLayer, eta, alpha);
    }
}
/* end of layer*/

/* MultiLayerPerceptron (MLP) */
/*
@param LayerDim[i] represents number of perceptrons in Layer[i]
*/
MultiLayerPerceptron::MultiLayerPerceptron(const Topology &t,activationFunc f, int runningSamplesCount, double eta, double alpha){
    int numLayers=t.size();
    currentError=0.0;
    averageError=0.0;
    this->runningSamplesCount=runningSamplesCount;
    this->eta=eta;
    this->alpha=alpha;

    // initialize layers
    for(int i=0;i<numLayers;i++){
        int numOutput=(i==numLayers-1)?0:t[i+1];
        //one extra bias neuron therefore t[i]+1
        Layer(i,t[i]+1,numOutput,f);
    }
    // set the bias neurons to 1.0
    for(int i=0;i<numLayers;i++){
        layers[i].back().setOutput(1.0);
    }
}
void MultiLayerPerceptron::feedForward(const vector<double> &inputVals){
    assert(inputVals.size()==layers[0].size()-1);
    
    // set input neurons
    Layer &inputLayer = layers.front();
    int numInput = inputLayer.size();
    for(int i=0;i<numInput-1;i++){
        inputLayer[i].setOutput(inputVals[i]);
    }

    // feed forward
    for(int i=1;i<numInput;i++){
        Layer &prevLayer=layers[i-1];
        Layer &currentLayer=layers[i];
        currentLayer.feedForward(prevLayer);
    }
}

void MultiLayerPerceptron::backProp(const vector<double> &outputVals){
    assert(outputVals.size()==layers.back().size()-1);
    //calculate loss
    Layer &outputLayer = layers.back();
    double loss = 0;
    // not include the bias
    for(unsigned long int i=0;i<outputLayer.size()-1;i++){
        loss+=pow((outputLayer[i].getOutputVal()-outputVals[i]),2);
    }
    loss=sqrt(loss);
    currentError=loss;
    averageError=(averageError*(runningSamplesCount-1)+currentError)/runningSamplesCount;

    // output layer gradients
    for(unsigned long int i=0;i<outputLayer.size()-1;i++){
        outputLayer[i].calcOutputGradient(outputVals[i]);
    }

    // hidden layer gradients
    for(unsigned long int i=layers.size()-2;i>0;i--){
        Layer &currentLayer = layers[i];
        Layer &nextLayer = layers[i+1];
        currentLayer.calcHiddenGradient(nextLayer);
    }

    // update weights
    for(unsigned long int i=layers.size()-1;i>0;i--){
        Layer &currentLayer = layers[i];
        Layer &prevLayer = layers[i-1];
        currentLayer.updateWeights(prevLayer, eta, alpha);
    }
}
void MultiLayerPerceptron::getResults(vector<double> &results){
    Layer &outputLayer = layers.back();
    results.clear();
    for(unsigned long int i=0;i<outputLayer.size()-1;i++){
        results.push_back(outputLayer[i].getOutputVal());
    }
}
/* end of MLP */