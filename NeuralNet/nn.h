#include<vector>
#include<cstdlib>
#include<cassert>
#include<cmath>

using namespace std;

class Connection {
    private:
        static double randomWeight(){
            return rand()/double(RAND_MAX);
        }
    public:
        double weight;
        double deltaWeight;
        Connection(){
            weight = randomWeight();
            deltaWeight=0;
        }
};

typedef enum activationFunc{
    e_ReLU,
    e_sigmoid,
    e_tanh,
}activationFunc;

class Perceptron {
    private:
        int perceptronNum;
        int layerNum;
        double outputVal;
        double gradient;
        vector<Connection> outputConnections;
        activationFunc f;
        static double f_ReLU(double x){
            return (x<0)?0:x;
        }
        static double f_sigmoid(double x){
            return exp(x)/(exp(x)+1);
        }
        static double f_tanh(double x){
            return tanh(x);
        }

        // derivatives
        static double d_ReLU(double x){
            return max(0.0,x);
        }

        static double d_sigmoid(double x){
            return f_sigmoid(x)*(1-f_sigmoid(x));
        }
        static double d_tanh(double x){
            return 1/(cosh(x)*cosh(x));
        }
        double sumDNext(Layer &nextLayer){
            double sum=0.0;
            for(int i=0;i<nextLayer.size()-1;i++){
                sum+=outputConnections[i].weight * nextLayer[i].gradient;
            }
        }
    public:
        Perceptron(int pNum, int layerNum, const int numOutput, activationFunc f){
            this->perceptronNum=pNum;
            this->layerNum=layerNum;
            outputConnections = vector<Connection> (numOutput, Connection());
            this->f = f;
        }

        ~Perceptron();
        void setOutput(double outputVal){
            this->outputVal=outputVal;
        }
        double getOutputVal(){
            return outputVal;
        }
        double getWeight(int i){
            return outputConnections[i].weight;
        }
        double f_activate(double s){
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
        double d_activate(double s){
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
        void setActivationFunc(activationFunc f){
            this->f=f;
        }
        void feedForward(Layer &prevLayer){
            double sum=0;
            for(int i=0;i<prevLayer.size();i++){
                sum+=prevLayer[i].getWeight(perceptronNum)*prevLayer[i].getOutputVal();
            }
            this->outputVal=f_activate(sum);
        }
        void calcOutputGradient(double targetVal){
            double delta = targetVal-outputVal;
            gradient = delta*d_activate(outputVal);
        }

        void calcHiddenGradient(Layer &nextLayer){
            double sumOfDNext = sumDNext(nextLayer);
            gradient=sumOfDNext*d_activate(outputVal);
        }
        // eta is the learnign rate
        // alpha is the momentum
        void updateWeights(Layer &prevLayer, double eta=0.15, double alpha=0.5){
            for(int i=0;i<prevLayer.size();++i){
                Perceptron &prevPerceptron=prevLayer[i];
                double prevDelta = prevPerceptron.outputConnections[this->perceptronNum].deltaWeight;
                double newDelta = 
                    eta
                    *prevPerceptron.getOutputVal()
                    *gradient
                    + alpha
                    * prevDelta;
            }
        }
};

class Layer{
    private:
        int layerNum;
        vector<Perceptron> perceptrons;
    public:
        Layer(const int layerNum, const int numPerceptrons,const int numOutput, activationFunc f){
            this->layerNum=layerNum;
            // last one is the bias neuron
            for(int p = 0; p <= numPerceptrons; p++){
                perceptrons.push_back(Perceptron(p,layerNum, numOutput,f));
            }
        }
        unsigned int size(){
            return perceptrons.size();
        }
        Perceptron &operator[] (unsigned int idx){
            return perceptrons[idx];
        }
        Perceptron back(){
            return perceptrons.back();
        }
        Perceptron front(){
            return perceptrons.front();
        }
        void feedForward(Layer &prevLayer){
            //initialize sum to 0

            //dont feed forward bias neuron
            for(int i=0;i<this->perceptrons.size()-1;i++){
                perceptrons[i].feedForward(prevLayer);
            }
        }
        void calcHiddenGradient(Layer &nextLayer){
            for(int i=0;i<perceptrons.size();i++){
                perceptrons[i].calcHiddenGradient(nextLayer);
            }
        }
        void updateWeights(Layer &prevLayer){
            for(int i=0;i<perceptrons.size()-1;i++){
                perceptrons[i].updateWeights(prevLayer);
            }
        }
};

typedef vector<int> Topology;

typedef class MultiLayerPerceptron {
    private:
        vector<Layer> layers;
        double currentError;
        double averageError;
        int runningSamplesCount;

    public:
        /*
        @param LayerDim[i] represents number of perceptrons in Layer[i]
        */
        MultiLayerPerceptron(const Topology &t,activationFunc f, int runningSamplesCount){
            unsigned int numLayers=t.size();
            currentError=0.0;
            averageError=0.0;
            this->runningSamplesCount=runningSamplesCount;
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
        void feedForward(const vector<double> &inputVals){
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

        void backProp(const vector<double> &outputVals){
            assert(outputVals.size()==layers.back().size()-1);
            //calculate loss
            Layer &outputLayer = layers.back();
            double loss = 0;
            // not include the bias
            for(int i=0;i<outputLayer.size()-1;i++){
                loss+=pow((outputLayer[i].getOutputVal()-outputVals[i]),2);
            }
            loss=sqrt(loss);
            currentError=loss;
            averageError=(averageError*(runningSamplesCount-1)+currentError)/runningSamplesCount;

            // output layer gradients
            for(int i=0;i<outputLayer.size()-1;i++){
                outputLayer[i].calcOutputGradient(outputVals[i]);
            }

            // hidden layer gradients
            for(int i=layers.size()-2;i>0;i--){
                Layer &currentLayer = layers[i];
                Layer &nextLayer = layers[i+1];
                currentLayer.calcHiddenGradient(nextLayer);
            }

            // update weights
            for(int i=layers.size()-1;i>0;i--){
                Layer &currentLayer = layers[i];
                Layer &prevLayer = layers[i-1];
                currentLayer.updateWeights(prevLayer);
            }
        }
} MLP;