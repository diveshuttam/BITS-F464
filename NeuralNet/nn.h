#include<vector>
#include<cstdlib>
#include<cassert>
#include<cmath>

using namespace std;

class Connection {
    private:
        static double randomWeight();
    public:
        double weight;
        double deltaWeight;
        Connection();
};

class Layer;

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
        static double f_ReLU(double x);
        static double f_sigmoid(double x);
        static double f_tanh(double x);

        // derivatives
        static double d_ReLU(double x);
        static double d_sigmoid(double x);
        static double d_tanh(double x);
        double sumDNext(Layer &nextLayer);
    public:
        Perceptron(int pNum, int layerNum, const int numOutput, activationFunc f);

        void setOutput(double outputVal);
        double getOutputVal();
        double getWeight(int i);
        double f_activate(double s);
        double d_activate(double s);
        void setActivationFunc(activationFunc f);
        void feedForward(Layer &prevLayer);
        void calcOutputGradient(double targetVal);
        void calcHiddenGradient(Layer &nextLayer);

        // eta is the learning rate
        // alpha is the momentum
        void updateWeights(Layer &prevLayer, double eta, double alpha);
};

class Layer{
    private:
        int layerNum;
        vector<Perceptron> perceptrons;
    public:
        Layer(const int layerNum, const int numPerceptrons,const int numOutput, activationFunc f);
        unsigned int size();
        Perceptron &operator[] (int idx);
        Perceptron back();
        Perceptron front();
        void feedForward(Layer &prevLayer);
        void calcHiddenGradient(Layer &nextLayer);
        void updateWeights(Layer &prevLayer, double eta, double alpha);
};

typedef vector<int> Topology;

typedef class MultiLayerPerceptron {
    private:
        vector<Layer> layers;
        double currentError;
        double averageError;
        int runningSamplesCount;
        double eta; // learning rate
        double alpha; // momentum

    public:
        /*
        @param LayerDim[i] represents number of perceptrons in Layer[i]
        */
        MultiLayerPerceptron(const Topology &t,activationFunc f, int runningSamplesCount, double eta, double alpha);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &outputVals);
        void getResults(vector<double> &results);
} MLP;