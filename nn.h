#include<vector>
using namespace std;

class Perceptron {
    private:
        float w;
        float b;
    
    public:
        Perceptron(float w_, float b_){
            w = w_;
            b = b_;
        }

        ~Perceptron();
        
        float getOutput(float x){
            return w*x + b;
        }

        float setParams(float w_, float b_){
            w = w_;
            b = b_;
        }

};

class Layer {
    private:
        vector<Perceptron> ly;
    public:
        Layer(int numOfPerceptrons){
            for(int p = 0; p < numOfPerceptrons; p++){
                ly.push_back(Perceptron(0,0));
            }
        }
};

typedef class MultiLayerPerceptron {
    private:
        vector<Layer> lys;

    public:
        /*
        @param LayerDim[i] represents number of perceptrons in Layer[i]
        */
        MultiLayerPerceptron(vector<int> LayersDim){
            for(auto numOfPerceptrons:LayersDim){
                lys.push_back(Layer(numOfPerceptrons));
            }
        }

        ~ MultiLayerPerceptron();

} MLP;