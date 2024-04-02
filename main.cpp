#include <iostream>
#include <Eigen/Dense>

class Transformer {
private:
    int d_model;
    int num_heads;
    int seq_length;

    Eigen::MatrixXd W_q, W_k, W_v;
    Eigen::MatrixXd W_ffn1, W_ffn2;

public:
    Transformer(int d_model, int num_heads, int seq_length) {
        // Positional encoding: capture the order of tokens
        // Self-attension weights
        W_q = Eigen::MatrixXd::Random(d_model, d_model);
        W_k = Eigen::MatrixXd::Random(d_model, d_model);
        W_v = Eigen::MatrixXd::Random(d_model, d_model);

        // Feedforward weights
        W_ffn1 = Eigen::MatrixXd::Random(d_model, d_model);
        W_ffn2 = Eigen::MatrixXd::Random(d_model, d_model);

        // Initialize other hyperparameters
        this->d_model = d_model;
        this->num_heads = num_heads;
        this->seq_length = seq_length;
    }
};

int main() {
    int d_model = 64;       // model dimension
    int num_heads = 8;      // number of attension heads
    int seq_length = 10;    // sequence length
                         
    // Create an input sequence
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(seq_length, d_model);
    std::cout << "Input shape: " << input.rows() << " x " << input.cols() << std::endl;

    return 0;
}
