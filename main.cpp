#include <iostream>
#include <Eigen/Dense>

class Transformer {
public:
    // Positional encoding: capture the order of tokens
    // Self-attension weights
    W_q = Eigen::MatrixXd::Random(d_model, d_model);
    W_k = Eigen::MatrixXd::Random(d_model, d_model);
    W_v = Eigen::MatrixXd::Random(d_model, d_model);
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
