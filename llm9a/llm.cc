#include "llm.h"
#include <cstring>

constexpr int unroll = 4;
typedef double double4_t __attribute__ ((vector_size (unroll * sizeof(double))));
typedef float float4_t __attribute__ ((vector_size (unroll * sizeof(float))));

void multi_head_attention(
    const LLamaConfig &config,
    const LLamaLayer &layer,
    float *keys,
    float *values,
    float *activation,
    int position);

void matmul(float *out, const float *x, const float *w, int n, int d);

void llm(LLamaConfig config, LLamaParameters params, const std::vector<token_t> &tokens, std::vector<float> &logits) {
    using namespace utils;

    std::vector<std::vector<float>> keys(config.n_layers, std::vector<float>(config.seq_len * config.dim));
    std::vector<std::vector<float>> values(config.n_layers, std::vector<float>(config.seq_len * config.dim));
    // a few convenience variables
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    
    // initialize the activation by the embedding of the current token
    // std::vector<float> activation(params.TokenEmbeddingMatrix.begin() + token_id * dim,
    // params.TokenEmbeddingMatrix.begin() + (token_id + 1) * dim);
    
    // scratch buffers to use during the computations
    std::vector<float> activation(dim);
    std::vector<float> buffer(dim);
    std::vector<float> buffer2(dim);
    std::vector<float> hidden_buffer(hidden_dim);
    std::vector<float> hidden_buffer2(hidden_dim);
    
    for (unsigned position = 0; position < tokens.size(); ++position) {
        int token_id = (int)tokens[position];
        std::copy(params.TokenEmbeddingMatrix.begin() + token_id*dim, params.TokenEmbeddingMatrix.begin() + (token_id + 1)*dim, activation.begin());
        // forward all the layers
        for (int l = 0; l < config.n_layers; l++) {
            auto &layer = params.LayerWeights[l];

            // attention rmsnorm
            rmsnorm(buffer.data(),
                    activation.data(),
                    layer.rms_attention.data(),
                    dim);

            multi_head_attention(config, layer, keys[l].data(), values[l].data(), buffer.data(), position);

            // final matmul to get the output of the attention
            matmul(buffer2.data(), buffer.data(), layer.out_weight_matrix.data(), dim, dim);

            // residual connection back into x
            #pragma omp parallel for simd schedule(static)
            for (int i = 0; i < dim; i++) {
                activation[i] += buffer2[i];
            }

            rmsnorm(buffer.data(), activation.data(), layer.rms_feed_forward.data(), dim);

            matmul(hidden_buffer.data(), buffer.data(), layer.feed_forward_w1.data(), dim, hidden_dim);
            matmul(hidden_buffer2.data(), buffer.data(), layer.feed_forward_w3.data(), dim, hidden_dim);
            swiglu(hidden_buffer.data(), hidden_buffer.data(), hidden_buffer2.data(), hidden_dim);
            // final matmul to get the output of the ffn
            matmul(buffer.data(), hidden_buffer.data(), layer.feed_forward_w2.data(), hidden_dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++) {
                activation[i] += buffer[i];
            }
        }

        // final rmsnorm
        rmsnorm(activation.data(), activation.data(), params.RmsFinal.data(), dim);

        // classifier into logits
        float *current_logits = logits.data() + position * config.vocab_size;
        matmul(current_logits, activation.data(), params.TokenOutputMatrix.data(), dim, config.vocab_size);
    }
}

void matmul(float *out, const float *x, const float *w, int n, int d) {
    const int n_v = n/unroll;
    const int nblock = 4;
    const int db = (d/nblock)*nblock;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < db; i += nblock) {
        const float *row0 = w + i*n;
        const float *row1 = w + (i+1)*n;
        const float *row2 = w + (i+2)*n;
        const float *row3 = w + (i+3)*n;
        double4_t tmp0 = {0,0,0,0};
        double4_t tmp1 = {0,0,0,0};
        double4_t tmp2 = {0,0,0,0};
        double4_t tmp3 = {0,0,0,0};
        
        for (int j = 0; j < n_v; j++) {
            float4_t xv, rv0, rv1, rv2, rv3;
            memcpy(&xv, x + j * unroll, sizeof(float4_t));
            memcpy(&rv0, row0 + j * unroll, sizeof(float4_t));
            memcpy(&rv1, row1 + j * unroll, sizeof(float4_t));
            memcpy(&rv2, row2 + j * unroll, sizeof(float4_t));
            memcpy(&rv3, row3 + j * unroll, sizeof(float4_t));
            float4_t p0 = xv * rv0;
            float4_t p1 = xv * rv1;
            float4_t p2 = xv * rv2;
            float4_t p3 = xv * rv3;

            double4_t pd0, pd1, pd2, pd3;
            for (int k = 0; k < unroll; k++) {
                pd0[k] = p0[k];
                pd1[k] = p1[k];
                pd2[k] = p2[k];
                pd3[k] = p3[k];
            }
            tmp0 += pd0;
            tmp1 += pd1;
            tmp2 += pd2;
            tmp3 += pd3;
        }
        
        double sum0 = 0.f;
        double sum1 = 0.f;
        double sum2 = 0.f;
        double sum3 = 0.f;
        for (int k = 0; k < unroll; k++) {
            sum0 += tmp0[k];
            sum1 += tmp1[k];
            sum2 += tmp2[k];
            sum3 += tmp3[k];
        }
        
        for (int j = n_v*unroll; j < n; j++) {
            double xj = x[j];
            sum0 += xj*row0[j]; 
            sum1 += xj*row1[j]; 
            sum2 += xj*row2[j]; 
            sum3 += xj*row3[j]; 

        } 
        out[i] = (float)sum0;
        out[i+1] = (float)sum1;
        out[i+2] = (float)sum2;
        out[i+3] = (float)sum3;
        // uses doubles to ensure numerical stability.
        // out[i] = std::inner_product(x, x + n, w + i * n, 0.0);
    }
}

void multi_head_attention(
    const LLamaConfig &config,
    const LLamaLayer &layer,
    float *keys,
    float *values,
    float *activation,
    int position) {
    using namespace utils;
    int dim = config.dim;
    int head_size = config.head_size();

    // key and value point to the kv cache
    float *key = keys + position * dim;
    float *value = values + position * dim;

    std::vector<float> query(dim);
    // qkv matmuls for this position.
    matmul(query.data(),
           activation,
           layer.query_weight_matrix.data(),
           dim, dim);

    // key and value are generated directly at the desired position
    // inside the cache
    matmul(key,
           activation,
           layer.key_weight_matrix.data(),
           dim, dim);
    matmul(value,
           activation,
           layer.value_weight_matrix.data(),
           dim, dim);

    rope(config, query.data(), key, position);

    
    // multi-head attention. iterate over all heads
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < config.n_heads; h++) {
        // get the query vector for this head
        float *q = query.data() + h * head_size;
        std::vector<float> attention(config.seq_len);

        calculate_attention(config, attention.data(), q, position,
                            keys + h * head_size);
        lookup_with_attention(config, attention.data(),
                              activation + h * head_size,
                              position,
                              values + h * head_size);
    }
}
