#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sstream>
#include <string>
#include <vector>

#include "include/generate.h"
#include "include/gpt2.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static bool parse_int(const char* text, int* value) {
    if (text == nullptr || value == nullptr) {
        return false;
    }

    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *value = (int)parsed;
    return true;
}

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static bool read_command_output(const std::string& command, std::string* output) {
    if (output == nullptr) {
        return false;
    }

    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return false;
    }

    output->clear();
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        *output += buffer;
    }

    int status = pclose(pipe);
    return status == 0;
}

static bool parse_id_list(const std::string& text, std::vector<int>* ids) {
    if (ids == nullptr) {
        return false;
    }

    std::istringstream in(text);
    ids->clear();
    int token = 0;
    while (in >> token) {
        ids->push_back(token);
    }
    return !ids->empty();
}

static bool encode_text_with_tokenizer(const std::string& model_dir, const std::string& text, std::vector<int>* ids) {
    if (ids == nullptr) {
        return false;
    }

    char temp_path[] = "/tmp/gpt2_text_XXXXXX";
    int fd = mkstemp(temp_path);
    if (fd < 0) {
        return false;
    }

    FILE* fp = fdopen(fd, "w");
    if (fp == nullptr) {
        close(fd);
        unlink(temp_path);
        return false;
    }
    fputs(text.c_str(), fp);
    fclose(fp);

    std::string command =
        "python3 tools/gpt2_tokenizer.py --model-dir " + shell_quote(model_dir) +
        " --encode-file " + shell_quote(temp_path);

    std::string output;
    bool ok = read_command_output(command, &output);
    unlink(temp_path);
    if (!ok) {
        return false;
    }
    return parse_id_list(output, ids);
}

static bool decode_ids_with_tokenizer(const std::string& model_dir, const std::vector<int>& ids, std::string* text) {
    if (text == nullptr || ids.empty()) {
        return false;
    }

    std::string command =
        "python3 tools/gpt2_tokenizer.py --model-dir " + shell_quote(model_dir) + " --decode";
    for (int id : ids) {
        command += " ";
        command += std::to_string(id);
    }

    if (!read_command_output(command, text)) {
        return false;
    }

    while (!text->empty() && ((*text)[text->size() - 1] == '\n' || (*text)[text->size() - 1] == '\r')) {
        text->pop_back();
    }
    return true;
}

static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

static void print_ids(const char* label, const int* ids, int n) {
    printf("%s:", label);
    for (int i = 0; i < n; ++i) {
        printf(" %d", ids[i]);
    }
    printf("\n");
}

static void print_usage(const char* argv0) {
    printf("Usage: %s [--model-dir DIR] [--generate N] [--text STRING | token_id ...]\n", argv0);
    printf("If no input is provided, defaults to text \"Hello, I am\".\n");
}

int main(int argc, char** argv) {
    std::string model_dir = "models/gpt2-bin";
    std::string input_text;
    int max_new_tokens = 0;
    std::vector<int> input_ids;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--model-dir requires a path\n");
                return 1;
            }
            model_dir = argv[++i];
            continue;
        }

        if (strcmp(argv[i], "--generate") == 0) {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &max_new_tokens) || max_new_tokens < 0) {
                fprintf(stderr, "--generate requires a non-negative integer\n");
                return 1;
            }
            ++i;
            continue;
        }

        if (strcmp(argv[i], "--text") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--text requires a string\n");
                return 1;
            }
            input_text = argv[++i];
            continue;
        }

        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }

        int token_id = 0;
        if (!parse_int(argv[i], &token_id)) {
            fprintf(stderr, "invalid token id: %s\n", argv[i]);
            return 1;
        }
        input_ids.push_back(token_id);
    }

    if (!input_text.empty() && !input_ids.empty()) {
        fprintf(stderr, "use either --text or token ids, not both\n");
        return 1;
    }

    if (input_text.empty() && input_ids.empty()) {
        input_text = "Hello, I am";
    }

    if (!input_text.empty()) {
        if (!encode_text_with_tokenizer(model_dir, input_text, &input_ids)) {
            fprintf(stderr, "failed to encode text with tokenizer from %s\n", model_dir.c_str());
            return 1;
        }
    }

    std::string config_path = model_dir + "/gpt2_config.json";

    GPT2Config cfg = {};
    if (!load_gpt2_config_from_json(&cfg, config_path.c_str())) {
        fprintf(stderr, "failed to load config from %s\n", config_path.c_str());
        return 1;
    }

    const int input_len = (int)input_ids.size();
    const int total_len = input_len + max_new_tokens;
    if (input_len <= 0) {
        fprintf(stderr, "input must contain at least one token\n");
        return 1;
    }
    if (total_len > cfg.max_position) {
        fprintf(stderr, "total length %d exceeds max_position %d\n", total_len, cfg.max_position);
        return 1;
    }

    GPT2Weights w = {};
    if (!load_gpt2_weights_from_dir(&w, cfg, model_dir.c_str())) {
        fprintf(stderr, "failed to load weights from %s\n", model_dir.c_str());
        return 1;
    }

    GPT2Workspace ws;
    if (!create_gpt2_workspace(&ws, cfg, total_len)) {
        fprintf(stderr, "failed to create workspace\n");
        destroy_gpt2_weights(&w, cfg);
        return 1;
    }

    if (!input_text.empty()) {
        printf("input text: %s\n", input_text.c_str());
    }
    print_ids("input ids", input_ids.data(), input_len);

    if (max_new_tokens > 0) {
        std::vector<int> output_ids(total_len);
        generate_greedy(input_ids.data(), input_len, max_new_tokens, output_ids.data(), w, ws, cfg);
        print_ids("generated ids", output_ids.data(), total_len);

        std::string decoded;
        if (decode_ids_with_tokenizer(model_dir, output_ids, &decoded)) {
            printf("generated text: %s\n", decoded.c_str());
        }
    } else {
        int* d_input_ids = nullptr;
        std::vector<float> last_logits(cfg.vocab_size);

        check_cuda(cudaMalloc(&d_input_ids, input_len * sizeof(int)), "malloc d_input_ids");
        check_cuda(
            cudaMemcpy(d_input_ids, input_ids.data(), input_len * sizeof(int), cudaMemcpyHostToDevice),
            "copy input ids"
        );

        gpt2_forward(d_input_ids, ws.logits, w, ws, cfg, input_len);
        check_cuda(cudaDeviceSynchronize(), "sync after gpt2_forward");

        check_cuda(
            cudaMemcpy(
                last_logits.data(),
                ws.logits + (input_len - 1) * cfg.vocab_size,
                cfg.vocab_size * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "copy last logits"
        );

        int next_token = argmax(last_logits.data(), cfg.vocab_size);
        printf("next token id: %d\n", next_token);

        std::vector<int> next_only = {next_token};
        std::string next_text;
        if (decode_ids_with_tokenizer(model_dir, next_only, &next_text)) {
            printf("next token text: %s\n", next_text.c_str());
        }

        std::vector<int> continued_ids = input_ids;
        continued_ids.push_back(next_token);
        std::string continued_text;
        if (decode_ids_with_tokenizer(model_dir, continued_ids, &continued_text)) {
            printf("continued text: %s\n", continued_text.c_str());
        }
        cudaFree(d_input_ids);
    }

    destroy_gpt2_workspace(&ws);
    destroy_gpt2_weights(&w, cfg);
    return 0;
}
