#include <fstream>
#include <string>

#include "../include/gpt2.h"

static bool extract_json_int(const std::string& text, const char* key, int* value) {
    if (value == nullptr) {
        return false;
    }

    std::string pattern = "\"";
    pattern += key;
    pattern += "\"";

    size_t key_pos = text.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }

    size_t colon_pos = text.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) {
        return false;
    }

    size_t value_start = text.find_first_of("-0123456789", colon_pos + 1);
    if (value_start == std::string::npos) {
        return false;
    }

    size_t value_end = text.find_first_not_of("-0123456789", value_start);
    std::string value_text = text.substr(value_start, value_end - value_start);
    *value = std::stoi(value_text);
    return true;
}

bool validate_config(const GPT2Config& cfg) {
    if (cfg.hidden <= 0) return false;
    if (cfg.heads <= 0) return false;
    if (cfg.n_layer <= 0) return false;
    if (cfg.vocab_size <= 0) return false;
    if (cfg.max_position <= 0) return false;
    if (cfg.hidden % cfg.heads != 0) return false;
    return true;
}

bool load_gpt2_config_from_json(GPT2Config* cfg, const char* path) {
    if (cfg == nullptr || path == nullptr) {
        return false;
    }

    std::ifstream in(path);
    if (!in) {
        return false;
    }

    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    GPT2Config loaded = {};
    if (!extract_json_int(text, "hidden", &loaded.hidden)) return false;
    if (!extract_json_int(text, "heads", &loaded.heads)) return false;
    if (!extract_json_int(text, "n_layer", &loaded.n_layer)) return false;
    if (!extract_json_int(text, "vocab_size", &loaded.vocab_size)) return false;
    if (!extract_json_int(text, "max_position", &loaded.max_position)) return false;
    if (!validate_config(loaded)) return false;

    *cfg = loaded;
    return true;
}
