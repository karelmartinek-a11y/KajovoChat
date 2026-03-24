#include "kajovochat_windows_aec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

constexpr double kDefaultMu = 0.14;
constexpr double kDefaultLeakage = 0.9995;
constexpr double kResidualEps = 1e-9;

struct WindowsAecState {
    int samplerate;
    int filter_length;
    int max_shift_samples;
    double mu;
    double leakage;
    std::vector<double> weights;
    std::vector<double> reference_history;
    std::int64_t reference_history_start;
    std::int64_t reference_total_samples;
    std::int64_t last_delay_samples;
    double last_delay_score;
};

static double clamp_double(double value, double minimum, double maximum) {
    return std::max(minimum, std::min(value, maximum));
}

static double rms_from_block(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value * value;
    }
    return std::sqrt(sum / static_cast<double>(values.size()) + kResidualEps);
}

static std::vector<double> pcm16_to_double_block(const std::int16_t* data, int count) {
    std::vector<double> block;
    block.reserve(std::max(0, count));
    for (int i = 0; i < count; ++i) {
        block.push_back(static_cast<double>(data[i]) / 32768.0);
    }
    return block;
}

static void trim_history(WindowsAecState& state, std::size_t keep_samples) {
    if (state.reference_history.size() <= keep_samples) {
        return;
    }
    const std::size_t drop = state.reference_history.size() - keep_samples;
    state.reference_history.erase(state.reference_history.begin(), state.reference_history.begin() + static_cast<std::ptrdiff_t>(drop));
    state.reference_history_start += static_cast<std::int64_t>(drop);
}

static std::int64_t delay_ms_to_samples(int delay_ms, int samplerate) {
    const double delay = static_cast<double>(delay_ms) * static_cast<double>(samplerate) / 1000.0;
    return static_cast<std::int64_t>(std::llround(delay));
}

static std::int64_t clamp_delay_samples(std::int64_t value, int max_shift_samples) {
    const std::int64_t upper = static_cast<std::int64_t>(std::max(0, max_shift_samples));
    if (value < 0) {
        return 0;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static double correlation_score(
    const std::vector<double>& history,
    std::int64_t history_start,
    const std::vector<double>& mic_block,
    std::int64_t block_start_index,
    std::int64_t delay_samples) {
    const int sample_count = static_cast<int>(mic_block.size());
    if (sample_count <= 0) {
        return -1.0;
    }

    double sum_ref = 0.0;
    double sum_mic = 0.0;
    double sum_ref_sq = 0.0;
    double sum_mic_sq = 0.0;
    double sum_cross = 0.0;
    int valid = 0;
    for (int i = 0; i < sample_count; i += 2) {
        const std::int64_t aligned_index = block_start_index + static_cast<std::int64_t>(i) - delay_samples;
        if (aligned_index < history_start) {
            continue;
        }
        const std::int64_t history_pos = aligned_index - history_start;
        if (history_pos < 0 || history_pos >= static_cast<std::int64_t>(history.size())) {
            continue;
        }
        const double ref = history[static_cast<std::size_t>(history_pos)];
        const double mic = mic_block[static_cast<std::size_t>(i)];
        sum_ref += ref;
        sum_mic += mic;
        sum_ref_sq += ref * ref;
        sum_mic_sq += mic * mic;
        sum_cross += ref * mic;
        ++valid;
    }
    if (valid < 8) {
        return -1.0;
    }
    const double n = static_cast<double>(valid);
    const double ref_mean = sum_ref / n;
    const double mic_mean = sum_mic / n;
    const double ref_var = std::max(1e-9, (sum_ref_sq / n) - (ref_mean * ref_mean));
    const double mic_var = std::max(1e-9, (sum_mic_sq / n) - (mic_mean * mic_mean));
    const double covariance = (sum_cross / n) - (ref_mean * mic_mean);
    return covariance / std::sqrt(ref_var * mic_var);
}

static std::int64_t refine_delay_samples(
    const WindowsAecState& state,
    const std::vector<double>& mic_block,
    std::int64_t block_start_index,
    std::int64_t requested_delay) {
    const std::int64_t base_delay = clamp_delay_samples(requested_delay, state.max_shift_samples);
    const std::int64_t anchor_delay = state.last_delay_samples > 0 ? state.last_delay_samples : base_delay;
    const std::int64_t search_center = anchor_delay > 0 ? anchor_delay : base_delay;
    const std::int64_t search_radius = std::max<std::int64_t>(24, std::min<std::int64_t>(state.max_shift_samples, std::max<std::int64_t>(state.filter_length / 6, 96)));
    const std::int64_t lower = clamp_delay_samples(search_center - search_radius, state.max_shift_samples);
    const std::int64_t upper = clamp_delay_samples(search_center + search_radius, state.max_shift_samples);

    std::int64_t best_delay = base_delay;
    double best_score = -1.0;
    for (std::int64_t candidate = lower; candidate <= upper; candidate += std::max<std::int64_t>(8, state.filter_length / 32)) {
        const double score = correlation_score(
            state.reference_history,
            state.reference_history_start,
            mic_block,
            block_start_index,
            candidate);
        if (score > best_score) {
            best_score = score;
            best_delay = candidate;
        }
    }

    if (best_score < 0.0) {
        return base_delay;
    }
    if (state.last_delay_samples > 0) {
        const std::int64_t jump_limit = std::max<std::int64_t>(24, state.filter_length / 5);
        const double score_margin = 0.03;
        const double previous_score = state.last_delay_score;
        const bool too_far = std::llabs(best_delay - state.last_delay_samples) > jump_limit;
        const bool not_better_enough = best_score < previous_score + score_margin;
        if (too_far && not_better_enough) {
            return anchor_delay;
        }
    }
    return best_delay;
}

}  // namespace

extern "C" {

void* kajovochat_aec_create(int samplerate, int filter_length, int max_shift_samples) {
    if (samplerate <= 0 || filter_length <= 0 || max_shift_samples < 0) {
        return nullptr;
    }
    try {
        auto* state = new WindowsAecState{
            samplerate,
            filter_length,
            max_shift_samples,
            kDefaultMu,
            kDefaultLeakage,
            std::vector<double>(static_cast<std::size_t>(filter_length), 0.0),
            {},
            0,
            0,
            0,
            -1.0,
        };
        state->reference_history.reserve(static_cast<std::size_t>(std::max(4096, filter_length + max_shift_samples + 1024)));
        return state;
    } catch (...) {
        return nullptr;
    }
}

void kajovochat_aec_destroy(void* handle) {
    auto* state = static_cast<WindowsAecState*>(handle);
    delete state;
}

int kajovochat_aec_process(
    void* handle,
    const std::int16_t* mic,
    int mic_samples,
    const std::int16_t* reference,
    int ref_samples,
    int delay_ms,
    std::int16_t* out_pcm,
    int out_capacity,
    double* out_quality,
    double* out_improvement,
    double* out_residual,
    int* out_is_strong) {
    if (!handle || !mic || !out_pcm || mic_samples <= 0 || out_capacity <= 0 || ref_samples < 0) {
        return -1;
    }

    auto* state = static_cast<WindowsAecState*>(handle);
    const int process_samples = std::min(mic_samples, out_capacity);
    const int safe_delay_ms = std::max(0, delay_ms);
    const std::int64_t requested_delay = delay_ms_to_samples(safe_delay_ms, state->samplerate);

    std::vector<double> mic_block = pcm16_to_double_block(mic, process_samples);
    std::vector<double> ref_block = pcm16_to_double_block(reference, ref_samples);
    const std::int64_t block_start_index = state->reference_total_samples;

    state->reference_history.insert(state->reference_history.end(), ref_block.begin(), ref_block.end());
    state->reference_total_samples += static_cast<std::int64_t>(ref_block.size());
    const std::size_t history_keep = static_cast<std::size_t>(
        std::max(8192, state->filter_length + state->max_shift_samples + std::max(process_samples, ref_samples) + 512));
    trim_history(*state, history_keep);

    const std::int64_t delay_samples = refine_delay_samples(*state, mic_block, block_start_index, requested_delay);
    state->last_delay_samples = delay_samples;
    state->last_delay_score = correlation_score(
        state->reference_history,
        state->reference_history_start,
        mic_block,
        block_start_index,
        delay_samples);

    const double pre_mic_rms = rms_from_block(mic_block);
    const bool adapt_block = bool(
        state->last_delay_score >= 0.08
        || std::llabs(delay_samples - requested_delay) <= std::max<std::int64_t>(24, state->filter_length / 8));

    std::vector<double> output_block(static_cast<std::size_t>(process_samples), 0.0);
    std::vector<double> predicted_block(static_cast<std::size_t>(process_samples), 0.0);

    for (int i = 0; i < process_samples; ++i) {
        const std::int64_t aligned_index = block_start_index + static_cast<std::int64_t>(i) - delay_samples;
        double prediction = 0.0;
        double norm = 1e-6;
        if (aligned_index >= state->reference_history_start) {
            const std::int64_t history_pos = aligned_index - state->reference_history_start;
            const std::int64_t available = history_pos + 1;
            const int taps_available = std::min(
                state->filter_length,
                static_cast<int>(std::max<std::int64_t>(0, available)));
            for (int tap = 0; tap < taps_available; ++tap) {
                const std::int64_t sample_pos = history_pos - tap;
                if (sample_pos < 0 || sample_pos >= static_cast<std::int64_t>(state->reference_history.size())) {
                    continue;
                }
                const double x = state->reference_history[static_cast<std::size_t>(sample_pos)];
                prediction += state->weights[static_cast<std::size_t>(tap)] * x;
                norm += x * x;
            }
            const double error = mic_block[static_cast<std::size_t>(i)] - prediction;
            if (adapt_block) {
                const double adapt_scale = state->mu * error / norm;
                for (int tap = 0; tap < taps_available; ++tap) {
                    const std::int64_t sample_pos = history_pos - tap;
                    if (sample_pos < 0 || sample_pos >= static_cast<std::int64_t>(state->reference_history.size())) {
                        continue;
                    }
                    const double x = state->reference_history[static_cast<std::size_t>(sample_pos)];
                    state->weights[static_cast<std::size_t>(tap)] = clamp_double(
                        state->weights[static_cast<std::size_t>(tap)] * state->leakage + adapt_scale * x,
                        -1.5,
                        1.5);
                }
            }
            output_block[static_cast<std::size_t>(i)] = error;
            predicted_block[static_cast<std::size_t>(i)] = prediction;
        } else {
            output_block[static_cast<std::size_t>(i)] = mic_block[static_cast<std::size_t>(i)];
            predicted_block[static_cast<std::size_t>(i)] = 0.0;
        }
    }

    const double mic_rms = pre_mic_rms;
    const double residual_rms = rms_from_block(output_block);
    const double predicted_rms = rms_from_block(predicted_block);
    const double improvement = clamp_double(1.0 - (residual_rms / std::max(mic_rms, 1e-6)), 0.0, 1.0);
    const double predicted_ratio = clamp_double(predicted_rms / std::max(mic_rms, 1e-6), 0.0, 1.0);
    const double quality = clamp_double(improvement * std::max(0.35, predicted_ratio), 0.0, 1.0);
    const bool strong = improvement >= 0.22 && residual_rms <= std::max(0.0035, mic_rms * 0.45);

    for (int i = 0; i < process_samples; ++i) {
        const double sample = clamp_double(output_block[static_cast<std::size_t>(i)], -1.0, 1.0);
        out_pcm[i] = static_cast<std::int16_t>(std::lround(sample * 32767.0));
    }
    if (out_capacity > process_samples) {
        std::memset(out_pcm + process_samples, 0, static_cast<std::size_t>(out_capacity - process_samples) * sizeof(std::int16_t));
    }

    if (out_quality) {
        *out_quality = quality;
    }
    if (out_improvement) {
        *out_improvement = improvement;
    }
    if (out_residual) {
        *out_residual = clamp_double(residual_rms / std::max(mic_rms, 1e-6), 0.0, 1.0);
    }
    if (out_is_strong) {
        *out_is_strong = strong ? 1 : 0;
    }

    return 0;
}

}  // extern "C"
