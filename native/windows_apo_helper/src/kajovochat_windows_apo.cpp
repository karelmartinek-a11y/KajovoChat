#include "../../windows_aec_helper/include/kajovochat_windows_aec.h"
#include "../../windows_aec_helper/src/kajovochat_windows_aec.cpp"
#include "../include/kajovochat_windows_apo_capture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

namespace {

struct WindowsApoCaptureState {
    int samplerate;
    double previous_level;
};

static double clamp_capture_value(double value, double minimum, double maximum) {
    return std::max(minimum, std::min(value, maximum));
}

static double block_rms(const std::int16_t* pcm, int sample_count) {
    if (!pcm || sample_count <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int index = 0; index < sample_count; ++index) {
        const double value = static_cast<double>(pcm[index]) / 32768.0;
        sum += value * value;
    }
    return std::sqrt(sum / static_cast<double>(sample_count) + 1e-9);
}

}  // namespace

extern "C" {

void* kajovochat_apo_capture_create(int samplerate) {
    if (samplerate <= 0) {
        return nullptr;
    }
    try {
        return new WindowsApoCaptureState{samplerate, 0.0};
    } catch (...) {
        return nullptr;
    }
}

void kajovochat_apo_capture_destroy(void* handle) {
    auto* state = static_cast<WindowsApoCaptureState*>(handle);
    delete state;
}

int kajovochat_apo_capture_process(
    void* handle,
    const std::int16_t* mic,
    int mic_samples,
    std::int16_t* out_pcm,
    int out_capacity,
    double* out_quality,
    double* out_voice_likelihood,
    int* out_processing_flags
) {
    if (!handle || !mic || !out_pcm || mic_samples <= 0 || out_capacity <= 0) {
        return -1;
    }

    auto* state = static_cast<WindowsApoCaptureState*>(handle);
    const int process_samples = std::min(mic_samples, out_capacity);
    std::copy(mic, mic + process_samples, out_pcm);
    if (out_capacity > process_samples) {
        std::fill(out_pcm + process_samples, out_pcm + out_capacity, 0);
    }

    const double level = block_rms(mic, process_samples);
    const double smoothed = (state->previous_level * 0.7) + (level * 0.3);
    state->previous_level = smoothed;

    if (out_quality) {
        *out_quality = clamp_capture_value(0.16 + smoothed * 0.4, 0.0, 1.0);
    }
    if (out_voice_likelihood) {
        *out_voice_likelihood = clamp_capture_value(smoothed * 2.2, 0.0, 1.0);
    }
    if (out_processing_flags) {
        *out_processing_flags = 0x1;  // system capture contract active
    }
    return 0;
}

}  // extern "C"
