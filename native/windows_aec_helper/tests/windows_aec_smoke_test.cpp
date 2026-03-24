#include "kajovochat_windows_aec.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

double rms(const std::vector<std::int16_t>& samples) {
    if (samples.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::int16_t sample : samples) {
        const double value = static_cast<double>(sample);
        sum += value * value;
    }
    return std::sqrt(sum / static_cast<double>(samples.size()));
}

}  // namespace

int main() {
    void* handle = kajovochat_aec_create(24000, 256, 960);
    if (!handle) {
        std::cerr << "create failed\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-9000, 9000);
    std::vector<std::int16_t> ref(24000);
    for (auto& sample : ref) {
        sample = static_cast<std::int16_t>(dist(rng));
    }

    const int delay_samples = 288;
    std::vector<std::int16_t> mic(ref.size(), 0);
    for (std::size_t i = static_cast<std::size_t>(delay_samples); i < ref.size(); ++i) {
        mic[i] = static_cast<std::int16_t>(std::lround(static_cast<double>(ref[i - delay_samples]) * 0.72));
    }

    std::vector<std::int16_t> out(mic.size(), 0);
    double quality = 0.0;
    double improvement = 0.0;
    double residual = 0.0;
    int strong = 0;

    if (kajovochat_aec_process(
            handle,
            mic.data(),
            static_cast<int>(mic.size()),
            ref.data(),
            static_cast<int>(ref.size()),
            12,
            out.data(),
            static_cast<int>(out.size()),
            &quality,
            &improvement,
            &residual,
            &strong) != 0) {
        std::cerr << "first process failed\n";
        kajovochat_aec_destroy(handle);
        return 2;
    }

    if (kajovochat_aec_process(
            handle,
            mic.data(),
            static_cast<int>(mic.size()),
            ref.data(),
            static_cast<int>(ref.size()),
            12,
            out.data(),
            static_cast<int>(out.size()),
            &quality,
            &improvement,
            &residual,
            &strong) != 0) {
        std::cerr << "second process failed\n";
        kajovochat_aec_destroy(handle);
        return 3;
    }

    const double input_rms = rms(mic);
    const double output_rms = rms(out);

    kajovochat_aec_destroy(handle);

    if (!(output_rms < input_rms * 0.9)) {
        std::cerr << "expected RMS reduction, input=" << input_rms << " output=" << output_rms << "\n";
        return 4;
    }
    if (!(improvement > 0.08)) {
        std::cerr << "expected improvement, got " << improvement << "\n";
        return 5;
    }
    if (!(quality >= 0.0 && quality <= 1.0)) {
        std::cerr << "quality out of range\n";
        return 6;
    }
    return 0;
}
