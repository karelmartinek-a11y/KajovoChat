#pragma once

#include <cstdint>

#if defined(_WIN32) && defined(KAJOVOCHAT_WINDOWS_AEC_BUILD_DLL)
#  define KAJOVOCHAT_WINDOWS_AEC_API __declspec(dllexport)
#elif defined(_WIN32)
#  define KAJOVOCHAT_WINDOWS_AEC_API __declspec(dllimport)
#else
#  define KAJOVOCHAT_WINDOWS_AEC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

KAJOVOCHAT_WINDOWS_AEC_API void* kajovochat_aec_create(int samplerate, int filter_length, int max_shift_samples);
KAJOVOCHAT_WINDOWS_AEC_API void kajovochat_aec_destroy(void* handle);
KAJOVOCHAT_WINDOWS_AEC_API int kajovochat_aec_process(
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
    int* out_is_strong
);

#ifdef __cplusplus
}
#endif
