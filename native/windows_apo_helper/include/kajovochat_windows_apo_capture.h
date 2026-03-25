#pragma once

#include <cstdint>

#if defined(_WIN32) && defined(KAJOVOCHAT_WINDOWS_APO_BUILD_DLL)
#  define KAJOVOCHAT_WINDOWS_APO_CAPTURE_API __declspec(dllexport)
#elif defined(_WIN32)
#  define KAJOVOCHAT_WINDOWS_APO_CAPTURE_API __declspec(dllimport)
#else
#  define KAJOVOCHAT_WINDOWS_APO_CAPTURE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

KAJOVOCHAT_WINDOWS_APO_CAPTURE_API void* kajovochat_apo_capture_create(int samplerate);
KAJOVOCHAT_WINDOWS_APO_CAPTURE_API void kajovochat_apo_capture_destroy(void* handle);
KAJOVOCHAT_WINDOWS_APO_CAPTURE_API int kajovochat_apo_capture_process(
    void* handle,
    const std::int16_t* mic,
    int mic_samples,
    std::int16_t* out_pcm,
    int out_capacity,
    double* out_quality,
    double* out_voice_likelihood,
    int* out_processing_flags
);

#ifdef __cplusplus
}
#endif
