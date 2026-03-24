#pragma once

#include <windows.h>

#include "audioenginebaseapo.h"

#if defined(_WIN32) && defined(KAJOVOCHAT_WINDOWS_APO_BUILD_DLL)
#  define KAJOVOCHAT_WINDOWS_APO_API __declspec(dllexport)
#elif defined(_WIN32)
#  define KAJOVOCHAT_WINDOWS_APO_API __declspec(dllimport)
#else
#  define KAJOVOCHAT_WINDOWS_APO_API
#endif

// {97A08C5E-05D2-4F6D-B5D8-5A4D8F9A7E42}
inline const CLSID CLSID_KajovoApo =
    {0x97a08c5e, 0x05d2, 0x4f6d, {0xb5, 0xd8, 0x5a, 0x4d, 0x8f, 0x9a, 0x7e, 0x42}};

// {FD7F2B29-24D0-4B5C-B177-592C39F9CA10}
inline const IID IID_IAudioProcessingObject =
    {0xfd7f2b29, 0x24d0, 0x4b5c, {0xb1, 0x77, 0x59, 0x2c, 0x39, 0xf9, 0xca, 0x10}};

// {0E5ED805-ABA6-49C3-8F9A-2B8C889C4FA8}
inline const IID IID_IAudioProcessingObjectConfiguration =
    {0x0e5ed805, 0xaba6, 0x49c3, {0x8f, 0x9a, 0x2b, 0x8c, 0x88, 0x9c, 0x4f, 0xa8}};

// {9E1D6A6D-DDBC-4E95-A4C7-AD64BA37846C}
inline const IID IID_IAudioProcessingObjectRT =
    {0x9e1d6a6d, 0xddbc, 0x4e95, {0xa4, 0xc7, 0xad, 0x64, 0xba, 0x37, 0x84, 0x6c}};

// {5FA00F27-ADD6-499A-8A9D-6B98521FA75B}
inline const IID IID_IAudioSystemEffects =
    {0x5fa00f27, 0xadd6, 0x499a, {0x8a, 0x9d, 0x6b, 0x98, 0x52, 0x1f, 0xa7, 0x5b}};

// {BAFE99D2-7436-44CE-9E0E-4D89AFBFFF56}
inline const IID IID_IAudioSystemEffects2 =
    {0xbafe99d2, 0x7436, 0x44ce, {0x9e, 0x0e, 0x4d, 0x89, 0xaf, 0xbf, 0xff, 0x56}};
