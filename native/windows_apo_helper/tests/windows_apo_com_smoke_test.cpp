#include <windows.h>

#include <iostream>

#include "kajovochat_windows_apo_com.h"

int main() {
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        std::cerr << "CoInitializeEx failed\n";
        return 1;
    }

    IClassFactory* factory = nullptr;
    HRESULT hr = DllGetClassObject(CLSID_KajovoApo, IID_IClassFactory, reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || !factory) {
        std::cerr << "DllGetClassObject failed\n";
        CoUninitialize();
        return 2;
    }

    IAudioProcessingObject* apo = nullptr;
    hr = factory->CreateInstance(nullptr, IID_IAudioProcessingObject, reinterpret_cast<void**>(&apo));
    factory->Release();
    if (FAILED(hr) || !apo) {
        std::cerr << "CreateInstance failed\n";
        CoUninitialize();
        return 3;
    }

    HNSTIME latency = -1;
    if (FAILED(apo->GetLatency(&latency)) || latency != 0) {
        std::cerr << "GetLatency failed\n";
        apo->Release();
        CoUninitialize();
        return 4;
    }

    APO_REG_PROPERTIES* props = nullptr;
    if (FAILED(apo->GetRegistrationProperties(&props)) || !props) {
        std::cerr << "GetRegistrationProperties failed\n";
        apo->Release();
        CoUninitialize();
        return 5;
    }
    if (props->u32NumAPOInterfaces < 3) {
        std::cerr << "Unexpected APO interface count\n";
        CoTaskMemFree(props);
        apo->Release();
        CoUninitialize();
        return 6;
    }
    CoTaskMemFree(props);

    UINT32 channel_count = 0;
    if (FAILED(apo->GetInputChannelCount(&channel_count)) || channel_count != 1) {
        std::cerr << "GetInputChannelCount failed\n";
        apo->Release();
        CoUninitialize();
        return 7;
    }

    IAudioProcessingObjectRT* apo_rt = nullptr;
    hr = apo->QueryInterface(IID_IAudioProcessingObjectRT, reinterpret_cast<void**>(&apo_rt));
    if (FAILED(hr) || !apo_rt) {
        std::cerr << "QueryInterface RT failed\n";
        apo->Release();
        CoUninitialize();
        return 8;
    }
    if (apo_rt->CalcOutputFrames(480) != 480 || apo_rt->CalcInputFrames(480) != 480) {
        std::cerr << "Frame calc failed\n";
        apo_rt->Release();
        apo->Release();
        CoUninitialize();
        return 9;
    }
    apo_rt->Release();

    IAudioSystemEffects2* effects = nullptr;
    hr = apo->QueryInterface(IID_IAudioSystemEffects2, reinterpret_cast<void**>(&effects));
    if (FAILED(hr) || !effects) {
        std::cerr << "QueryInterface Effects failed\n";
        apo->Release();
        CoUninitialize();
        return 10;
    }
    LPGUID effect_ids = reinterpret_cast<LPGUID>(0x1);
    UINT effect_count = 123;
    if (FAILED(effects->GetEffectsList(&effect_ids, &effect_count, nullptr)) || effect_count != 0) {
        std::cerr << "GetEffectsList failed\n";
        effects->Release();
        apo->Release();
        CoUninitialize();
        return 11;
    }
    effects->Release();

    if (FAILED(apo->Reset())) {
        std::cerr << "Reset failed\n";
        apo->Release();
        CoUninitialize();
        return 12;
    }

    apo->Release();
    if (DllCanUnloadNow() != S_OK) {
        std::cerr << "DllCanUnloadNow expected S_OK\n";
        CoUninitialize();
        return 13;
    }

    CoUninitialize();
    return 0;
}
