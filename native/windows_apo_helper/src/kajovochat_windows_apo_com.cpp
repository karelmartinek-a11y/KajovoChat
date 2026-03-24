#include <atomic>
#include <cstring>

#include <initguid.h>
#include <strsafe.h>

#include "audioenginebaseapo.h"
#include "kajovochat_windows_apo_com.h"

namespace {

using ULONG_REFCOUNT = std::atomic_ulong;

ULONG_REFCOUNT g_server_locks{0};
ULONG_REFCOUNT g_object_refs{0};

static HRESULT copy_apo_reg_properties(APO_REG_PROPERTIES** ppRegProps) {
    if (!ppRegProps) {
        return E_POINTER;
    }
    constexpr UINT32 kInterfaceCount = 4;
    const std::size_t alloc_size = sizeof(APO_REG_PROPERTIES) + (kInterfaceCount - 1) * sizeof(IID);
    auto* props = static_cast<APO_REG_PROPERTIES*>(CoTaskMemAlloc(alloc_size));
    if (!props) {
        return E_OUTOFMEMORY;
    }
    std::memset(props, 0, alloc_size);
    props->clsid = CLSID_KajovoApo;
    props->Flags = static_cast<APO_FLAG>(APO_FLAG_INPLACE | APO_FLAG_SAMPLESPERFRAME_MUST_MATCH | APO_FLAG_FRAMESPERSECOND_MUST_MATCH | APO_FLAG_BITSPERSAMPLE_MUST_MATCH);
    StringCchCopyW(props->szFriendlyName, _countof(props->szFriendlyName), L"KajovoChat APO skeleton");
    StringCchCopyW(props->szCopyrightInfo, _countof(props->szCopyrightInfo), L"(C) KajovoChat");
    props->u32MajorVersion = 1;
    props->u32MinorVersion = 0;
    props->u32MinInputConnections = 1;
    props->u32MaxInputConnections = 1;
    props->u32MinOutputConnections = 1;
    props->u32MaxOutputConnections = 1;
    props->u32MaxInstances = 1;
    props->u32NumAPOInterfaces = kInterfaceCount;
    props->iidAPOInterfaceList[0] = IID_IAudioProcessingObject;
    props->iidAPOInterfaceList[1] = IID_IAudioProcessingObjectConfiguration;
    props->iidAPOInterfaceList[2] = IID_IAudioProcessingObjectRT;
    props->iidAPOInterfaceList[3] = IID_IAudioSystemEffects2;
    *ppRegProps = props;
    return S_OK;
}

class KajovoApo final
    : public IAudioProcessingObject
    , public IAudioProcessingObjectConfiguration
    , public IAudioProcessingObjectRT
    , public IAudioSystemEffects2 {
public:
    KajovoApo() {
        ++g_object_refs;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) {
            return E_POINTER;
        }
        *ppvObject = nullptr;
        if (riid == IID_IUnknown || riid == IID_IAudioProcessingObject) {
            *ppvObject = static_cast<IAudioProcessingObject*>(this);
        } else if (riid == IID_IAudioProcessingObjectConfiguration) {
            *ppvObject = static_cast<IAudioProcessingObjectConfiguration*>(this);
        } else if (riid == IID_IAudioProcessingObjectRT) {
            *ppvObject = static_cast<IAudioProcessingObjectRT*>(this);
        } else if (riid == IID_IAudioSystemEffects || riid == IID_IAudioSystemEffects2) {
            *ppvObject = static_cast<IAudioSystemEffects2*>(this);
        } else {
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        return ++_ref_count;
    }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG remaining = --_ref_count;
        if (remaining == 0) {
            delete this;
        }
        return remaining;
    }

    HRESULT STDMETHODCALLTYPE Reset(void) override {
        _locked = false;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetLatency(HNSTIME* pTime) override {
        if (!pTime) {
            return E_POINTER;
        }
        *pTime = 0;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetRegistrationProperties(APO_REG_PROPERTIES** ppRegProps) override {
        return copy_apo_reg_properties(ppRegProps);
    }

    HRESULT STDMETHODCALLTYPE Initialize(UINT32 cbDataSize, BYTE* pbyData) override {
        (void)cbDataSize;
        (void)pbyData;
        _initialized = true;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE IsInputFormatSupported(
        IAudioMediaType* pOppositeFormat,
        IAudioMediaType* pRequestedInputFormat,
        IAudioMediaType** ppSupportedInputFormat) override {
        (void)pOppositeFormat;
        if (ppSupportedInputFormat) {
            *ppSupportedInputFormat = nullptr;
        }
        return pRequestedInputFormat ? S_OK : E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE IsOutputFormatSupported(
        IAudioMediaType* pOppositeFormat,
        IAudioMediaType* pRequestedOutputFormat,
        IAudioMediaType** ppSupportedOutputFormat) override {
        (void)pOppositeFormat;
        if (ppSupportedOutputFormat) {
            *ppSupportedOutputFormat = nullptr;
        }
        return pRequestedOutputFormat ? S_OK : E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetInputChannelCount(UINT32* pu32ChannelCount) override {
        if (!pu32ChannelCount) {
            return E_POINTER;
        }
        *pu32ChannelCount = 1;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE LockForProcess(
        UINT32 u32NumInputConnections,
        APO_CONNECTION_DESCRIPTOR** ppInputConnections,
        UINT32 u32NumOutputConnections,
        APO_CONNECTION_DESCRIPTOR** ppOutputConnections) override {
        (void)u32NumInputConnections;
        (void)ppInputConnections;
        (void)u32NumOutputConnections;
        (void)ppOutputConnections;
        _locked = true;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE UnlockForProcess(void) override {
        _locked = false;
        return S_OK;
    }

    void STDMETHODCALLTYPE APOProcess(
        UINT32 u32NumInputConnections,
        APO_CONNECTION_PROPERTY** ppInputConnections,
        UINT32 u32NumOutputConnections,
        APO_CONNECTION_PROPERTY** ppOutputConnections) override {
        if (!u32NumInputConnections || !u32NumOutputConnections || !ppInputConnections || !ppOutputConnections) {
            return;
        }
        APO_CONNECTION_PROPERTY* input = ppInputConnections[0];
        APO_CONNECTION_PROPERTY* output = ppOutputConnections[0];
        if (!input || !output) {
            return;
        }
        output->u32ValidFrameCount = input->u32ValidFrameCount;
        output->u32BufferFlags = BUFFER_VALID;
    }

    UINT32 STDMETHODCALLTYPE CalcInputFrames(UINT32 u32OutputFrameCount) override {
        return u32OutputFrameCount;
    }

    UINT32 STDMETHODCALLTYPE CalcOutputFrames(UINT32 u32InputFrameCount) override {
        return u32InputFrameCount;
    }

    HRESULT STDMETHODCALLTYPE GetEffectsList(LPGUID* ppEffectsIds, UINT* pcEffects, HANDLE Event) override {
        (void)Event;
        if (!pcEffects) {
            return E_POINTER;
        }
        if (ppEffectsIds) {
            *ppEffectsIds = nullptr;
        }
        *pcEffects = 0;
        return S_OK;
    }

private:
    ~KajovoApo() {
        --g_object_refs;
    }

    ULONG_REFCOUNT _ref_count{1};
    bool _initialized{false};
    bool _locked{false};
};

class KajovoApoClassFactory final : public IClassFactory {
public:
    KajovoApoClassFactory() {
        ++g_object_refs;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) {
            return E_POINTER;
        }
        *ppvObject = nullptr;
        if (riid == IID_IUnknown || riid == IID_IClassFactory) {
            *ppvObject = static_cast<IClassFactory*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        return ++_ref_count;
    }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG remaining = --_ref_count;
        if (remaining == 0) {
            delete this;
        }
        return remaining;
    }

    HRESULT STDMETHODCALLTYPE CreateInstance(IUnknown* pUnkOuter, REFIID riid, void** ppvObject) override {
        if (pUnkOuter != nullptr) {
            return CLASS_E_NOAGGREGATION;
        }
        auto* apo = new (std::nothrow) KajovoApo();
        if (!apo) {
            return E_OUTOFMEMORY;
        }
        const HRESULT hr = apo->QueryInterface(riid, ppvObject);
        apo->Release();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE LockServer(BOOL fLock) override {
        if (fLock) {
            ++g_server_locks;
        } else {
            --g_server_locks;
        }
        return S_OK;
    }

private:
    ~KajovoApoClassFactory() {
        --g_object_refs;
    }
    ULONG_REFCOUNT _ref_count{1};
};

}  // namespace

extern "C" HRESULT WINAPI DllCanUnloadNow() {
    return (g_object_refs.load() == 0 && g_server_locks.load() == 0) ? S_OK : S_FALSE;
}

extern "C" HRESULT WINAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, LPVOID* ppv) {
    if (!ppv) {
        return E_POINTER;
    }
    *ppv = nullptr;
    if (rclsid != CLSID_KajovoApo) {
        return CLASS_E_CLASSNOTAVAILABLE;
    }
    auto* factory = new (std::nothrow) KajovoApoClassFactory();
    if (!factory) {
        return E_OUTOFMEMORY;
    }
    const HRESULT hr = factory->QueryInterface(riid, ppv);
    factory->Release();
    return hr;
}

extern "C" BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance;
    (void)reserved;
    (void)reason;
    return TRUE;
}
