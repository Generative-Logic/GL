/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschränkt)

 This program is free software : you can redistribute it and /or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.If not, see < https://www.gnu.org/licenses/>.

 ------------------------------------------------------------------------------

 This software is also available under a commercial license.For details,
 see: https://generative-logic.com/license

 Contributions to this project must be made under the terms of the
 Contributor License Agreement(CLA).See the project's CONTRIBUTING.md file.*/

#include "peak_snapshot.hpp"
#include "memory_census.hpp"
#include "../project_paths.hpp"
#include "../parameters.hpp"
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif
#include <cstdio>
#include <cstring>
#include <new>
#include <filesystem>
#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace gl { namespace peak_snapshot {
namespace {
// Opt-in memory measurement: process-local diagnostic state; disabled unless requested.
Position position;
int mode = 0;
uint64_t target = 0;
int targetPhase = -1;
bool captured = false;
char batchName[96]{};
#ifdef _WIN32
struct SpanRecord { uint64_t address, bytes; uint32_t owner, kind; };
struct Owner { uint32_t parent; char name[96]; };
struct Writer {
    HANDLE ranges, names;
    Owner owners[8192]{};
    uint32_t slots[16384]{};
    uint32_t count = 1;
    SpanRecord buffer[4096]{};
    uint32_t used = 0;
};
/// @brief Persist all bytes to a diagnostic file.
/// @details A failed or short operating-system write asserts immediately.
/// @param file Writable file handle.
/// @param data Stable byte buffer.
/// @param bytes Exact byte count fitting DWORD.
/// @return Nothing.
void writeBytes(HANDLE file, const void* data, DWORD bytes) {
    DWORD written=0;
    const BOOL ok=WriteFile(file,data,bytes,&written,nullptr);
    assert(ok && written==bytes);
}
/// @brief Flush buffered physical ownership intervals.
/// @details The buffer is reused only after its complete write.
/// @param writer Active capture writer.
/// @return Nothing.
void flush(Writer& writer) {
    if (writer.used) { writeBytes(writer.ranges,writer.buffer,writer.used*sizeof(SpanRecord)); writer.used=0; }
}
/// @brief Intern a named child without allocating on the heap.
/// @details Exact parent/name equality determines identity; collisions probe onward.
/// @param context Active Writer.
/// @param parent Existing owner identifier.
/// @param name Field label shorter than the fixed label capacity.
/// @return Stable child identifier.
uint32_t enter(void* context,uint32_t parent,const char* name) {
    auto& w=*static_cast<Writer*>(context);
    assert(parent<w.count && std::strlen(name)<sizeof(Owner::name));
    uint32_t hash=2166136261u ^ parent;
    for (const char* p=name;*p;++p) hash=(hash^static_cast<unsigned char>(*p))*16777619u;
    uint32_t slot=hash & 16383;
    while(w.slots[slot]) {
        const uint32_t id=w.slots[slot];
        if(w.owners[id].parent==parent && std::strcmp(w.owners[id].name,name)==0) return id;
        slot=(slot+1)&16383;
    }
    assert(w.count<8192);
    const uint32_t id=w.count++;
    w.slots[slot]=id; w.owners[id].parent=parent;
    std::memcpy(w.owners[id].name,name,std::strlen(name)+1);
    return id;
}
/// @brief Buffer one exact ownership interval.
/// @details The offline partitioner intersects ranges with the saved resident pages.
/// @param context Active Writer.
/// @param owner Existing hierarchy identifier.
/// @param address First owned byte.
/// @param bytes Number of owned bytes.
/// @param kind Specificity and physical role.
/// @return Nothing.
void span(void* context,uint32_t owner,const void* address,std::size_t bytes,uint32_t kind) {
    auto& w=*static_cast<Writer*>(context);
    assert(owner<w.count && bytes>0 && kind<=11);
    const uintptr_t start=reinterpret_cast<uintptr_t>(address);
    assert(start+bytes>start);
    w.buffer[w.used++]={start,bytes,owner,kind};
    if(w.used==4096) flush(w);
}

/// @brief Record the allocation/free partition of a locked Windows heap.
/// @details Region bounds are supplied by HeapWalk; no private heap layout is
/// guessed. The recorded page set supplies residency independently of commitment.
/// @param heap Heap already locked by this thread before the external snapshot.
/// @return Nothing.
void observeWindowsHeap(HANDLE heap) {
    char label[48]{};std::snprintf(label,sizeof(label),"heap_%p",heap);
    Scope runtime("WindowsHeaps");Scope owner(label);
    bool regions[256]{};
    PROCESS_HEAP_ENTRY entry{};
    while(HeapWalk(heap,&entry)) {
        if(entry.wFlags & PROCESS_HEAP_REGION) {
            regions[entry.iRegionIndex]=true;
            Scope region("regionMetadataAndCapacity");note(entry.lpData,entry.cbData,9);
        } else if(entry.wFlags & PROCESS_HEAP_UNCOMMITTED_RANGE) {
            // Uncommitted address space has no resident bytes to attribute.
        } else {
            const bool busy=(entry.wFlags & PROCESS_HEAP_ENTRY_BUSY)!=0;
            if(busy && !regions[entry.iRegionIndex]) {
                MEMORY_BASIC_INFORMATION region{};
                assert(VirtualQuery(entry.lpData,&region,sizeof(region))==sizeof(region));
                Scope metadata("largeAllocationRegion");note(region.BaseAddress,region.RegionSize,9);
            }
            Scope blocks(busy?"allocatedBlocks":"freeBlocks");
            note(entry.lpData,entry.cbData,busy?11:10);
        }
    }
    assert(GetLastError()==ERROR_NO_MORE_ITEMS && "Windows heap snapshot enumeration failed");
}

/// @brief Observe main-thread mimalloc areas and allocated blocks.
/// @details The main thread owns this heap. The allocator's public visitor merges
/// already-freed lists but does not allocate or free a live user object. Other
/// thread heaps are not falsely classified as free; their regions stay explicit.
/// @return Nothing.
void observeMainHeap() {
#ifdef USE_MIMALLOC
    Scope allocator("mimalloc");
    const bool ok=mi_heap_visit_blocks(mi_heap_get_default(),true,
        [](const mi_heap_t*,const mi_heap_area_t* area,void* block,std::size_t bytes,void*) {
            if(block) {Scope owner("unassignedLiveBlocks");note(block,bytes,11);}
            else {Scope owner("mainThreadAreaCapacity");note(area->blocks,area->committed,9);}
            return true;
        },nullptr);
    assert(ok && "mimalloc snapshot enumeration failed");
#endif
}

/// @brief Identify resident pages belonging to mimalloc's reserved regions.
/// @details Reads the externally saved page addresses. Region membership alone
/// does not distinguish metadata, retained capacity or another thread's live data.
/// @param path Saved resident-page list.
/// @return Nothing.
void observeAllocatorRegions(const std::filesystem::path& path) {
#ifdef USE_MIMALLOC
    HANDLE file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    assert(file!=INVALID_HANDLE_VALUE);
    uint64_t pages[4096]{};DWORD count=0;
    Scope allocator("mimalloc");Scope regions("uninspectedSegments");
    while(true) {
        assert(ReadFile(file,pages,sizeof(pages),&count,nullptr));
        assert(count%sizeof(uint64_t)==0);
        if(count==0) break;
        for(DWORD i=0;i<count/sizeof(uint64_t);++i)
            if(mi_is_in_heap_region(reinterpret_cast<const void*>(pages[i])))
                note(reinterpret_cast<const void*>(pages[i]),4096,4);
    }
    CloseHandle(file);
#else
    static_cast<void>(path);
#endif
}

/// @brief Capture resident pages externally before walking any owning structure.
/// @details Background proof storage is quiescent. The helper's buffers belong to
/// another process. The subsequent address walk cannot add pages to the saved set.
/// @return Nothing.
void capture() {
    assert(prepareRoot && collectRoot && !active());
    prepareRoot(rootContext);
    for(auto* root=ExtraRootScope::head;root;root=root->previous)
        if(root->prepare) root->prepare(root->context);
    const auto prefix=projectRoot()/".debug"/(std::string("peak_capture_")+batchName);
    wchar_t python[32768]{};
    const DWORD n=GetEnvironmentVariableW(L"GL_PEAK_PYTHON",python,32768);
    assert(n>0 && n<32768);
    const auto pagePath=std::filesystem::path(prefix.wstring()+L".pages");
    const auto rangePath=prefix.wstring()+L".ranges";
    const auto ownerPath=prefix.wstring()+L".owners";
    const auto script=projectRoot()/".scripts"/"peak_memory_collect.py";
    std::wstring command=L"\""+std::wstring(python)+L"\" \""+script.wstring()
        +L"\" --pid "+std::to_wstring(GetCurrentProcessId())+L" --prefix \""+prefix.wstring()+L"\"";
    STARTUPINFOW startup{}; startup.cb=sizeof(startup);
    PROCESS_INFORMATION process{};
    const BOOL started=CreateProcessW(python,command.data(),nullptr,nullptr,FALSE,
        CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,projectRoot().c_str(),&startup,&process);
    assert(started && "peak capture helper must start");
    HANDLE heaps[1024]{};
    const DWORD heapCount=GetProcessHeaps(1024,heaps);
    DWORD locked=0;
    if(heapCount>0 && heapCount<=1024)
        while(locked<heapCount && HeapLock(heaps[locked])) ++locked;
    const bool heapsLocked=heapCount>0 && heapCount<=1024 && locked==heapCount;
    // Always release the owned helper even on a preparation failure.
    const DWORD previousSuspend=ResumeThread(process.hThread);
    assert(previousSuspend==1);
    assert(WaitForSingleObject(process.hProcess,INFINITE)==WAIT_OBJECT_0);
    DWORD exitCode=1; assert(GetExitCodeProcess(process.hProcess,&exitCode));
    CloseHandle(process.hThread); CloseHandle(process.hProcess);
    if(!heapsLocked || exitCode!=0) {
        while(locked>0) {--locked;assert(HeapUnlock(heaps[locked]));}
        assert(heapsLocked && "all Windows heaps must lock before snapshot");
        assert(exitCode==0 && "peak capture helper failed");
    }
    // Snapshot is now fixed. Observer allocations below cannot enter its page list.
    void* storage=VirtualAlloc(nullptr,sizeof(Writer),MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    assert(storage);
    auto& writer=*new(storage) Writer{};
    writer.ranges=CreateFileW(rangePath.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    writer.names=CreateFileW(ownerPath.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    assert(writer.ranges!=INVALID_HANDLE_VALUE && writer.names!=INVALID_HANDLE_VALUE);
    sink={&writer,enter,span}; currentOwner=0;
    for(DWORD i=0;i<heapCount;++i) observeWindowsHeap(heaps[i]);
    while(locked>0) {--locked;assert(HeapUnlock(heaps[locked]));}
    observeMainHeap();
    observeAllocatorRegions(pagePath);
    {Scope diagnostics("SnapshotPreparation");
      {Scope owner("commandLine");allocation(command.data(),(command.capacity()+1)*sizeof(wchar_t),(command.size()+1)*sizeof(wchar_t));}
      {Scope owner("pythonPathBuffer");note(python,sizeof(python),3);}
      {Scope owner("heapHandles");note(heaps,sizeof(heaps),3);}
      {Scope owner("rangePath");allocation(rangePath.data(),(rangePath.capacity()+1)*sizeof(wchar_t),(rangePath.size()+1)*sizeof(wchar_t));}
      {Scope owner("ownerPath");allocation(ownerPath.data(),(ownerPath.capacity()+1)*sizeof(wchar_t),(ownerPath.size()+1)*sizeof(wchar_t));}
    }
    collectRoot(rootContext);
    for(auto* root=ExtraRootScope::head;root;root=root->previous) {
        Scope owner(root->name);root->collect(root->context);
    }
    sink={}; currentOwner=0;
    flush(writer);
    writeBytes(writer.names,writer.owners,writer.count*sizeof(Owner));
    CloseHandle(writer.ranges); CloseHandle(writer.names);
    writer.~Writer(); assert(VirtualFree(storage,0,MEM_RELEASE));
    captured=true;
}
#endif
}

/// @brief Observe one standard C++ allocation and its exact allocator overhead.
/// @details Microsoft's standard allocator manually aligns buffers at least 4 KiB;
/// its own read-only adjustment recovers the original operator-new pointer. The
/// requested capacity, live prefix, alignment header and usable-size tail remain
/// distinct intervals. Current observed GL element types require at most default
/// new alignment; over-aligned owners need their explicit allocation contract.
/// @param address Standard-allocator allocation address.
/// @param capacity Requested owned bytes.
/// @param used Live prefix, bounded by capacity.
/// @return Nothing.
void allocation(const void* address,std::size_t capacity,std::size_t used) {
    assert(used<=capacity);
    if(!active() || capacity==0) return;
    assert(address);
    note(address,capacity,1);note(address,used,3);
#if defined(_MSC_VER) && defined(USE_MIMALLOC)
    void* original=const_cast<void*>(address);
    std::size_t requested=capacity;
    if(capacity>=std::_Big_allocation_threshold)
        std::_Adjust_manually_vector_aligned(original,requested);
    const auto start=reinterpret_cast<uintptr_t>(original);
    const auto data=reinterpret_cast<uintptr_t>(address);
    const auto usable=mi_usable_size(original);
    assert(start<=data && usable>=requested && data+capacity<=start+usable);
    {Scope prefix("allocationPrefix");note(original,data-start,3);}
    {Scope rounding("allocatorRounding");note(reinterpret_cast<const void*>(data+capacity),start+usable-data-capacity,3);}
#endif
}

/// @brief Initialize the optional two-pass census for one batch.
/// @details Capture requires a position file written by the first pass.
/// @param batch Native batch identifier.
/// @return Nothing.
void initialize(const char* batch) {
#ifdef _WIN32
    char setting[32]{};
    const DWORD n=GetEnvironmentVariableA("GL_PEAK_MODE",setting,sizeof(setting));
    if(n==0) return;
    assert(n<sizeof(setting) && (std::strcmp(setting,"scan")==0 || std::strcmp(setting,"capture")==0));
    assert(mode==0 && std::strlen(batch)<sizeof(batchName));
    for(const char* p=batch;*p;++p)
        assert((*p>='A' && *p<='Z') || (*p>='a' && *p<='z') || (*p>='0' && *p<='9') || *p=='_');
    mode=std::strcmp(setting,"scan")==0?1:2;
    std::memcpy(batchName,batch,std::strlen(batch)+1);
    if(mode==2) {
        const auto path=projectRoot()/".debug"/(std::string("peak_scan_")+batchName+".txt");
        FILE* file=nullptr; assert(_wfopen_s(&file,path.c_str(),L"rb")==0 && file);
        unsigned long long ordinal=0; long long bytes=0;
        assert(fscanf_s(file,"%llu %d %lld",&ordinal,&targetPhase,&bytes)==3);
        assert(ordinal>0 && targetPhase>=0 && bytes>=0);
        target=ordinal; assert(std::fclose(file)==0);
    }
#else
    static_cast<void>(batch);
#endif
}
/// @brief Advance the sampled execution position and capture the selected replay.
/// @details Only positions with a complete root binding can be selected.
/// @param bytes Current working-set size.
/// @param phase Named process phase identifier.
/// @return Nothing.
void checkpoint(int64_t bytes,int phase) {
    if(mode==0 || collectRoot==nullptr) return;
    position.observe(bytes,phase);
    if(mode==2 && position.ordinal==target) {
        assert(phase==targetPhase && !captured);
#ifdef _WIN32
        capture();
#endif
    }
}
/// @brief Save each pass's position and process high-water separately.
/// @details The second pass need not equal the first; it must reach its target.
/// @return Nothing.
void finish() {
    if(mode==0) return;
    assert(position.maximumOrdinal>0 && (mode!=2 || captured));
    const auto peaks=memory_census::readProcessPeaks();
    const auto path=projectRoot()/".debug"/(std::string(mode==1?"peak_scan_":"peak_replay_")+batchName+".txt");
    FILE* file=nullptr;
#ifdef _WIN32
    assert(_wfopen_s(&file,path.c_str(),L"wb")==0 && file);
#else
    file=std::fopen(path.c_str(),"wb"); assert(file);
#endif
    assert(std::fprintf(file,"%llu %d %lld\nprocess_peak %lld\npositions %llu\ntarget %llu\n",
        static_cast<unsigned long long>(position.maximumOrdinal),position.maximumPhase,
        static_cast<long long>(position.maximumBytes),static_cast<long long>(peaks.peakWorkingSetBytes),
        static_cast<unsigned long long>(position.ordinal),static_cast<unsigned long long>(target))>0);
    assert(std::fclose(file)==0);
}
} }
