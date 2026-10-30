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

#include "test_harness.hpp"
#include "../infra/peak_snapshot.hpp"
#include "../infra/peak_heap.hpp"
#include "../memory_infra/lb_arena.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../memory_infra/paged_vector.hpp"
#include "../memory_infra/paged_hash_index.hpp"
#include "../project_paths.hpp"
#include <algorithm>
#include <fstream>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace {
struct Span { uintptr_t start; std::size_t size; uint32_t owner,kind; };
struct Record { std::vector<Span> spans; uint32_t next=1; };
struct CaptureScope {
    Record& record;
    explicit CaptureScope(Record& r):record(r) {
        ASSERT_TRUE(!gl::peak_snapshot::active());
        gl::peak_snapshot::sink={&record,
            [](void* context,uint32_t,const char*){return static_cast<Record*>(context)->next++;},
            [](void* context,uint32_t owner,const void* p,std::size_t n,uint32_t kind){
                static_cast<Record*>(context)->spans.push_back({reinterpret_cast<uintptr_t>(p),n,owner,kind});
            }};
    }
    ~CaptureScope(){gl::peak_snapshot::sink={};gl::peak_snapshot::currentOwner=0;}
};
}
TEST(peak_snapshot, position_preserves_first_maximum) {
#ifdef _WIN32
    char value[8]{};
    if(GetEnvironmentVariableA("GL_PEAK_ASSERT_PROBE",value,sizeof(value))) {
        gl::peak_snapshot::Position invalid;invalid.observe(-1,0);
    }
#endif
    gl::peak_snapshot::Position p;
    p.observe(10,1);p.observe(30,2);p.observe(30,3);p.observe(20,4);
    ASSERT_EQ(p.ordinal,uint64_t(4));ASSERT_EQ(p.maximumOrdinal,uint64_t(2));
    ASSERT_EQ(p.maximumBytes,int64_t(30));ASSERT_EQ(p.maximumPhase,2);
}
TEST(peak_snapshot, scopes_restore_and_disabled_walk_does_not_emit) {
    ASSERT_TRUE(!gl::peak_snapshot::active());
    int value=7;
    ASSERT_EQ(gl::peak_snapshot::measure("value",value,[&]{return value;}),7);
    Record record;
    { CaptureScope capture(record);
      { gl::peak_snapshot::Scope outer("outer");
        const auto id=gl::peak_snapshot::currentOwner;
        {gl::peak_snapshot::Scope inner("inner");ASSERT_NE(id,gl::peak_snapshot::currentOwner);}
        ASSERT_EQ(id,gl::peak_snapshot::currentOwner);
        gl::peak_snapshot::note(&value,sizeof(value));
        gl::peak_snapshot::note(nullptr,0);
      }
      ASSERT_EQ(gl::peak_snapshot::currentOwner,uint32_t(0));
    }
    ASSERT_EQ(record.spans.size(),std::size_t(1));ASSERT_EQ(record.spans[0].size,sizeof(int));
}
TEST(peak_snapshot, two_vectors_keep_separate_half_page_owners) {
    gl::GlobalMemoryManager manager;manager.init({2<<20,1<<18,1<<11});
    gl::LbArena arena(&manager);gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<int> a(&arena,&dirty),b(&arena,&dirty);
    a.push_back(7);b.push_back(9);
    Record record;
    {CaptureScope capture(record);
     gl::peak_snapshot::measure("a",a,[&]{return a.liveBytes();});
     gl::peak_snapshot::measure("b",b,[&]{return b.liveBytes();});}
    std::vector<Span> held;
    for(auto row:record.spans)if(row.kind==1)held.push_back(row);
    ASSERT_EQ(held.size(),std::size_t(2));ASSERT_EQ(held[0].size,std::size_t(2048));
    ASSERT_EQ(held[1].size,std::size_t(2048));ASSERT_NE(held[0].owner,held[1].owner);
    ASSERT_NE(held[0].start,held[1].start);
    ASSERT_TRUE(held[0].start+2048<=held[1].start || held[1].start+2048<=held[0].start);
}
TEST(peak_snapshot, roots_restore_and_pool_arena_slot_walks_keep_state) {
    int context=4;
    {gl::peak_snapshot::RootScope outer(&context,nullptr,nullptr);
     {gl::peak_snapshot::RootScope inner(nullptr,nullptr,nullptr);ASSERT_EQ(gl::peak_snapshot::rootContext,nullptr);}
     ASSERT_EQ(gl::peak_snapshot::rootContext,&context);}
    ASSERT_EQ(gl::peak_snapshot::rootContext,nullptr);
    gl::GlobalMemoryManager manager;manager.init({2<<20,1<<18,1<<11});
    gl::LbArena arena(&manager);const auto page=arena.allocPage();
    gl::LbStore store(&manager,32,8);
    void* first=store.allocateSlot();void* freed=store.allocateSlot();store.freeSlot(freed);
    Record record;std::size_t live=0;
    {CaptureScope capture(record);
     manager.observeResidentRanges();arena.observeResidentRanges();
     store.observeLiveSlots([&](void* object){ASSERT_EQ(object,first);++live;});}
    ASSERT_EQ(live,std::size_t(1));ASSERT_TRUE(!record.spans.empty());
    ASSERT_EQ(store.slotsInUse(),std::size_t(1));
    store.freeSlot(first);arena.freePage(page);
}
TEST(peak_snapshot, external_capture_writer_round_trip_when_requested) {
#ifdef _WIN32
    // Ordinary unit runs exercise the disabled state without external tools.
    // The explicit integration-test mode exercises the real collector below.
    char requested[8]{};
    if (GetEnvironmentVariableA("GL_PEAK_TEST_CAPTURE", requested, sizeof(requested)) == 0) {
        ASSERT_TRUE(!gl::peak_snapshot::active());
        return;
    }
    ASSERT_EQ(std::string(requested), std::string("1"));
    // Unique test artifacts remain in the ignored debug directory for audit.
    const std::string batch="unit_"+std::to_string(GetCurrentProcessId());
    const auto debug=gl::projectRoot()/".debug";
    std::filesystem::create_directories(debug);
    {std::ofstream plan(debug/("peak_scan_"+batch+".txt"));plan<<"1 0 0\n";ASSERT_TRUE(plan.good());}
    wchar_t python[32768]{};
    ASSERT_TRUE(GetEnvironmentVariableW(L"GL_PEAK_PYTHON",python,32768)>0);
    ASSERT_TRUE(SetEnvironmentVariableA("GL_PEAK_MODE","capture"));
    gl::peak_snapshot::initialize(batch.c_str());
    int value=7;
    {gl::peak_snapshot::RootScope root(&value,[](void*){},[](void* context){
        gl::peak_snapshot::Scope owner("testObject");
        for(int i=0;i<4100;++i)gl::peak_snapshot::note(context,sizeof(int));
     });
     gl::peak_snapshot::checkpoint(0,0);}
    gl::peak_snapshot::finish();
    ASSERT_TRUE(SetEnvironmentVariableA("GL_PEAK_MODE",nullptr));
    const auto prefix=debug/("peak_capture_"+batch);
    struct OwnerRow {uint32_t parent;char name[96];};
    std::ifstream owners(prefix.string()+".owners",std::ios::binary);
    OwnerRow owner{};uint32_t id=0,testOwner=0;
    while(owners.read(reinterpret_cast<char*>(&owner),sizeof(owner))) {
        if(owner.parent==0 && std::string(owner.name)=="testObject")testOwner=id;
        ++id;
    }
    ASSERT_NE(testOwner,uint32_t(0));
    struct RangeRow {uint64_t address,bytes;uint32_t owner,kind;};
    std::ifstream ranges(prefix.string()+".ranges",std::ios::binary);
    RangeRow range{};std::size_t observed=0;
    while(ranges.read(reinterpret_cast<char*>(&range),sizeof(range))) {
        ASSERT_LT(range.owner,id);
        if(range.owner==testOwner) {
            ASSERT_EQ(range.address,static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&value)));
            ASSERT_EQ(range.bytes,uint64_t(sizeof(value)));++observed;
        }
    }
    ASSERT_EQ(observed,std::size_t(4100));
    ASSERT_TRUE(std::filesystem::file_size(prefix.string()+".pages")>0);
#endif
}

TEST(peak_snapshot, typed_heap_fields_include_large_aligned_buffers_and_nodes) {
    gl::LogicalEntity entity;
    entity.category="test";entity.signature=std::string(9000,'x');
    entity.elements={"small",std::string(128,'y')};
    entity.contrapositives.push_back({{"source"},"forward","reverse"});
    entity.conjunctionRefutations.emplace_back("left","right");
    entity.supportingImplications={"support"};entity.composedSupportingImplications={"composed"};
    gl::CompiledExpressionMap definitions;definitions.emplace("definition",entity);
    ce::CoreExpressionConfig config;config.definition=std::filesystem::path(L"C:/snapshot/test");
    config.definitionSets["slot"]={"pattern",true};config.inputIndices={1};
    std::vector<int> large(3000,7);
    std::tuple<int,std::string> tuple{4,"tuple"};
    std::set<std::string> values{"a","longer"};
    Record record;
    {CaptureScope capture(record);
     gl::peak_snapshot::field("definitions",definitions);
     gl::peak_snapshot::field("configuration",config);
     gl::peak_snapshot::field("largeVector",large);
     gl::peak_snapshot::field("tuple",tuple);
     gl::peak_snapshot::field("set",values);}
    const auto address=reinterpret_cast<uintptr_t>(large.data());
    bool elements=false,prefix=false;
    for(const auto& span:record.spans) {
        elements=elements || (span.start==address && span.size==large.size()*sizeof(int));
        prefix=prefix || (span.start<address && span.start+span.size==address);
    }
    ASSERT_TRUE(elements);
#if defined(_MSC_VER) && defined(USE_MIMALLOC)
    // allocation() observes the Microsoft allocator's alignment header.
    ASSERT_TRUE(prefix);
#else
    // Other allocator implementations expose only capacity and live bytes.
    ASSERT_FALSE(prefix);
#endif
    ASSERT_EQ(large.size(),std::size_t(3000));ASSERT_EQ(large[2999],7);
    ASSERT_EQ(definitions.at("definition").signature,std::string(9000,'x'));
}
TEST(peak_snapshot, transient_roots_and_vector_storage_restore) {
    int value=4;
    {gl::peak_snapshot::ExtraRootScope root("value",&value,nullptr,[](void*){});
     ASSERT_EQ(gl::peak_snapshot::ExtraRootScope::head,&root);
     {gl::peak_snapshot::ExtraRootScope child("child",&value,nullptr,[](void*){});
      ASSERT_EQ(gl::peak_snapshot::ExtraRootScope::head,&child);}
     ASSERT_EQ(gl::peak_snapshot::ExtraRootScope::head,&root);}
    ASSERT_EQ(gl::peak_snapshot::ExtraRootScope::head,nullptr);
    std::vector<int> empty;
    Record record;
    {CaptureScope capture(record);ASSERT_EQ(gl::peak_snapshot::vectorBytes("empty",empty),uint64_t(0));}
    ASSERT_EQ(record.spans.size(),std::size_t(1));
}


TEST(peak_snapshot, ordinary_page_resolution_has_no_observation_even_with_an_armed_sink) {
    gl::GlobalMemoryManager manager; manager.init({2<<20,1<<18,1<<11});
    gl::LbArena arena(&manager); const int32_t page = arena.allocPage();
    Record record;
    { CaptureScope capture(record);
      char* mutablePage = arena.pageAt(page);
      const gl::LbArena& readOnly = arena;
      ASSERT_EQ(static_cast<const char*>(mutablePage), readOnly.pageAt(page));
    }
    ASSERT_TRUE(record.spans.empty());
    arena.freePage(page);
}

TEST(peak_snapshot, explicit_page_census_covers_all_directory_depths) {
    struct TinyArena {
        int32_t pages[64][4]{};
        int32_t pageBytes() const { return 16; }
        const char* pageAt(int32_t vid) const {
            ASSERT_TRUE(vid >= 0 && vid < 64);
            return reinterpret_cast<const char*>(pages[vid]);
        }
    };
    for (int32_t count : {1, 4, 5, 17}) {
        TinyArena arena; int32_t next = count;
        int32_t root = 0;
        int32_t expectedDirectories = 0;
        if (count > 1) {
            root = next++; ++expectedDirectories;
            if (count <= 4) {
                for (int32_t i = 0; i < count; ++i) arena.pages[root][i] = i;
            } else {
                for (int32_t outer = 0; outer < (count + 15) / 16; ++outer) {
                    int32_t middle = root;
                    if (count > 16) { middle = next++; ++expectedDirectories; arena.pages[root][outer] = middle; }
                    for (int32_t group = 0; group < 4 && outer * 16 + group * 4 < count; ++group) {
                        const int32_t lower = next++; ++expectedDirectories;
                        arena.pages[middle][group] = lower;
                        for (int32_t i = 0; i < 4 && outer * 16 + group * 4 + i < count; ++i)
                            arena.pages[lower][i] = outer * 16 + group * 4 + i;
                    }
                }
            }
        }
        Record record; int32_t visited = 0;
        { CaptureScope capture(record);
#ifdef _WIN32
          char probe[8]{};
          if (GetEnvironmentVariableA("GL_PEAK_DIRECTORY_ASSERT_PROBE", probe, sizeof(probe)))
              gl::peak_snapshot::observePagedPages(arena, root, 0, 2, [](int32_t, const char*) {});
#endif
          gl::peak_snapshot::observePagedPages(arena, root, count, 2,
              [&](int32_t index, const char* address) {
                  ASSERT_EQ(index, visited++); ASSERT_EQ(address, arena.pageAt(index));
              });
        }
        ASSERT_EQ(visited, count);
        std::set<uintptr_t> directories, data;
        for (const auto& span : record.spans) {
            ASSERT_EQ(span.size, std::size_t(16));
            if (span.kind == 2) directories.insert(span.start);
            else { ASSERT_EQ(span.kind, uint32_t(1)); data.insert(span.start); }
        }
        ASSERT_EQ(directories.size(), static_cast<std::size_t>(expectedDirectories));
        ASSERT_EQ(data.size(), static_cast<std::size_t>(count));
    }
}

TEST(peak_snapshot, gpu_host_capture_retains_recycled_committed_blocks) {
    gl::GlobalMemoryManager manager;
    manager.init({8 * 262144LL, 262144, 2048, gl::PoolKind::GpuHost});
    char* range = manager.acquireGpuHostRange(3);
    manager.releaseGpuHostRange(range, 3);
    const auto live = manager.blocksInUse();
    Record record;
    { CaptureScope capture(record); manager.observeResidentRanges(); }
    uint64_t committed = 0;
    for (const auto& span : record.spans) if (span.kind == 4) committed += span.size;
    ASSERT_EQ(committed, static_cast<uint64_t>(manager.gpuHostCommittedBytes()));
    ASSERT_EQ(committed, 4 * 262144ull);
    ASSERT_EQ(manager.blocksInUse(), live);
    ASSERT_EQ(live, 1);
}
