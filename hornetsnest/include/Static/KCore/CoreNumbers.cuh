#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct KCoreData {
    vid_t *src;
    vid_t *dst;
    int   *counter;
};

class KCore : public StaticAlgorithm<HornetGraph> { // What does StaticAlgorithm mean?  Also, why the :?
public:
    KCore(HornetGraph &hornet);
    ~KCore();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }
    void set_hcopy(HornetGraph *h_copy);
    // void get_core_numbers();
    // void max_clique_heuristic();
    

private:
    HostDeviceVar<KCoreData> hd_data;

    long edge_vertex_count;

    load_balancing::VertexBased1 load_balancing;

    TwoLevelQueue<vid_t> vqueue;
    TwoLevelQueue<vid_t> peel_vqueue;
    TwoLevelQueue<vid_t> active_queue;
    TwoLevelQueue<vid_t> iter_queue;
    TwoLevelQueue<vid_t> vertex_frontier;
    // TwoLevelQueue<vid_t> clique_queue;

    vid_t *vertex_pres { nullptr };
    vid_t *vertex_color { nullptr };
    vid_t *vertex_deg { nullptr };
    vid_t **vertex_nbhr_pointer { nullptr };
    uint32_t *vertex_core_number { nullptr };
    int *vertex_nbhr_offsets { nullptr };
    bool *edge_in_clique { nullptr };

    uint32_t *max_clique_size { nullptr };
};

}
