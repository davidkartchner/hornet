#include <Device/Util/Timer.cuh>
#include "Static/KCore/KCore.cuh"
#include <fstream>

#include <nvToolsExt.h>

#define DELETE 1

using namespace timer;
namespace hornets_nest {

KCore::KCore(HornetGraph &hornet) : // Constructor
                        StaticAlgorithm(hornet), 
                        vqueue(hornet),
                        peel_vqueue(hornet),
                        active_queue(hornet),
                        iter_queue(hornet),
                        load_balancing(hornet)
                        {

    gpu::allocate(vertex_pres, hornet.nV()); // Creates arrays of length n for whether vertices are present, their current degree, and their color
    gpu::allocate(vertex_color, hornet.nV());
    gpu::allocate(vertex_deg, hornet.nV());
    gpu::allocate(hd_data().src,    hornet.nE()); // Allocate space for endpoints of edges and counter
    gpu::allocate(hd_data().dst,    hornet.nE());
    gpu::allocate(hd_data().counter, 1);
}

KCore::~KCore() { // Deconstructor, frees up all GPU memory used by algorithm
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_deg);
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
}


// Why are we creating separate structs for all of these?  is it because each struct can only have 1 operator?
// I think that these structs are defined to do things with the member data of KCore object.  But why structs?
struct ActiveVertices { // Create data structure to keep track of active vertices
    vid_t *vertex_pres; // What is vid_t?  Is it a vertex array?
    vid_t *deg;
    TwoLevelQueue<vid_t> active_queue; // What is a TwoLevelQueue?  I think it has something to do with load balancing...

    OPERATOR(Vertex &v) { // What does OPERATOR mean?  Does it just do this for everything in the struct?
        vid_t id = v.id(); // I think that it simply looks at a given vertex and if it has nonzero degree, pops it in the active queue, marks it as present, and adds its degree to the degree array.
        if (v.degree() > 0) {
            vertex_pres[id] = 1;
            active_queue.insert(id);
            deg[id] = v.degree();
        }
    }
};

struct PeelVertices { // Data structure to keep track of vertices to peel off
    vid_t *vertex_pres;
    vid_t *deg;
    uint32_t peel;
    TwoLevelQueue<vid_t> peel_queue;
    TwoLevelQueue<vid_t> iter_queue;
    
    OPERATOR(Vertex &v) { // Mark present vertices with insufficicnt degree for peeling
        vid_t id = v.id();
        if (vertex_pres[id] == 1 && deg[id] <= peel) {
            vertex_pres[id] = 2;
            peel_queue.insert(id);
            iter_queue.insert(id);
        }
    }
};

struct RemovePres { // Struct to remove vertices marked by PeelVertices
    vid_t *vertex_pres;
    
    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (vertex_pres[id] == 2) {
            vertex_pres[id] = 0;
        }
    }
};

struct DecrementDegree {  // Struct to decrement degrees of every vertex attached to a removed vertex
    vid_t *deg;

    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();
        atomicAdd(&deg[src], -1);
        atomicAdd(&deg[dst], -1);
    }
};

struct UpdateCoreNumber{ // Update the core number of each vertex peeled off in current iteration
    vid_t *core_number;

    OPERATOR(Vertex &v, int &peel){
        vid_t id = v.id();
        if (vertex_pres[id] == 2){
            core_number[id] = peel;
        } 
    }
}

struct ExtractSubgraph { // Struct to extract subgraph of vertices that get peeled off?  (Why not the other way around?)
    HostDeviceVar<KCoreData> hd;
    vid_t *vertex_pres;
    
    OPERATOR(Vertex &v, Edge &e) {
        vid_t src = v.id();
        vid_t dst = e.dst_id();
        if (vertex_pres[src] == 2 && vertex_pres[dst] == 2) {
            int spot = atomicAdd(hd().counter, 1); // What do atomic operations do?
            hd().src[spot] = src; // We do still keep the vertex numbers of the marked vertices
            hd().dst[spot] = dst;
        }
    }
};

struct GetDegOne { // Mark all vertices of degree 1
    TwoLevelQueue<vid_t> vqueue;
    vid_t *vertex_color;

    OPERATOR(Vertex &v) {
        vid_t id = v.id();
        if (v.degree() == 1) {
            vqueue.insert(id);
            vertex_color[id] = 1;
        }
    }
};

struct DegOneEdges {
    HostDeviceVar<KCoreData> hd;
    vid_t *vertex_color;

    OPERATOR(Vertex &v, Edge &e) { // I think that this edge should be coming from the given vertex
        vid_t src = v.id();
        vid_t dst = e.dst_id();

        if (vertex_color[src] || vertex_color[dst]) { // If one of the endpoints has degree 1, add to subgraph hd
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            if (!vertex_color[src] || !vertex_color[dst]) { // Get different subgraph for ones for which both endpoints are not degree 1
                int spot_rev = atomicAdd(hd().counter, 1);
                hd().src[spot_rev] = dst;
                hd().dst[spot_rev] = src;
            }
        }
    }
};

void KCore::reset() {  // What does this do?
    vqueue.swap();
    peel_vqueue.swap();
    active_queue.swap();
    iter_queue.swap();
}

// Delete all edges in given batch
void oper_bidirect_batch(HornetGraph &hornet, vid_t *src, vid_t *dst, 
                         int size, uint8_t op) {

    gpu::BatchUpdate batch_update(src, dst, size, gpu::BatchType::DEVICE); //What does specifying this GPU BatchUpdate object do?

    // Delete edges in the forward and backward directions.
    hornet.deleteEdgeBatch(batch_update, gpu::batch_property::IN_PLACE); // What do you mean "in forward and backward directions?"
}

void core_numbers_new(HornetGraph &hornet, 
    HostDeviceVar<KCoreData>& hd, 
    TwoLevelQueue<vid_t> &peel_queue,
    TwoLevelQueue<vid_t> &active_queue,
    TwoLevelQueue<vid_t> &iter_queue,
    load_balancing::VertexBased1 load_balancing,
    vid_t *deg,
    vid_t *vertex_pres,
    vid_t *core_number,
    uint32_t *max_peel,
    int *batch_size){

    
    forAllVertices(hornet, ActiveVertices { vertex_pres, deg, active_queue }); // Get active vertices in parallel (puts in input queue)
    active_queue.swap(); // Swap input to output queue

    int n_active = active_queue.size();
    uint32_t peel = 0;

    while (n_active > 0) {
        // Why do we use a particular queue in forAllVertices?  Does it go through all vertices in this queue?
        forAllVertices(hornet, active_queue, 
                PeelVertices { vertex_pres, deg, peel, peel_queue, iter_queue} );
        iter_queue.swap();
        
        n_active -= iter_queue.size();
    
        if (iter_queue.size() == 0) {
            peel++; // Once we have removed all vertices with core <= current peel, increment peel
            peel_queue.swap();
            // if (n_active > 0) {
                // Shouldn't this be the peel_queue? If not, why?
                // Would this be faster if it were peel_queue?
                forAllVertices(hornet, active_queue, RemovePres { vertex_pres });
                forAllVertices(hornet, active_queue, RemovePres { vertex_pres }); // Why do we never update the active queue? Does this modify its data in some way?
            // }
        } else {
            forAllEdges(hornet, iter_queue, DecrementDegree { deg }, load_balancing); // Go through vertices in iter_queue and decrement the degree of their nbhrs
        }
    }
}


// I think this writes the peel of all of the edges to a json file
void json_dump(vid_t *src, vid_t *dst, uint32_t *peel, uint32_t peel_edges) {
    std::ofstream output_file;
    output_file.open("output.txt");
    
    output_file << "{\n";
    for (uint32_t i = 0; i < peel_edges; i++) {
        output_file << "\"" << src[i] << "," << dst[i] << "\": " << peel[i];
        if (i < peel_edges - 1) {
            output_file << ",";
        }
        output_file << "\n";
    }
    output_file << "}";
    output_file.close();
}

void KCore::run() {
    omp_set_num_threads(72);
    vid_t *src     = new vid_t[hornet.nE()];
    vid_t *dst     = new vid_t[hornet.nE()];
    uint32_t len = hornet.nE() / 2 + 1;
    uint32_t *peel = new uint32_t[hornet.nE()];
    uint32_t peel_edges = 0;
    uint32_t ne = hornet.nE();
    std::cout << "ne: " << ne << std::endl;

    auto pres = vertex_pres;
    auto deg = vertex_deg;
    auto color = vertex_color;
    
    // What does this do?
    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ deg[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ color[i] = 0; } );

    Timer<DEVICE> TM;
    TM.start();

    /* Begin degree 1 vertex preprocessing optimization */ 

    // Find vertices of degree 1.
    forAllVertices(hornet, GetDegOne { vqueue, vertex_color });
    vqueue.swap();

    // Find the edges incident to these vertices.
    gpu::memsetZero(hd_data().counter);  // reset counter. 
    forAllEdges(hornet, vqueue, 
                    DegOneEdges { hd_data, vertex_color }, load_balancing);

    // Mark edges with peel 1.
    int peel_one_count = 0;
    cudaMemcpy(&peel_one_count, hd_data().counter, sizeof(int), cudaMemcpyDeviceToHost);
    #pragma omp parallel for
    for (int i = 0; i < peel_one_count; i++) {
        peel[i] = 1;
    }

    cudaMemcpy(src, hd_data().src, peel_one_count * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(dst, hd_data().dst, peel_one_count * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);

    peel_edges = (uint32_t)peel_one_count;

    // Delete peel 1 edges.
    oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, peel_one_count, DELETE);

    /* Begin running main kcore algorithm */
    while (peel_edges < ne) {
        uint32_t max_peel = 0;
        int batch_size = 0;

        kcores_new(hornet, hd_data, peel_vqueue, active_queue, iter_queue, 
                   load_balancing, vertex_deg, vertex_pres, &max_peel, &batch_size);
        std::cout << "max_peel: " << max_peel << "\n";

        if (batch_size > 0) {
            cudaMemcpy(src + peel_edges, hd_data().src, 
                       batch_size * sizeof(vid_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(dst + peel_edges, hd_data().dst, 
                       batch_size * sizeof(vid_t), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (int i = 0; i < batch_size; i++) {
                peel[peel_edges + i] = max_peel;
            }

            peel_edges += batch_size;
        }
        oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, batch_size, DELETE);
    }
    TM.stop();
    TM.print("KCore");
    json_dump(src, dst, peel, peel_edges);
}

void KCore::release() {
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_deg);
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    hd_data().src = nullptr;
    hd_data().dst = nullptr;
}
}

