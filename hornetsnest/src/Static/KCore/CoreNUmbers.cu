#include <Device/Util/Timer.cuh>
#include "Static/KCore/CoreNumbers.cuh"
#include <fstream>
#include <set>

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
                        vertex_frontier(hornet),
                        clique_queue(hornet),
                        load_balancing(hornet)
                        {

    gpu::allocate(vertex_pres, hornet.nV()); // Creates arrays of length n for whether vertices are present, their current degree, and their color
    gpu::allocate(vertex_color, hornet.nV());
    gpu::allocate(vertex_deg, hornet.nV());
    gpu::allocate(vertex_core_number, hornet.nV()); // Keep track of core numbers of vertices
    gpu::allocate(vertex_clique_number, hornet.nV()); // Keep track of clique numbers of vertices
    gpu::allocate(hd_data().src,    hornet.nE()); // Allocate space for endpoints of edges and counter
    gpu::allocate(hd_data().dst,    hornet.nE());
    gpu::allocate(hd_data().counter, 1);
}

KCore::~KCore() { // Deconstructor, frees up all GPU memory used by algorithm
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_deg);
    gpu::free(vertex_core_number);
    gpu::free(vertex_clique_number); 
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

struct FixedCoreNumVertices{
    vid_t *core_number;
    uint32_t curr_coreness;
    TwoLevelQueue<vid_t> vertex_frontier;

    OPERATOR(Vertex &v){
        id = v.id();
        if(core_number[id] == curr_coreness[id]){
            vertex_frontier.insert(id);
        }
    }

};

// Function to check if a we can add a vertex to a clique
bool check_clique(std::set<vid_t> nbhrs, std::set<vid_t> curr_clique){
    for (std::set<vid_t>::iterator i=curr_clique.begin(); i != curr_clique.end(); ++i){
        if (nbhrs.find(*i) == nbhrs.end()){
            return false;
        }
    }
    return true;
}

struct GetLocalClique{
    vid_t *clique_number;
    vid_t *core_number;
    uint32_t curr_max_size;
    TwoLevelQueue<vid_t> clique_queue;
    

    OPERATOR(Vertex &v){
        // construct std::set of neighbors of current vertex
        std::set<vid_t> *curr_clique;
        curr_clique.insert(v.id());
        int clique_size = 1;

        // Make sure vertex has coreness >= max_clique_size before inserting
        for (degree_t i=0; i<v.degree(); i++){
            u = v.edge(i).dst();
            vid_t id = u.id();
            std::set<vid_t> *nbhrs;

            // Check if nbhrs with coreness >= max_clique_size are part of a clique
            if (core_number[id] >= curr_max_clique){
                #pragma omp parallel for
                for (degree_t j=0; j<u.degree(); j++){
                    curr_nbhr_id = u.neighbor_id(j);

                    if (core_number[curr_nbhr_id] >= curr_max_clique){
                        nbhrs.insert(curr_nbhr_id);
                    } 
                }
                if (check_clique(&nbhrs, &curr_clique)){
                    curr_clique.insert(id);
                }
            }
        }
        uint32_t curr_size = curr_clique.size();
        if (curr_size > curr_max_clique){
            clique_queue.insert(v);
            clique_number[v] = curr_size;
            // curr_max_size = curr_size;
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

struct CoreRemoveVertices { // Data structure to keep track of vertices to peel off
    vid_t *vertex_pres;
    vid_t *core_number;
    uint32_t peel;
    
    OPERATOR(Vertex &v) { // Mark present vertices with insufficicnt degree for peeling
        vid_t id = v.id();
        if (vertex_pres[id] >= 1 && core_number[id] < peel) {
            vertex_pres[id] = 0;
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
    vid_t *vertex_pres;
    uint32_t peel;

    OPERATOR(Vertex &v){
        vid_t id = v.id();
        if (vertex_pres[id] == 2){
            core_number[id] = peel;
        } 
    }
};

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

void get_core_numbers(HornetGraph &hornet, 
    HostDeviceVar<KCoreData>& hd, 
    TwoLevelQueue<vid_t> &peel_queue,
    TwoLevelQueue<vid_t> &active_queue,
    TwoLevelQueue<vid_t> &iter_queue,
    load_balancing::VertexBased1 load_balancing,
    vid_t *deg,
    vid_t *vertex_pres,
    vid_t *core_number,
    uint32_t *max_peel){

    
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
                // Shouldn't this be the peel_queue? If not, why?
                // Would this be faster if it were peel_queue?
                forAllVertices(hornet, active_queue, UpdateCoreNumber { core_number, vertex_pres, peel });
                forAllVertices(hornet, active_queue, RemovePres { vertex_pres }); // Why do we never update the active queue? Does this modify its data in some way?
        } else {
            forAllEdges(hornet, iter_queue, DecrementDegree { deg }, load_balancing); // Go through vertices in iter_queue and decrement the degree of their nbhrs
        }
    }
}


void max_clique_heuristic(HornetGraph &hornet,
    HostDeviceVar<KCoreData>& hd, 
    TwoLevelQueue<vid_t> &peel_queue,
    TwoLevelQueue<vid_t> &active_queue,
    TwoLevelQueue<vid_t> &iter_queue,
    TwoLevelQueue<vid_t> &clique_queue,
    TwoLevelQueue<vid_t> &vertex_frontier,
    load_balancing::VertexBased1 load_balancing,
    vid_t *deg,
    vid_t *vertex_pres,
    vid_t *core_number,
    vid_t *clique_number,
    uint32_t *max_clique_size){

    uint32_t max_peel = 0;
    get_core_numbers(hornet, hd, peel_queue, active_queue, iter_queue, 
        load_balancing, deg, vertex_pres, core_number, &max_peel);
    // Get active vertices (with clique number > 0)
    forAllVertices(hornet, ActiveVertices { vertex_pres, core_number, active_queue }); // Get active vertices in parallel (puts in input queue)
    active_queue.swap(); // Swap input to output queue
    int n_active = active_queue.size();
    uint32_t peel = 0;
    uint32_t curr_clique_size = 1;
    peel = max_peel;

    
    while (peel >= curr_clique_size & n_active > 0) {
        forAllVertices(hornet, active_queue, FixedCoreNumVertices{ core_number, peel, vertex_frontier });   
        std::cout << "Vertex Frontier Size before swap: " << vertex_frontier.size() << std::endl;     
        vertex_frontier.swap();
        std::cout << "Vertex Frontier Size after swap: " << vertex_frontier.size() << std::endl;   
        n_active -= vertex_frontier.size();
    
        if (vertex_frontier.size() > 0) {
            // Get clique numbers of vertices of frontier core number
             forAllVertices(hornet, vertex_frontier, GetLocalClique { clique_number, core_number, curr_clique_size, clique_queue});
             clique_queue.swap();

            if (clique_queue.size() > 0){
             // Update curr_clique_size
                for (int i=0; i < clique_queue.size(); i++){
                    if (clique_number[i] > curr_clique_size) {
                        curr_clique_size = clique_number[i];
                    }
                }

                // Remove vertices without sufficiently high core number 
                peel = curr_clique_size; 
                forAllVertices(hornet, active_queue, CoreRemoveVertices {vertex_pres, core_number, peel});
            }
        } else {
            forAllEdges(hornet, iter_queue, DecrementDegree { deg }, load_balancing); // Go through vertices in iter_queue and decrement the degree of their nbhrs
        }
        peel--;
    }
    *max_clique_size = curr_clique_size;
    std::cout << "Max Clique Found: " << max_clique_size << std::endl;
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

    // /* Begin degree 1 vertex preprocessing optimization */ 

    // // Find vertices of degree 1.
    // forAllVertices(hornet, GetDegOne { vqueue, vertex_color });
    // vqueue.swap();

    // // Find the edges incident to these vertices.
    // gpu::memsetZero(hd_data().counter);  // reset counter. 
    // forAllEdges(hornet, vqueue, 
    //                 DegOneEdges { hd_data, vertex_color }, load_balancing);

    // // Mark edges with peel 1.
    // int peel_one_count = 0;
    // cudaMemcpy(&peel_one_count, hd_data().counter, sizeof(int), cudaMemcpyDeviceToHost);
    // #pragma omp parallel for
    // for (int i = 0; i < peel_one_count; i++) {
    //     peel[i] = 1;
    // }

    cudaMemcpy(src, hd_data().src, hornet.nV() * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(dst, hd_data().dst, hornet.nV() * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);

    uint32_t max_clique_size;
    max_clique_heuristic(hornet, hd_data, peel_vqueue, active_queue, iter_queue, clique_queue, vertex_frontier,
                       load_balancing, vertex_deg, vertex_pres, vertex_core_number, vertex_clique_number, &max_clique_size);



    // peel_edges = (uint32_t)peel_one_count;

    // // Delete peel 1 edges.
    // oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, peel_one_count, DELETE);

    /* Begin running main kcore algorithm */
    // while (peel_edges < ne) {
    //     uint32_t max_peel = 0;
    //     int batch_size = 0;

    //     kcores_new(hornet, hd_data, peel_vqueue, active_queue, iter_queue, 
    //                load_balancing, vertex_deg, vertex_pres, &max_peel, &batch_size);
    //     std::cout << "max_peel: " << max_peel << "\n";

    //     if (batch_size > 0) {
    //         cudaMemcpy(src + peel_edges, hd_data().src, 
    //                    batch_size * sizeof(vid_t), cudaMemcpyDeviceToHost);

    //         cudaMemcpy(dst + peel_edges, hd_data().dst, 
    //                    batch_size * sizeof(vid_t), cudaMemcpyDeviceToHost);

    //         #pragma omp parallel for
    //         for (int i = 0; i < batch_size; i++) {
    //             peel[peel_edges + i] = max_peel;
    //         }

    //         peel_edges += batch_size;
    //     }
    //     oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, batch_size, DELETE);
    // }
    TM.stop();
    TM.print("KCore");
}

void KCore::release() {
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_deg);
    gpu::free(vertex_core_number);
    gpu::free(vertex_clique_number); 
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    hd_data().src = nullptr;
    hd_data().dst = nullptr;
}
}
