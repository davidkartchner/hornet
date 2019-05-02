#include <Device/Util/Timer.cuh>
#include "Static/KCore/CoreNumbers.cuh"
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
                        vertex_frontier(hornet),
                        load_balancing(hornet)
                        {

    gpu::allocate(vertex_pres, hornet.nV()); // Creates arrays of length n for whether vertices are present, their current degree, and their color
    gpu::allocate(vertex_color, hornet.nV());
    gpu::allocate(vertex_deg, hornet.nV());
    gpu::allocate(vertex_core_number, hornet.nV()); // Keep track of core numbers of vertices
    gpu::allocate(vertex_nbhr_pointer, hornet.nV()); // Keep track of clique numbers of vertices
    gpu::allocate(hd_data().src,    hornet.nE()); // Allocate space for endpoints of edges and counter
    gpu::allocate(hd_data().dst,    hornet.nE());
    gpu::allocate(hd_data().counter, 1);
    gpu::allocate(edge_in_clique,    hornet.nE());
    gpu::allocate(vertex_nbhr_offsets,    hornet.nV());
    // gpu::allocate(max_clique_size, 1);
}

KCore::~KCore() { // Deconstructor, frees up all GPU memory used by algorithm
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_deg);
    gpu::free(vertex_core_number);
    gpu::free(vertex_nbhr_pointer); 
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    gpu::free(edge_in_clique);
    gpu::free(vertex_nbhr_offsets);
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
    uint32_t *core_number;
    uint32_t curr_coreness;
    TwoLevelQueue<vid_t> vertex_frontier;

    OPERATOR(Vertex &v){
        vid_t id = v.id();
        if(core_number[id] == curr_coreness){
            vertex_frontier.insert(id);
        }
    }

};


struct InitializeOffsets{
    int *vertex_nbhr_offsets;
    // vid_t *vertex_degrees;
    HostDeviceVar<KCoreData> hd;

    OPERATOR(Vertex &v){
        uint32_t deg = v.degree();
        vid_t id = v.id();
        vertex_nbhr_offsets[id] = atomicAdd(hd().counter, deg);
        // atomicAdd(hd().counter, deg);
    }

};
// bool check_clique(Vertex &v, Vertex &u){
//     #pragma omp parallel for
//     bool is_clique = true;
//     for (Edge::iterator i = v.edge_begin(); i != v.edge_end(); i++){
//         bool found = false;
//         if (WeightT *i.weight() == 1){
//             vid_t id = *i.src_id();

//             #pragma omp parallel for
//             for (Edge::iterator j = u.edge_begin(); j != u.edge_end(); j++){
//                 if (*j.dst_id() == id){
//                     bool found = true;
//                 }
//             }
//             if (!found){
//                 is_clique = false;
//             }
//         }
//     }
//     return is_clique;
// }

// __device__ __forceinline__
// void workPerBlock(vid_t numVertices,
//                   vid_t* __restrict__ outMpStart,
//                   vid_t* __restrict__ outMpEnd,
//                   int blockSize) {
//     vid_t       verticesPerMp = numVertices / gridDim.x;
//     vid_t     remainderBlocks = numVertices % gridDim.x;
//     vid_t   extraVertexBlocks = (blockIdx.x > remainderBlocks) ? remainderBlocks
//                                                                : blockIdx.x;
//     vid_t regularVertexBlocks = (blockIdx.x > remainderBlocks) ?
//                                     blockIdx.x - remainderBlocks : 0;

//     vid_t mpStart = (verticesPerMp + 1) * extraVertexBlocks +
//                      verticesPerMp * regularVertexBlocks;
//     *outMpStart   = mpStart;
//     *outMpEnd     = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
// }



// template<typename HornetDevice>
// __global__ void GetLocalClique(
//     HornetDevice hornet ,
//     uint32_t *core_number,
//     int *vertex_nbhr_offsets,
//     bool *edge_in_clique,
//     uint32_t max_clique_size,
//     int N){

// 	int k = threadIdx.x + blockIdx.x *blockDim.x;
//     if (k>=N) return;
     
//     for 



//     }


// run(){
//             const int BLOCK_SIZE = 256;
//             int blocks = (elements)/BLOCK_SIZE + (((elements)%BLOCK_SIZE)?1:0);

//               GetLocalClique<<<blocks,BLOCK_SIZE>>>(hornet.device_side(),...,elements);

	
// }

struct GetPointersAndDegrees{
    vid_t **vertex_nbhr_pointer;
    vid_t *deg;

    OPERATOR(Vertex &v){
        vid_t id = v.id();
        vertex_nbhr_pointer[id] = v.neighbor_ptr();
        deg[id] = v.degree();
    }
};


struct GetLocalClique{
    uint32_t *core_number;
    int *vertex_nbhr_offsets;
    bool *edge_in_clique;
    vid_t **vertex_nbhr_pointer;
    vid_t *deg;
    uint32_t max_clique_size;


    OPERATOR(Vertex &v){
        // construct std::set of neighbors of current vertex
        /* I want a vertex properties that include:
            whether a vertex was visited on current sweep and
            whether it's in the max clique
        */

        uint32_t curr_size = 1;

        // Make sure vertex has coreness >= max_clique_size before inserting
        vid_t* vNeighPtr = v.neighbor_ptr();
        vid_t v_id = v.id();
        vid_t length_v = deg[v_id];
        int offset = vertex_nbhr_offsets[v_id];
        printf("Offset: %d \n", offset);
        for (vid_t i = 0; i < length_v; i++){
            vid_t u_id = vNeighPtr[i]; 
            // Vertex u = hornet.vertex(u_id); // How can I get this?
            // printf( "before pointer assignment" );
            // Get nbhr info for u
            vid_t* uNeighPtr = vertex_nbhr_pointer[u_id];
            vid_t length_u = deg[u_id];
            // printf("after pointer assignment\n");

            // Loop through neibhbors of v currently in clique and check to see if also nbhrs of u
            #pragma omp parallel for
            bool is_clique = true;
            for (vid_t j = 0; j < length_v; j++){
                // printf("Starting inner loop \n");
                bool found = false;
                
                if (edge_in_clique[offset + j]){
                    printf("entering first if statement \n");
                    vid_t w_id = vNeighPtr[j];

                    // Check if 
                    #pragma omp parallel for
                    for (vid_t k = 0; k < length_u; k++){
                        if (uNeighPtr[k] == w_id){
                            found = true;
                            break;
                        }
                    }
                    if (!found){
                        is_clique = false;
                    }
                }
            }
            
            // Check if nbhrs with coreness >= max_clique_size are part of a clique
            // If so, increment clique size
            if (is_clique){
                edge_in_clique[offset + i] = true;
                curr_size += 1;
                atomicMax(&max_clique_size, curr_size);
            }
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
    uint32_t *core_number;
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
    uint32_t *core_number;
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

struct SmallCoreEdges {
    vid_t *vertex_pres;
    HostDeviceVar<KCoreData> hd;
    

     OPERATOR(Vertex &v, Edge &e){
        vid_t src = v.id();
        vid_t dst = e.dst_id();

        if (!vertex_pres[src] || !vertex_pres[dst]){
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;
            if (vertex_pres[src] || vertex_pres[dst]){
                int spot_rev = atomicAdd(hd().counter, 1);
                hd().src[spot_rev] = dst;
                hd().dst[spot_rev] = src;
            }
        }
     }
};

struct ResetWeight { // Reset edge weight to 0 after finding current iteration of cliques

    OPERATOR(Vertex &v, Edge &e){
        e.set_weight(0);
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
    TwoLevelQueue<vid_t> &peel_queue,
    TwoLevelQueue<vid_t> &active_queue,
    TwoLevelQueue<vid_t> &iter_queue,
    load_balancing::VertexBased1 load_balancing,
    vid_t *deg,
    vid_t *vertex_pres,
    uint32_t *core_number,
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
    *max_peel = peel;
}


void max_clique_heuristic(HornetGraph &hornet,
    HostDeviceVar<KCoreData>& hd, 
    TwoLevelQueue<vid_t> &vertex_frontier,
    load_balancing::VertexBased1 load_balancing,
    vid_t *vertex_pres,
    uint32_t *core_number,
    int *vertex_nbhr_offsets,
    vid_t **vertex_nbhr_pointer,
    vid_t *vertex_degree,
    bool *edge_in_clique,
    uint32_t *max_clique_size, 
    uint32_t *peel,
    int *batch_size){

    // uint32_t curr_peel =  *peel;
    uint32_t clique_size = 0;
    clique_size++;
    // while (vertex_frontier.size() == 0){
        forAllVertices(hornet, FixedCoreNumVertices{ core_number, *peel, vertex_frontier });   
        std::cout << "Vertex Frontier Size before swap: " << vertex_frontier.size() << std::endl;     
        vertex_frontier.swap();
        std::cout << "Vertex Frontier Size after swap: " << vertex_frontier.size() << std::endl;   

        if (vertex_frontier.size() > 0) {
            // Get clique numbers of vertices of frontier core number
            forAllVertices(hornet, vertex_frontier, GetLocalClique { core_number, vertex_nbhr_offsets, edge_in_clique, vertex_nbhr_pointer, vertex_degree, clique_size });

            // Remove vertices without sufficiently high core number 
            // uint32_t *curr_max = clique_size; 
            forAllVertices(hornet, CoreRemoveVertices { vertex_pres, core_number, clique_size });
            forAllEdges(hornet, SmallCoreEdges { vertex_pres, hd }, load_balancing);
        }
    //     peel--;
    // }
    int size = 0;
    cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);
    *batch_size = size;
    *max_clique_size = clique_size;
    // std::cout << "Max Clique Found: " << max_clique_size << std::endl;
}



void KCore::run() {
    omp_set_num_threads(72);
    vid_t *src     = new vid_t[hornet.nE()];
    vid_t *dst     = new vid_t[hornet.nE()];
    uint32_t len = hornet.nE() / 2 + 1;
    // uint32_t peel = new uint32_t;
    uint32_t ne = hornet.nE();
    std::cout << "ne: " << ne << std::endl;
    // uint32_t max_clique_size = new uint32_t;
    

    
    auto pres = vertex_pres;
    auto deg = vertex_deg;
    auto color = vertex_color;
    auto core_number = vertex_core_number;
    auto offsets = vertex_nbhr_offsets;
    vid_t** nbhr_pointer = vertex_nbhr_pointer;
    auto clique_edges = edge_in_clique;
    
    
    // What does this do?
    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ deg[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ color[i] = 0; } );
    forAllnumV(hornet, [=] __device__ (int i){ offsets[i] = 0; } );
    // forAllnumV(hornet, [=] __device__ (int* i){ nbhr_pointer[i] = 0; } );
    forAllnumE(hornet, [=] __device__ (int i){ clique_edges[i] = false; } );
    

    Timer<DEVICE> TM;
    Timer<DEVICE> Tclique;
    TM.start();


    forAllVertices(hornet, InitializeOffsets { offsets, hd_data });
    /* Begin degree 1 vertex preprocessing optimization */ 
    // Find vertices of degree 1.
    forAllVertices(hornet, GetDegOne { vqueue, vertex_color });
    vqueue.swap();

    gpu::memsetZero(hd_data().counter);

    // Find the edges incident to these vertices.
    gpu::memsetZero(hd_data().counter);  // reset counter. 
    forAllEdges(hornet, vqueue, 
                    DegOneEdges { hd_data, vertex_color }, load_balancing);

    // Mark edges with peel 1.
    int peel_one_count = 0;
    cudaMemcpy(&peel_one_count, hd_data().counter, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(src, hd_data().src, hornet.nV() * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(dst, hd_data().dst, hornet.nV() * sizeof(vid_t), 
                    cudaMemcpyDeviceToHost);

    // Delete peel 1 edges.
    oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, peel_one_count, DELETE);


    // Get vertex core numbers
    uint32_t peel = 0;
    uint32_t max_clique_size = 0;
    max_clique_size++;
    uint32_t temp_clique_size;
    get_core_numbers(hornet, peel_vqueue, active_queue, iter_queue, 
        load_balancing, deg, vertex_pres, core_number, &peel);
    gpu::memsetZero(hd_data().counter);
    TM.stop();
    TM.print("CoreNumbers");

    // // Get active vertices (with clique number > 0)
    // forAllVertices(hornet, CoreActiveVertices { vertex_pres, core_number, active_queue }); // Get active vertices in parallel (puts in input queue)
    // active_queue.swap(); // Swap input to output queue

    // int n_active = active_queue.size();
    
    Tclique.start();
    // Begin actual clique heuristic algorithm
    while (peel >= max_clique_size) {
        int batch_size = 0;

        std::cout << "Vertex Pointers Not Initialized Yet" << std::endl;
        forAllVertices(hornet, GetPointersAndDegrees { nbhr_pointer, deg });
        forAllVertices(hornet, InitializeOffsets { offsets, hd_data });
        std::cout << "Initialized Vertex Pointers" << std::endl;

        max_clique_heuristic(hornet, hd_data, vertex_frontier, load_balancing,
                             vertex_pres, vertex_core_number, offsets, nbhr_pointer,  
                             deg, clique_edges, &temp_clique_size, &peel, &batch_size);

        // atomicMax(&max_clique_size, temp_clique_size);
        if (temp_clique_size > max_clique_size) {
            max_clique_size = temp_clique_size;
        }
        std::cout << "Current Max Clique: " << max_clique_size << "\n";

        oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, batch_size, DELETE);
        gpu::memsetZero(hd_data().counter);
        peel--;
    }
    Tclique.stop();
    Tclique.print("Clique Heuristic");
    std::cout << "Max Clique Found: " << max_clique_size << std::endl;
}

void KCore::release() {
    gpu::free(vertex_pres);
    gpu::free(vertex_color);
    gpu::free(vertex_deg);
    gpu::free(vertex_core_number);
    // gpu::free(vertex_clique_number); 
    gpu::free(hd_data().src);
    gpu::free(hd_data().dst);
    hd_data().src = nullptr;
    hd_data().dst = nullptr;
}
}

