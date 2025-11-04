#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "util/argparse.h" 
#include "mtx/Mtx.h"
#include "reorder/reorder.h"
#include "reorder/vnm_reorder.h"
std::vector<float> hybrid_reordering(std::string &mtxfile, std::string &outmtxfile, int schedule, int num_of_scheds, int V, int rounds);

int run_cyclic_schedule(std::string &mtxfile, std::string &outmtxfile, int num_of_scheds, int V=32, int rounds=10)
{
    int max_schedule_id = 0;
    float max_imprv = 0.0f;
    float max_imprv_time = 0.0f;

    // search for the most improve rate that requires least reorder time
    std::vector<std::vector<float>> outall;
    int base_schedule = 0;
    std::string res_fname = "./schedule_results/schedule_result_" + std::to_string(V) + ":" + std::to_string(N) + ":" + std::to_string(M) + ".csv";
    std::ofstream resultFile(res_fname, std::ios::app);
    std::string filename = mtxfile.substr(mtxfile.find_last_of("/\\") + 1);

    std::vector<float> out = hybrid_reordering(mtxfile, outmtxfile, /*NM schedule=*/base_schedule, num_of_scheds, /*V=*/V, /*round=*/rounds);

    resultFile << filename << "," << base_schedule << ","
                << out[0] << "," << out[1] << "," << out[2] << "," << out[3] << "," << out[4] << ","
                << out[5] << "," << out[6] << "," << out[7] << "," << out[8] << "," << out[9] << ","
                << out[10] << "," << out[11] << "," << out[12] << "," << out[13] << "," << out[14] << ","
                << out[15] << "\n";

    // Above for new strategy only

    return 0;
}

std::vector<int> VNM_reordering(Mtx<float> *spmat, int V, int rounds)
{
    std::vector<int> res = vnm_reordering(spmat, V, rounds);
    return res;
}

std::vector<int> eval_venom(Mtx<float> *spmat, int V, int rounds)
{

    std::vector<int> res = vnm_reordering(spmat, V, rounds, true);

    return res;
}

std::vector<float> hybrid_reordering(std::string &mtxfile, std::string &outmtxfile, int schedule_prime, int num_of_scheds, int V=32, int rounds=10)
{
    // read original mtx
    Mtx<float> *spmat = new Mtx<float>(mtxfile, M); 

    // reorder
    reorderUtil<float> *rutil = new reorderUtil<float>(spmat, 10);

    // terminate if no nnzs
    if (spmat->nnz == 0) { std::cout << std::endl; return {}; }

    std::vector<int> res_first;
    int schedule = schedule_prime;
     // a vnm check overhere for early stop
    std::vector<int> venom_res = eval_venom(spmat, V, rounds);
    int init_mb_score = venom_res[0], init_vnm_score = venom_res[1];
    START_TIMER;
    
    switch (schedule)
    {
        case 0:
            res_first = rutil->run_onepair(false, false, true);
            break;
        case 1:
            res_first = rutil->run_onepair_double(false, false);
            break;
        case 2:
            res_first = rutil->run_onepair_reverse(false, false);
            break;
        case 3:
            res_first = rutil->run_onepair_reverse_double(false, false);
            break;
        case 4:
            res_first = rutil->run_allpairs(false, false);
            break;
        case 5:
            res_first = rutil->run_allpairs_double(false, false);
            break;
        case 6:
            res_first = rutil->run_allpairs_reverse(false, false);
            break;
        case 7:
            res_first = rutil->run_allpairs_reverse_double(false, false);
            break;
        default:
            res_first = rutil->run(false, false);
            break;
    }

    // start interleaving rounds
    std::vector<int> res;
    int vnm_round = 0;
    int nm_round = res_first[4];
    // int nm_round = 0;
    int prev_score = init_vnm_score;

    // int x; std::cin >> x;
    for (int i=0; i<rounds; i++) {
        // // a vnm check overhere for early stop
        std::vector<int> vnm_res = VNM_reordering(spmat, V, rounds);
        vnm_round += vnm_res[4];

        // reorder
        reorderUtil<float> *rutil = new reorderUtil<float>(spmat, 10);
        
        // choose one sched
        switch (schedule)
        {
            case 0:
                res = rutil->run_onepair(false, false, false);
                break;
            case 1:
                res = rutil->run_onepair_double(false, false);
                break;
            case 2:
                res = rutil->run_onepair_reverse(false, false);
                break;
            case 3:
                res = rutil->run_onepair_reverse_double(false, false);
                break;
            case 4:
                res = rutil->run_allpairs(false, false);
                break;
            case 5:
                res = rutil->run_allpairs_double(false, false);
                break;
            case 6:
                res = rutil->run_allpairs_reverse(false, false);
                break;
            case 7:
                res = rutil->run_allpairs_reverse_double(false, false);
                break;
            default:
                res = rutil->run(false, false);
                break;
        }
        // std::cout << "Round " << i << " nnz(reorder)" << spmat->nnz << std::endl;
        
        // // std::cout << "Current invalid seg count " << res[3] << std::endl;

        nm_round += res[4];
        if (vnm_res[1] == 0)  // if already satisfied,
            break;

        // if not improved, switch to second schedule
        if (vnm_res[1] >= prev_score) {
            schedule = (schedule + 1) % num_of_scheds;
        }
        prev_score = vnm_res[1];

        // write to file
        // write to final dest
        delete rutil;
    }
    STOP_TIMER;

    float improve_rate = (res_first[2]!=0)?((res_first[2]-res[3])*1.0f/res_first[2]):0;
    float density = spmat->nnz*1.0f/(spmat->nrows*spmat->nrows);
    float nrows = static_cast<float>(spmat->nrows);
    float nnz = static_cast<float>(spmat->nnz);

    venom_res = eval_venom(spmat, V, rounds);
    int final_mb_score = venom_res[0], final_vnm_score = venom_res[1];
    // std::cout << " nnz final" << spmat->nnz << std::endl;
    if (final_vnm_score == 0)
        spmat->write_to_file(outmtxfile);

    delete spmat;
    // init_size, final_size, init_cnt, final_cnt, round-1
    std::cout << "nrows: " << nrows << ", "
          << "nnz: " << nnz << ", "
          << "density: " << density << ", "
          << "init_size_inv_segment: " << static_cast<float>(res_first[0]) << ", "
          << "final_size_inv_segment: " << static_cast<float>(res[1]) << ", "
          << "init_Pscore: " << static_cast<float>(res_first[2]) << ", "
          << "final_Pscore: " << static_cast<float>(res[3]) << ", "
          << "improve_rate: " << improve_rate << ", "
          << "init_mb_score: " << init_mb_score << ", "
          << "final_mb_score: " << final_mb_score << ", "
          << "init_vnm_score: " << init_vnm_score << ", "
          << "final_vnm_score: " << final_vnm_score << ", "
          << "nm_round: " << nm_round << ", "
          << "vnm_round: " << vnm_round << ", "
          << "total_rounds: " << (nm_round + vnm_round) << ", "
          << "duration_ms: " << milliseconds 
          << std::endl;

    return {static_cast<float>(nrows), static_cast<float>(nnz), static_cast<float>(density), 
            static_cast<float>(res_first[0]), static_cast<float>(res[1]), 
            static_cast<float>(res_first[2]), static_cast<float>(res[3]), 
            static_cast<float>(improve_rate), 
            static_cast<float>(init_mb_score), static_cast<float>(final_mb_score), 
            static_cast<float>(init_vnm_score), static_cast<float>(final_vnm_score), 
            static_cast<float>(nm_round), static_cast<float>(vnm_round), 
            static_cast<float>(nm_round + vnm_round), static_cast<float>(milliseconds)};

    
}

int main(int argc, const char **argv)
{
    // parse arg
    std::string mtxfile, outmtxfile;
    int maxiter, n, sched, v;
    parseArgs(argc, argv, mtxfile, outmtxfile, maxiter, n, sched, v, false);

    // 1. find the best 2:X traversal schedule for mtx
    // Note: for 2:4, we may test up to 8 scheds; for others, test 4 scheds
    int max_sched = (M == 4) ? 8 : 4;
    int best_sched = run_cyclic_schedule(mtxfile, outmtxfile, max_sched, v, maxiter);
}