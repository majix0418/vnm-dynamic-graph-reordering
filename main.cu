#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "util/argparse.h" 
#include "mtx/Mtx.h"
#include "reorder/reorder.h"
// #include "kernel/eval.h"
std::vector<float> hybrid_reordering(std::string &mtxfile, std::string &outmtxfile, int schedule, int V, int rounds);
std::vector<float> graph_NM_reordering(std::string &mtxfile, int schedule)
{
    // read original mtx
    Mtx<float> *spmat = new Mtx<float>(mtxfile, M); 

    // reorder
    reorderUtil<float> *rutil = new reorderUtil<float>(spmat, 10);
    std::vector<int> res;

    START_TIMER;

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
    
    STOP_TIMER;

    float improve_rate = (res[2]!=0)?((res[2]-res[3])*1.0f/res[2]):0;
    float density = spmat->nnz*1.0f/(spmat->nrows*spmat->nrows);
    float delta_per_round = ((res[2]-res[3])*1.0f/res[4]);
    float list_per_round = ((res[0]-res[1])*1.0f/res[4]);
    float nrows = static_cast<float>(spmat->nrows);
    float nnz = static_cast<float>(spmat->nnz);

    delete spmat;

    return {nrows, nnz, density, \
            static_cast<float>(res[0]), static_cast<float>(res[1]), \
            static_cast<float>(res[2]), static_cast<float>(res[3]), improve_rate, \
            static_cast<float>(res[4]), milliseconds, delta_per_round, list_per_round};
}

int find_best_2toX(std::string &mtxfile, std::string &outmtxfile, int num_of_scheds, int V=32, int rounds=10)
{
    int max_schedule_id = 0;
    float max_imprv = 0.0f;
    float max_imprv_time = 0.0f;

    // search for the most improve rate that requires least reorder time
    std::vector<int> outall;
    if (M > 4)
        outall = {0, 1, 2, 3};
    else
        outall = {0, 1, 2, 3, 4, 5, 6, 7};
    std::ofstream resultFile("schedule_result_4:2:8.csv", std::ios::app);
    // resultFile << "Filename,Schedule,nrows,nnz,density,init_size_inv_segment,final_size_inv_segment,"
    //            << "init_Pscore,final_Pscore,improve_rate,init_mb_score,final_mb_score,"
    //            << "init_vnm_score,final_vnm_score,nm_round,vnm_round,total_rounds,duration_ms\n";
    std::string filename = mtxfile.substr(mtxfile.find_last_of("/\\") + 1);
    // resultFile << filename << ",";
    for (int i = 0; i < num_of_scheds; i++) {
        if (std::find(outall.begin(), outall.end(), i) == outall.end()) {
            continue;
        }
        std::vector<float> out = hybrid_reordering(mtxfile, outmtxfile, /*NM schedule=*/i, /*V=*/V, /*round=*/rounds);
        
        resultFile << filename << "," << i << ","
                   << out[0] << "," << out[1] << "," << out[2] << "," << out[3] << "," << out[4] << ","
                   << out[5] << "," << out[6] << "," << out[7] << "," << out[8] << "," << out[9] << ","
                   << out[10] << "," << out[11] << "," << out[12] << "," << out[13] << "," << out[14] << ","
                   << out[15] << "\n";
    }

    resultFile.close();

    // for (int i=0; i<num_of_scheds; i++) {
    //     std::vector<float> out = graph_NM_reordering(mtxfile, i);

    //     // print empty for already satisfying ones
    //     if (out[1] == 0 || out[3] == 0) { /*std::cout << std::endl;*/ return 0; } // default 0

    //     // record best
    //     if (out[7] > max_imprv) {

    //         max_schedule_id = i;
    //         max_imprv = out[7];
    //         max_imprv_time = out[9];

    //     } else if (out[7] == max_imprv) {

    //         if (out[9] < max_imprv_time) {
                
    //             max_schedule_id = i;
    //             max_imprv = out[7];
    //             max_imprv_time = out[9];
    //         }
    //     }

    //     outall.push_back(out);

    //     std::vector<float> out = graph_NM_reordering(mtxfile, i);

    //     // print empty for already satisfying ones
    //     if (out[1] == 0 || out[3] == 0) { /*std::cout << std::endl;*/ return 0; } // default 0

    //     // record best
    //     if (out[7] > max_imprv) {

    //         max_schedule_id = i;
    //         max_imprv = out[7];
    //         max_imprv_time = out[9];

    //     } else if (out[7] == max_imprv) {

    //         if (out[9] < max_imprv_time) {
                
    //             max_schedule_id = i;
    //             max_imprv = out[7];
    //             max_imprv_time = out[9];
    //         }
    //     }

    //     outall.push_back(out);
    // }

    // std::cout << outall[max_schedule_id][0] << ", " << outall[max_schedule_id][1] << ", " << outall[max_schedule_id][2] << ", " \
    //             << outall[max_schedule_id][3] << ", " << outall[max_schedule_id][4] << ", " \
    //             << outall[max_schedule_id][5] << ", " << outall[max_schedule_id][6] << ", " << outall[max_schedule_id][7] << ", " \
    //             << outall[max_schedule_id][8] << ", " << outall[max_schedule_id][9] << ", " << outall[max_schedule_id][10] << ", " \
    //             << outall[max_schedule_id][11] << ", " << max_schedule_id << std::endl;
    
    return max_schedule_id;
}

std::vector<int> VNM_reordering(int V=32)
{
    // Read from temp.mtx
    std::string cppString = "python3 reordering.py --m " + std::to_string(M) + " --v " + std::to_string(V);

    const char* command = cppString.c_str();

    // Use the system function to execute the shell command.
    int result = system(command);

    // Open a file for reading (replace "input.txt" with your file's path)
    std::ifstream inputFile("output.txt");

    // Vector to store the substrings
    std::vector<int> res;

    if (inputFile.is_open()) {
        std::string line;

        // Read the file line by line
        while (std::getline(inputFile, line)) {
            // Process each line here
            // std::cout << line << std::endl;

            // Create a stringstream from the input string
            std::istringstream ss(line);

            std::string token;
            while (std::getline(ss, token, ',')) {
                res.push_back(std::stoi(token));
            }
        }
    }

    // Close the file
    inputFile.close();

    return res;
}

std::vector<int> eval_venom(int V=32)
{

    std::string cppString = "python3 reordering.py --evalonly --m " + std::to_string(M) + " --v " + std::to_string(V);

    const char* command = cppString.c_str();

    // Use the system function to execute the shell command.
    int result = system(command);

    // Open a file for reading (replace "input.txt" with your file's path)
    std::ifstream inputFile("output.txt");

    // Vector to store the substrings
    std::vector<int> res;

    if (inputFile.is_open()) {
        std::string line;

        // Read the file line by line
        while (std::getline(inputFile, line)) {
            // Process each line here
            // std::cout << line << std::endl;

            // Create a stringstream from the input string
            std::istringstream ss(line);

            std::string token;
            while (std::getline(ss, token, ',')) {
                res.push_back(std::stoi(token));
            }
        }
    }

    // Close the file
    inputFile.close();

    return res;
}

std::vector<float> hybrid_reordering(std::string &mtxfile, std::string &outmtxfile, int schedule, int V=32, int rounds=10)
{
    // read original mtx
    Mtx<float> *spmat = new Mtx<float>(mtxfile, M); 

    // reorder
    reorderUtil<float> *rutil = new reorderUtil<float>(spmat, rounds);

    // terminate if no nnzs
    if (spmat->nnz == 0) { std::cout << std::endl; return {}; }


    START_TIMER;

    // choose one sched
    std::vector<int> res_first;
    switch (schedule)
    {
        case 0:
            res_first = rutil->run_onepair(false, false, false);
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

    // write to file
    std::string tempfile = "temp.mtx";
    spmat->write_to_file(tempfile);
    // spmat->init_from_file(tempfile, M);
    // spmat = new Mtx<float>(tempfile, M); 
    // a vnm check overhere for early stop
    std::vector<int> venom_res = eval_venom(V);
    int init_mb_score = venom_res[0], init_vnm_score = venom_res[1];

    // write to final dest
    // if (V == 1 || rounds == 0) spmat->write_to_file(outmtxfile);

    // start interleaving rounds
    std::vector<int> res;
    int vnm_round = 0;
    int nm_round = res_first[4];

    // int x; std::cin >> x;

    for (int i=0; i<rounds; i++) {

        // vnm reordering read from & save to "temp.mtx"
        // Not needed when converting to 2:4
        // for (int j=0; j<spmat->nnz; j++) {
        //     std::cout << spmat->coo_rowind_h[j] << " " << spmat->coo_colind_h[j] << std::endl;
        // }
        if (M > 4)
        {
            std::vector<int> vnm_res = VNM_reordering(V);
            vnm_round += vnm_res[4];
        }
        // int x; std::cin >> x;

        // read original mtx
        Mtx<float> *spmat = new Mtx<float>(tempfile, M); 
        // std::string tempfile1 = "temp1.mtx";
        // spmat->write_to_file(tempfile1);
        // reorder
        reorderUtil<float> *rutil = new reorderUtil<float>(spmat, rounds);
        
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
        // std::cout << "Current invalid seg count " << res[3] << std::endl;

        nm_round += res[4];

        // write to file
        // spmat->write_to_file(tempfile);

        // write to final dest

        // // a vnm check overhere for early stop
        std::vector<int> venom_res = eval_venom(V);
        if (venom_res[1] == 0)
            break;
        // x; std::cin >> x;
    }
    STOP_TIMER;

    float improve_rate = (res_first[2]!=0)?((res_first[2]-res[3])*1.0f/res_first[2]):0;
    float density = spmat->nnz*1.0f/(spmat->nrows*spmat->nrows);
    float nrows = static_cast<float>(spmat->nrows);
    float nnz = static_cast<float>(spmat->nnz);

    venom_res = eval_venom(V);
    int final_mb_score = venom_res[0], final_vnm_score = venom_res[1];
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
    int best_sched = find_best_2toX(mtxfile, outmtxfile, 8, v, maxiter);

    // 2. hybrid reordering
    // hybrid_reordering(mtxfile, outmtxfile, /*NM schedule=*/0, /*V=*/v, /*round=*/ maxiter);
    // hybrid_reordering(mtxfile, outmtxfile, /*NM schedule=*/best_sched, /*V=*/v, /*round=*/ maxiter);
}