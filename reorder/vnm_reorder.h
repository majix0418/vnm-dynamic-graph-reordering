#ifndef VNM_REORDERING_H
#define VNM_REORDERING_H

#include "../mtx/Mtx.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <bitset>
#include <algorithm>
#include <unordered_map>


// Helper functions
// Function to calculate Hamming distance between two bit strings
int hamming_distance(const std::string& bit_string1, const std::string& bit_string2) {
    if (bit_string1.length() != bit_string2.length()) {
        throw std::invalid_argument("Bit strings must be of the same length");
    }

    int distance = 0;
    for (size_t i = 0; i < bit_string1.length(); ++i) {
        if (bit_string1[i] != bit_string2[i]) {
            ++distance;
        }
    }
    return distance;
}

// Function to generate Gray codes of length `n`
std::vector<std::string> generate_gray_code(int n) {
    if (n == 0) {
        return {"0"};
    } else if (n == 1) {
        return {"0", "1"};
    }

    std::vector<std::string> gray_code = generate_gray_code(n - 1);
    std::vector<std::string> reflected_gray_code(gray_code.rbegin(), gray_code.rend());

    for (std::string& code : gray_code) {
        code = "0" + code;
    }
    for (std::string& code : reflected_gray_code) {
        code = "1" + code;
    }

    gray_code.insert(gray_code.end(), reflected_gray_code.begin(), reflected_gray_code.end());
    return gray_code;
}

// Function to construct the bitcode hash table
std::unordered_map<std::string, int> construct_bitcode_hash_tab(int nb) {
    std::vector<std::string> gray_code = generate_gray_code(nb);
    std::unordered_map<std::string, int> bithashtab;
    int cnt = 0;

    for (const std::string& code : gray_code) {
        if (std::count(code.begin(), code.end(), '1') <= 2) {
            bithashtab[code] = cnt;
        } else {
            bithashtab[code] = -cnt;
        }
        ++cnt;
    }

    return bithashtab;
}

int count_bits(int value) {
    return std::bitset<32>(value).count();
}

int count_invalid_block_strict(const std::vector<std::vector<int>> &segmat, int V) {
    
    int inv_venom_blk = 0;
    int rows = segmat.size();
    int cols = segmat[0].size();
    // cols = rows / M. each seg is a v*M block. Check 1. if each row has <= 2 elements; 2. if the combined block has <= 4 non-empty cols
    for (int seg = 0; seg < cols; ++seg) {
        for (int i = 0; i < rows; i += V) {
            int bval = 0;
            bool invalid_nm = false;

            for (int j = i; j < i + V && j < rows; ++j) {
                if (count_bits(segmat[j][seg]) > 2) {
                    invalid_nm = true;
                }
                bval |= segmat[j][seg];
            }

            if (count_bits(bval) > 4 || invalid_nm) {
                inv_venom_blk++;
            }
        }
    }

    return inv_venom_blk;
}

int count_invalid_blocks(const std::vector<std::vector<int>> &segmat, int V) {
    int inv_cnt = 0;
    int rows = segmat.size();
    int cols = segmat[0].size();

    for (int seg = 0; seg < cols; ++seg) {
        for (int i = 0; i < rows; i += V) {
            int bval = 0;
            for (int j = i; j < i + V && j < rows; ++j) {
                bval |= segmat[j][seg];
            }
            if (count_bits(bval) > 4) inv_cnt++;
        }
    }
    return inv_cnt;
}

std::vector<std::pair<int, std::vector<int>>> hash_segmat(const std::vector<std::vector<int>> &segmat, const std::unordered_map<std::string, int> &bithashtab) {
    std::vector<std::pair<int, std::vector<int>>> hashed_segmat;
    int m = segmat.size();
    int n = segmat[0].size() * M;

    for (int i = 0; i < m; ++i) {
        std::vector<int> tmp;
        for (int j = 0; j < n/M; ++j) {
            std::string bit_string = std::bitset<32>(segmat[i][j]).to_string().substr(32 - M);
            // std::cout << "Bit string: " << bit_string << std::endl;
            tmp.push_back(bithashtab.at(bit_string));
        }
        hashed_segmat.emplace_back(i, tmp);
    }

    return hashed_segmat;
}

std::vector<int> vnm_reordering(Mtx<float>* spmat, int V, int MAXITER, bool evalonly=false) {

    int m, n, nnz;
    m = spmat->nrows;
    n = spmat->ncols;
    nnz = spmat->nnz;

    std::vector<int> row_indices = spmat->coo_rowind_h;
    std::vector<int> col_indices = spmat->coo_colind_h;
    std::vector<float> values(nnz, 1.0);
    // std::vector<float> values = spmat->coo_values_h;
    int padded_m = ((m + M - 1) / M) * M;
    int padded_n = ((n + M - 1) / M) * M;

    // Generate segments
    std::vector<std::vector<int>> segmat(padded_m, std::vector<int>(padded_n / M, 0));
    for (int i = 0; i < nnz; ++i) {
        segmat[row_indices[i]][col_indices[i] / M] |= (1 << (M - 1 - col_indices[i] % M));
        // std::cout << "(" << row_indices[i] << "," << col_indices[i] << ")" << std::bitset<32>(segmat[row_indices[i]][col_indices[i] / M]).to_string().substr(32 - M) << " ";
    }
    int init_inv_cnt = count_invalid_blocks(segmat, V);
    int init_inv_cnt_strict = count_invalid_block_strict(segmat, V);
    int inv_cnt = init_inv_cnt_strict;

    if (evalonly) {
        return {init_inv_cnt, init_inv_cnt_strict};
    }
    // Mtx<float> *spmat_copy = new Mtx<float>(padded_m, padded_n, nnz, row_indices, col_indices, values);

    std::unordered_map<std::string, int> bithashtab = construct_bitcode_hash_tab(M);
    std::vector<int> res = {init_inv_cnt, init_inv_cnt_strict, init_inv_cnt, init_inv_cnt_strict, 0};

    // Reordering logic
    int round = 0;
    
    while (init_inv_cnt_strict > 0 && round < MAXITER) {
        // for (int j=0; j<spmat->nnz; j++) {
        //     std::cout << spmat->coo_rowind_h[j] << " " << spmat->coo_colind_h[j] << std::endl;
        // }
        // std::cout << "Round: " << round << ", Invalid Blocks: " << inv_cnt << std::endl;
        // for (int j = 0; j < padded_m; ++j) {
        //     for (int k = 0; k < padded_n / M; ++k) {
        //         std::cout << segmat[j][k] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        std::vector<std::pair<int, std::vector<int>>> hashed_segmat = hash_segmat(segmat, bithashtab);
        // for (int i = 0; i < hashed_segmat.size(); ++i) {
        //     std::cout << hashed_segmat[i].first << " ";
        //     for (int j = 0; j < hashed_segmat[i].second.size(); ++j) {
        //         // std::cout << std::bitset<32>(hashed_segmat[i].second[j]).to_string().substr(32 - M) << " ";
        //         std::cout <<  hashed_segmat[i].second[j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        std::sort(hashed_segmat.begin(), hashed_segmat.end(), [](const auto &a, const auto &b) {
            return a.second < b.second;
        });

        std::vector<int> new_id(padded_m);
        for (int i = 0; i < padded_m; ++i) {
            new_id[i] = hashed_segmat[i].first;
        }
        

        std::vector<int> new_row_indices(row_indices.size());
        std::vector<int> new_col_indices(col_indices.size());
        for (size_t i = 0; i < row_indices.size(); ++i) {
            new_row_indices[i] = new_id[row_indices[i]];
            new_col_indices[i] = new_id[col_indices[i]];
        }

        std::fill(segmat.begin(), segmat.end(), std::vector<int>(padded_n / M, 0));
        for (int i = 0; i < nnz; ++i) {
            segmat[new_row_indices[i]][new_col_indices[i] / M] |= (1 << (M - 1 - new_col_indices[i] % M));
        }
        // for (int i = 0; i < segmat.size(); ++i) {
        //     for (int j = 0; j < segmat[i].size(); ++j) {
        //         std::cout << std::bitset<32>(segmat[i][j]).to_string().substr(32 - M) << " ";
        //         // std::cout <<  hashed_segmat[i].second[j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        int new_inv_cnt = count_invalid_blocks(segmat, V);
        int new_inv_cnt_strict = count_invalid_block_strict(segmat, V);
        if (new_inv_cnt_strict <= inv_cnt) {
            // std::cout << padded_m << " " << padded_n << " " << nnz << " " << new_row_indices.size() << " " << new_col_indices.size() << std::endl;
            spmat->init_from_coo(padded_m, padded_n, nnz, new_row_indices, new_col_indices, values);
            // std::cout << spmat->nnz << " " << spmat->nrows << std::endl;
            res = {init_inv_cnt, init_inv_cnt_strict, new_inv_cnt, new_inv_cnt_strict, ++round};
        }
        else {
            break;
        }
        row_indices = new_row_indices;
        col_indices = new_col_indices;
        inv_cnt = new_inv_cnt_strict;
    }
    return res;
}


#endif 