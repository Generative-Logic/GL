/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschränkt)
 ... (License Header) ...
*/

#include "run_modes.hpp"
#include "analyze_expressions.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <iostream>

namespace run_modes {

    inline const std::filesystem::path PROJECT_ROOT =
        std::filesystem::path(__FILE__).parent_path()
        .parent_path()
        .parent_path()
        .parent_path();

    inline const std::filesystem::path RAW_PROOF_DIR =
        PROJECT_ROOT / "files" / "raw_proof_graph";

    inline const std::filesystem::path THEOREMS_FOLDER = PROJECT_ROOT / "files" / "theorems";
    inline const std::filesystem::path THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";
    // Define the path for the proved theorems file
    inline const std::filesystem::path PROVED_THEOREMS_FILE = THEOREMS_FOLDER / "proved_theorems.txt";

    static inline std::string trim_copy(const std::string& s) {
        const auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return {};
        const auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    // Helper to read a file into a unique set of non-empty lines
    std::unordered_set<std::string> loadLinesFromFile(const std::filesystem::path& path) {
        std::unordered_set<std::string> lines;
        if (!std::filesystem::exists(path)) return lines;

        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;
            lines.emplace(std::move(line));
        }
        return lines;
    }


    void fullRun(const std::string& anchor_id) {
        using namespace std;
        namespace fs = std::filesystem;

        if (!anchor_id.empty()) {
            std::cout << "[fullRun] anchor_id = " << anchor_id << "\n";
        }
        static gl::ExpressionAnalyzer expressionAnalyzer(anchor_id);

#if 0
        std::string expression1 = "(fold[N,s,+,f,n,m,p])";
        std::set<std::string> remainingArgs = { "N","s","+","f","n","m" };
        gl::BodyOfProves mb;
        mb.level = 0;

        expressionAnalyzer.compileCoreExpressionMap();
        expressionAnalyzer.prepareIntegration(expression1, remainingArgs, mb);
#endif

        // 1) Load already proved theorems
        std::cout << "Loading proved theorems..." << std::endl;
        std::unordered_set<std::string> proved_set = loadLinesFromFile(PROVED_THEOREMS_FILE);
        std::vector<std::string> proved_lst(proved_set.begin(), proved_set.end());

        // Sort to ensure deterministic behavior
        std::sort(proved_lst.begin(), proved_lst.end());

        // Note: broadcastTheorems() is removed. We now pass proved_lst directly to analyzeExpressions.
        std::cout << "Loaded " << proved_lst.size() << " proved theorems." << std::endl;

        // 2) Load theorems to be proved
        if (!fs::exists(THEOREMS_FILE)) {
            std::cerr << "[full_run] Missing theorems file: " << THEOREMS_FILE << "\n";
            return;
        }

        std::unordered_set<std::string> theorem_set = loadLinesFromFile(THEOREMS_FILE);
        std::vector<std::string> tmp_lst(theorem_set.begin(), theorem_set.end());
        std::sort(tmp_lst.begin(), tmp_lst.end());

        // 3) Analyze all theorems (Passing proved theorems as 2nd arg)
        expressionAnalyzer.analyzeExpressions(tmp_lst, proved_lst);

        // 4) Save newly proved theorems at once
        expressionAnalyzer.saveProvedTheorems();

        if (expressionAnalyzer.parameters.debug) {
            std::vector<std::string> expr_lst{
                "(AnchorPeano[1,2,3,4,5,6])",
                "(in3[6,7,8,4])",
                "(in2[rec0,7,3])"
            };
            expressionAnalyzer.findEnds(expr_lst, RAW_PROOF_DIR);
        }
        else {
            expressionAnalyzer.generateRawProofGraph(
                expressionAnalyzer.globalTheoremList, RAW_PROOF_DIR);
        }
    }

} // namespace run_modes