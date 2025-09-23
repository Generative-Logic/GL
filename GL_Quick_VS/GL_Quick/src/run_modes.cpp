/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschränkt)

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
        std::filesystem::path(__FILE__).parent_path()  // .../src
        .parent_path()                              // .../GL_Quick
        .parent_path()                              // .../GL_Quick_VS
        .parent_path();                             // repo root
    gl::ExpressionAnalyzer expressionAnalyzer;


    inline const std::filesystem::path RAW_PROOF_DIR =
        PROJECT_ROOT / "files" / "raw_proof_graph";

void quickRun() {
    // A subset of the expressions used by Python quick mode, captured as raw strings.
    // These are used to seed the scaffold analyzer and to demonstrate output shape.
    std::vector<std::string> expressionList{
    "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(in3[8,7,9,5])))",
    "(>[3,4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in2[7,8,4])(in3[7,3,8,5])))",
    "(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,6])(>[12](in3[8,10,12,6])(>[13](in3[9,10,13,6])(in3[11,12,13,5]))))))",
    "(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,6])(>[12,13](in3[7,12,13,6])(>[](in3[8,10,12,5])(in3[9,11,13,5]))))))",
    "(>[4,5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in2[10,11,4])(>[](in3[10,8,7,6])(in3[11,8,9,6])))))",
    "(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(in3[8,7,9,6])))",
    "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12](in3[9,10,12,5])(in3[11,8,12,5])))))",
    "(>[6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,6])(>[10,11](in3[7,10,11,6])(>[12](in3[8,12,10,6])(in3[9,12,11,6])))))",
    "(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[7,10,4])(>[11](in2[9,11,4])(in3[10,8,11,5])))))",
    "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12](in3[8,12,10,5])(in3[9,12,11,5])))))",
    "(>[2,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in3[2,7,8,5])(in3[7,2,8,5])))",
    "(>[2,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in3[2,7,8,6])(in3[7,2,8,6])))",
    "(>[3,4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in2[7,8,4])(in3[3,7,8,5])))"
    };

#if 0
    gl::ExpressionAnalyzer *pointer = &expressionAnalyzer;
	std::string input = "(>[b](in[b,1])(>[a,c,d](&(in2[b,c,4])(in3[a,b,d,5]))(>[e](in3[a,c,e,5])(in2[d,e,4]))))";
    std::string smooth = expressionAnalyzer.smoothenExpr(input);

    std::string testInput = "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12](in3[9,10,12,5])(in3[11,8,12,5])))))";
    std::pair<ce::TreeNode1*, int> pr = ce::parseExpr(testInput, expressionAnalyzer.coreExpressionMap);

    testInput = "(&(&(in[2,1])(&(in2[2,3,4])(&(fXY[4,1,1])(&(>[n](in[n,1])!(in2[n,2,4]))(>[m](in[m,1])(>[n1,n2](&(in2[n1,m,4])(in2[n2,m,4]))(=[n1,n2])))))))(&(&(fXYZ[5,1,1,1])(&(>[a](in[a,1])(>[b](in3[a,2,b,5])(=[a,b])))(>[b](in[b,1])(>[a,c,d](&(in2[b,c,4])(in3[a,b,d,5]))(&(>[e](in3[a,c,e,5])(in2[d,e,4]))(>[e](in2[d,e,4])(in3[a,c,e,5])))))))(&(fXYZ[6,1,1,1])(&(>[a](in[a,1])(>[b](in3[a,2,b,6])(=[b,2])))(>[b](in[b,1])(>[a,c,d](&(in2[b,c,4])(in3[a,b,d,6]))(&(>[e](in3[d,a,e,5])(in3[a,c,e,6]))(>[e](in3[a,c,e,6])(in3[d,a,e,5])))))))))";
    std::vector<std::string> testOutput = expressionAnalyzer.groomExpr(testInput);

    testInput = "!(>[y](in[y,1])!(in2[2,y,4]))";
	testOutput = expressionAnalyzer.listLastRemovedArgs(testInput);

    testInput = "!(>[y](in[y,1])!(in2[2,y,4]))";
    std::string newExpr;
    int newStartInt = 0;
    std::string newVar;
	std::tie(newExpr, newStartInt, newVar) = expressionAnalyzer.renameLastRemoved(testInput, 0, 0, 0);

    testInput = "(NaturalNumbers[1,2,3,4,5,6])";
	std::string testOutputString = expressionAnalyzer.expandExpr(testInput);

	std::vector<gl::EncodedExpression> encodedExpressions;
	encodedExpressions.emplace_back(gl::EncodedExpression("(in2[2,3,4])"));

    std::set<std::string> allArgs = expressionAnalyzer.getAllEncodedArgs(encodedExpressions);

    std::vector<gl::EncodedExpression> copy = encodedExpressions;
	expressionAnalyzer.setUnchangeables(copy, std::set<std::string>{"2"});

    std::set<std::string> unchangeables =  expressionAnalyzer.extractUnchangeables(copy);

    //expressionAnalyzer.resetUnchangeables(copy, std::set<std::string>());

    gl::NormalizedKey ky;
    std::map<std::string, std::string> coveredVariables;

    std::tie(ky, coveredVariables) = expressionAnalyzer.makeNormalizedEncodedKey(copy, false);

    std::vector<std::string> key{ "(in3[u_rec,u_8,9,u_5])", "(in3[u_rec,u_10,11,u_6])", "(in3[u_8,u_10,u_12,u_6])", "(in3[9,u_10,13,u_6])" };
    std::string value = "(in3[11,u_12,13,u_5])";
    std::string reconstructedImplication = expressionAnalyzer.reconstructImplication(key, value);

    std::vector<int> binaryList4{ 1, 1, 1, 1 };
    std::vector<std::string> extrExpressions;
    int numArgs;
    std::tie(extrExpressions, numArgs) = expressionAnalyzer.calcNumDiffArgs(key, binaryList4);

    std::vector<std::string> strList{"(in3[u_7,u_8,u_9,u_5])", "(in3[u_8,u_rec,12,u_6])", "(in3[u_9,u_rec,13,u_6])"};
	std::set<std::string> remArgs = expressionAnalyzer.getRemainingArgs(strList);

    bool isQualified = expressionAnalyzer.implicationIsQualified(strList, "(in3[11,12,13,u_5])");

    std::vector<std::string> chain{ "(in[m,u_1])", "(in2[n1,m,u_4])", "(in2[n2,m,u_4])" };
    std::string head = "(=[n1,n2])";
	std::set<gl::VariantItem> variants = expressionAnalyzer.createVariants(chain, head);

	gl::LocalMemory localMemory;
    int numberPatternOccurrences = expressionAnalyzer.countPatternOccurrences("(in3[7,3,it_0_lev_1_0,5])", localMemory);

    numberPatternOccurrences = expressionAnalyzer.countPatternOccurrencesEncoded(gl::EncodedExpression("(in3[7,3,it_0_lev_1_0,5])"), localMemory);

    std::set<std::string> implications;
    std::set<std::string> statements;
    int startInt = 0;
    testInput = "(&(&(in[2,1])(&(in2[2,3,4])(&(fXY[4,1,1])(&(>[n](in[n,1])!(in2[n,2,4]))(>[m](in[m,1])(>[n1,n2](&(in2[n1,m,4])(in2[n2,m,4]))(=[n1,n2])))))))(&(&(fXYZ[5,1,1,1])(&(>[a](in[a,1])(>[b](in3[a,2,b,5])(=[a,b])))(>[b](in[b,1])(>[a,c,d](&(in2[b,c,4])(in3[a,b,d,5]))(&(>[e](in3[a,c,e,5])(in2[d,e,4]))(>[e](in2[d,e,4])(in3[a,c,e,5])))))))(&(fXYZ[6,1,1,1])(&(>[a](in[a,1])(>[b](in3[a,2,b,6])(=[b,2])))(>[b](in[b,1])(>[a,c,d](&(in2[b,c,4])(in3[a,b,d,6]))(&(>[e](in3[d,a,e,5])(in3[a,c,e,6]))(>[e](in3[a,c,e,6])(in3[d,a,e,5])))))))))";
    std::tie(implications, statements, startInt) = expressionAnalyzer.disintegrateExpr(testInput, startInt, 0, 0, localMemory);

    std::string exprForDesintegration = "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(in3[8,7,9,5])))";
    std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > chain2;
    std::map<std::string, ce::CoreExpressionConfig> coreExpressionMap = ce::modifyCoreExpressionMap();
    head = ce::disintegrateImplication(exprForDesintegration, chain2, coreExpressionMap);

	ce::AnchorInfo anchorInfo = ce::initAnchor(coreExpressionMap);
    std::set<std::string> digitArgs = ce::findDigitArgs("(>[3,4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8](in2[7,8,4])(in3[7,3,8,5])))", anchorInfo, coreExpressionMap);

	std::string reshuffledMirrored = ce::createReshuffledMirrored("(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[7,10,4])(>[11](in2[9,11,4])(in3[10,8,11,5])))))", "NaturalNumbers", true, coreExpressionMap);

    std::set<std::string> difference = ce::extractDifference("(>[b](in[b,1])(>[a,c,d](in2[b,c,4])(>[](in3[a,b,d,6])(>[e](in3[a,c,e,6])))))");

    std::tuple<
        std::string,
        std::vector<std::string>,
        std::string,
        std::set<std::string>,
        std::string,
        std::string
    > auxy_imp = expressionAnalyzer.createAuxyImplication("(>[5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,6])(>[12](in3[8,10,12,6])(>[13](in3[9,10,13,6])(in3[11,12,13,5]))))))",
        "10",
        "rec",
        std::set<std::string>{"7", "8", "9"},
        anchorInfo.name);

    int test = 0;
#endif //#if 0

    expressionAnalyzer.analyzeExpressions(expressionList);

    if (parameters::debug) {
        std::vector<std::string> expr_lst{
            "(NaturalNumbers[1,2,3,4,5,6])",
            "(in3[7,8,9,5])",
            "(in2[rec0,7,4])"
        };
        expressionAnalyzer.findEnds(expr_lst);
    }

    expressionAnalyzer.generateRawProofGraph(
        expressionAnalyzer.globalTheoremList, RAW_PROOF_DIR);

}



inline const std::filesystem::path THEOREMS_FOLDER = PROJECT_ROOT / "files" / "theorems";
inline const std::filesystem::path THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";

static inline std::string trim_copy(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

void fullRun() {
    using namespace std;
    namespace fs = std::filesystem;

    // 1) Read theorems file -> set (dedup), then to sorted vector
    if (!fs::exists(THEOREMS_FILE)) {
        std::cerr << "[full_run] Missing theorems file: " << THEOREMS_FILE << "\n";
        return;
    }

    std::unordered_set<std::string> theorem_set;
    {
        std::ifstream in(THEOREMS_FILE);
        std::string line;
        while (std::getline(in, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;
            theorem_set.emplace(std::move(line));
        }
    }

	//cout << theorem_set.size() << " unique theorems read from " << THEOREMS_FILE << "\n";

    std::vector<std::string> tmp_lst(theorem_set.begin(), theorem_set.end());
    std::sort(tmp_lst.begin(), tmp_lst.end());

    // 2) Analyze all theorems (C++ version of analyze_expressions.analyze_expressions)
    expressionAnalyzer.analyzeExpressions(tmp_lst);

    if (parameters::debug) {
        std::vector<std::string> expr_lst{
            "(NaturalNumbers[1,2,3,4,5,6])",
            "(in3[7,8,9,5])",
            "(in2[rec0,7,4])"
        };
        expressionAnalyzer.findEnds(expr_lst);
    }

    expressionAnalyzer.generateRawProofGraph(
        expressionAnalyzer.globalTheoremList, RAW_PROOF_DIR);

}

} // namespace run_modes
