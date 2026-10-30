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
#include "structural_compiler.hpp"
#include "integration_rule_compiler.hpp"
#include "argument_analyzer.hpp"
#include "or_rule_compiler.hpp"
#include "existence_rule_compiler.hpp"
#include "supporting_rule_compiler.hpp"
#include "compilation_reuse.hpp"
#include "definition_expansion.hpp"
#include "expression_parser.hpp"
#include "../compiler.hpp"
#include <numeric>
#include <fstream>

namespace gl {
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param compilationReuse Borrowed compilationReuse component.
/// @param definitionExpansion Borrowed definitionExpansion component.
/// @param orRuleCompiler Borrowed orRuleCompiler component.
/// @param existenceRuleCompiler Borrowed existenceRuleCompiler component.
/// @param supportingRuleCompiler Borrowed supportingRuleCompiler component.
/// @param integrationRuleCompiler Borrowed integrationRuleCompiler component (the integration hooks).
/// @return Constructed compilation component.
StructuralCompiler::StructuralCompiler(CompiledDefinitionRegistry& definitions, CompilationReuse& compilationReuse, DefinitionExpansion& definitionExpansion, OrRuleCompiler& orRuleCompiler, ExistenceRuleCompiler& existenceRuleCompiler, SupportingRuleCompiler& supportingRuleCompiler, IntegrationRuleCompiler& integrationRuleCompiler)
    : definitions(definitions), compilationReuse(compilationReuse), definitionExpansion(definitionExpansion), orRuleCompiler(orRuleCompiler), existenceRuleCompiler(existenceRuleCompiler), supportingRuleCompiler(supportingRuleCompiler), integrationRuleCompiler(integrationRuleCompiler) {}
/// @brief Compile one source expression to its retained compact definition.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param inputExpression Source expression to compile.
/// @return The result of the documented operation.
std::string StructuralCompiler::compileExpression(const std::string& inputExpression) {
            std::string returnExpression;
            std::string expression = inputExpression;
            std::string expandedExpression = expression;

            bool isSimple = ce::expressionIsSimple(expression);

            // Early exit: if expression is simple and already compiled, do not re-compile.
            // This prevents runtime calls (e.g. from reformulateTheorem) from
            // re-expanding already-compiled definitions and creating duplicate
            // implications with new names.
            if (isSimple)
            {
                std::string coreExpression = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();
                if (definitions.compiledExpressions.find(coreExpression) != definitions.compiledExpressions.end())
                {
                    return expression;
                }
            }

            if (isSimple)
            {
                std::vector<std::string> args = ce::getArgs(expression);
                std::set<std::string> uniqueArgs = std::set<std::string>(args.begin(), args.end());
                if (uniqueArgs.size() < definitions.coreExpressionMap[extractExpressionUniversalSpan(StrSpan(expression)).toStdString()].arity)
                {
                    std::string coreExpression = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();

                    expression = definitions.coreExpressionMap[coreExpression].signature;
                }
            }

            expandedExpression = definitionExpansion.expandExpr(expression);

            if (isSimple and expandedExpression == expression)
            {
				std::string coreExpression = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();

                CompiledExpressionMap::const_iterator itLev =
                definitions.compiledExpressions.find(coreExpression);
                if (itLev == definitions.compiledExpressions.end())
                {
					int numArgs = ce::getArgs(expression).size();
                    std::string signature = "(" + coreExpression + "[";
                    for (int index = 1; index <= numArgs; index++)
                    {
                        signature += "u_" + std::to_string(index) + ",";
                    }
                    signature.pop_back(); // remove last comma
                    signature += "])";

                    LogicalEntity le("atomic", std::vector<std::string>(1, ""), signature, numArgs);
					definitions.compiledExpressions.insert(std::make_pair(coreExpression, le));
                }

                return expression;
            }

            if (isSimple && ce::expressionIsSimple(expandedExpression))
            {
                std::string coreExpression = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();

                CompiledExpressionMap::const_iterator itLev =
                    definitions.compiledExpressions.find(coreExpression);
                if (itLev == definitions.compiledExpressions.end())
                {
                    // Define replacements: map args to u_1, u_2, ...
                    std::map<std::string, std::string> replacementMap;
                    std::vector<std::string> args = ce::getArgs(expression);
                    for (size_t i = 0; i < args.size(); ++i) {
                        replacementMap[args[i]] = "u_" + std::to_string(i + 1);
                    }

                    LogicalEntity le;
                    le.category = "and";
                    le.arity = static_cast<int>(args.size());

                    // Apply u_ renamings to signature
                    le.signature = ce::replaceKeysInString(expression, replacementMap);

                    // Apply u_ renamings to the expanded expression and save as element
                    std::string processedElem = ce::replaceKeysInString(expandedExpression, replacementMap);
                    le.elements.push_back(processedElem);

                    definitions.compiledExpressions.insert(std::make_pair(coreExpression, le));
                    // Integration hook: the alias's integration rules ride
                    // its entry.
                    integrationRuleCompiler.compileIntegrationImplications(coreExpression);
                }

                return expression;
            }

            // --- OR detection: nested !(&...!(&...)) pattern ---
            if (isSimple && startsWithSpan(StrSpan(expandedExpression), "!(&", 3)) {
                std::vector<std::string> disjuncts;
                std::string current = expandedExpression;
                bool isOr = true;

                while (startsWithSpan(StrSpan(current), "!(&", 3)) {
                    auto lr = gl::DefinitionExpansion::extractValues(current, definitions.coreExpressionMap);
                    std::string left = lr.first;
                    std::string right = lr.second;

                    if (right.empty() || right[0] != '!') {
                        isOr = false;
                        break;
                    }
                    disjuncts.push_back(right.substr(1)); // un-negate
                    current = left;
                }

                if (isOr && !current.empty() && current[0] == '!') {
                    disjuncts.push_back(current.substr(1)); // last leaf
                    std::reverse(disjuncts.begin(), disjuncts.end());

                    // Safety: no nested OR
                    for (const auto& d : disjuncts) {
                        assert(!startsWithSpan(StrSpan(d), "!(&", 3) && "Nested OR not supported");
                    }

                    // Compile each disjunct recursively
                    std::vector<std::string> compiledDisjuncts;
                    for (const auto& d : disjuncts) {
                        compiledDisjuncts.push_back(
                            compileExpression(d));
                    }

                    // Build encoded chain
                    std::vector<EncodedExpression> encodedDisjuncts;
                    for (const auto& d : compiledDisjuncts) {
                        encodedDisjuncts.push_back(EncodedExpression(d, "main"));
                    }

                    // Mark all arguments as bound
                    for (EncodedExpression& ee : encodedDisjuncts) {
                        for (std::vector<std::string>& argumentDescription : ee.arguments) {
                            argumentDescription[0] = "True";
                        }
                    }

                    // Register as OR category
                    std::string orName = compilationReuse.excludeRepetitions(encodedDisjuncts, {}, "", "or", true, expression);
                    assert(orName == expression && "OR name should match expression");
                    // Or mint hook: the K-rule compacts ride the or's entry
                    // (D-309); the subset-exclusion
                    // family follows at the preMintReducedOrs seam.
                    orRuleCompiler.compileOrKRules(extractExpressionUniversalSpan(StrSpan(orName)).toStdString());
                    return orName;
                }
            }

            // --- Non-simple !(&...) handler: raw structural OR from theorems.txt ---
            // These expressions are NOT simple (not (name[args])), so the isSimple OR handler
            // above doesn't catch them. Handle before groomExpr/smoothenExpr which can't parse !(&...).
            if (!isSimple && startsWithSpan(StrSpan(expression), "!(&", 3)) {
                // !(&left right): the conjuncts are the negated disjuncts. A
                // conjunction has two children, so an or of three or more
                // disjuncts nests, (&(&!A!B)!C) or (&!A(&!B!C)); a positive
                // `&` child is associativity and is collected flat, a negated
                // `&` child is one conjunct. Leaf order is the disjunct
                // order, so a reloaded or with all its elements dedups onto
                // the registered entity (I-23).
                auto lr = gl::DefinitionExpansion::extractValues(expression, definitions.coreExpressionMap);
                std::vector<std::string> conjuncts;
                gl::DefinitionExpansion::collectConjuncts(lr.first, definitions.coreExpressionMap, conjuncts);
                gl::DefinitionExpansion::collectConjuncts(lr.second, definitions.coreExpressionMap, conjuncts);

                // Un-negate to get positive disjuncts
                auto unneg = [](const std::string& s) -> std::string {
                    return (!s.empty() && s[0] == '!') ? s.substr(1) : ("!" + s);
                };

                // Compile the positive disjuncts in disjunct order
                std::vector<EncodedExpression> encodedChain;
                encodedChain.reserve(conjuncts.size());
                for (const std::string& conjunct : conjuncts) {
                    encodedChain.push_back(EncodedExpression(compileExpression(unneg(conjunct)), "main"));
                }

                // Mark all arguments as bound
                for (EncodedExpression& ee : encodedChain) {
                    for (std::vector<std::string>& argumentDescription : ee.arguments) {
                        argumentDescription[0] = "True";
                    }
                }

                std::string coreExprOr = "or" + std::to_string(definitions.orCounter);
                definitions.orCounter++;
                std::string orName = compilationReuse.excludeRepetitions(encodedChain, {}, coreExprOr, "or", false, "");
                std::string tempName = extractExpressionUniversalSpan(StrSpan(orName)).toStdString();
                if (coreExprOr != tempName) { definitions.orCounter--; }
                // Or mint hook (idempotent on a registry reuse): the K-rule
                // compacts ride the or's entry (D-309).
                orRuleCompiler.compileOrKRules(tempName);
                return orName;
            }

            std::string coreExpr = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();

            std::vector<std::string> overallChain;

            // 1) groom
            const std::vector<std::string> subexprsAfterGrooming = gl::ExpressionParser::groomExpr(expandedExpression);

            // 2) smoothen
            std::vector<std::string> smoothenedSubexprsAfterGrooming;
            smoothenedSubexprsAfterGrooming.reserve(subexprsAfterGrooming.size());
            for (std::size_t i = 0; i < subexprsAfterGrooming.size(); ++i)
            {
                const std::string sm = gl::ExpressionParser::smoothenExpr(subexprsAfterGrooming[i]);
                smoothenedSubexprsAfterGrooming.push_back(sm);
            }

            // 3) iterate
            for (std::size_t i = 0; i < smoothenedSubexprsAfterGrooming.size(); ++i)
            {
                const std::string& subexpr = smoothenedSubexprsAfterGrooming[i];

                if (startsWithSpan(StrSpan(subexpr), "(>", 2))
                {
                    std::string implName;
                    LogicalEntity le;

                    ce::TreeNode1* root = ce::parseExpr(subexpr);

                    std::vector< std::tuple<
                        std::string,                    // left expr
                        std::vector<std::string>,       // args of current '>' node
                        std::set<std::string>           // left node's arguments (set)
                    > > tempChain;

                    std::string head =
                        ce::disintegrateImplication(subexpr, tempChain, definitions.coreExpressionMap);

                    std::string simpleHead = compileExpression(head);

                    std::vector<std::string> chain;
                    std::vector<std::string> simpleChain;
                    chain.reserve(tempChain.size());
                    for (std::size_t i = 0; i < tempChain.size(); ++i) {
                        chain.push_back(std::get<0>(tempChain[i]));

                        std::string simpleElement = compileExpression(std::get<0>(tempChain[i]));
                        simpleChain.push_back(simpleElement);
                    }
                    simpleChain.push_back(simpleHead);

                    std::vector<EncodedExpression> simpleEncodedChain;
                    simpleEncodedChain.reserve(simpleChain.size());
                    for (const std::string& elem : simpleChain)
                    {
                        simpleEncodedChain.push_back(EncodedExpression(elem, "main"));
                    }
                    for (EncodedExpression& ee : simpleEncodedChain)
                    {
                        for (const std::string arg : root->arguments)
                        {
                            for (std::vector<std::string>& argumentDescription : ee.arguments)
                            {
                                if (arg == argumentDescription[1])
                                {
                                    argumentDescription[0] = "True";
                                }
                            }
                        }
                    }

                    if (isSimple and smoothenedSubexprsAfterGrooming.size() == 1)
                    {
                        implName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end() - 1),
                            std::vector<EncodedExpression>(simpleEncodedChain.end() - 1, simpleEncodedChain.end()),
                            "",
                            "implication",
                            true,
                            expression);

                        if (implName != expression)
                        {
							assert(false && "Should not happen");
                        }

                        returnExpression = implName;
                    }
                    else
                    {
                        std::string coreExprImpl = "implication" + std::to_string(definitions.implCounter);
                        definitions.implCounter++;

                        implName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end() - 1),
                            std::vector<EncodedExpression>(simpleEncodedChain.end() - 1, simpleEncodedChain.end()),
                            coreExprImpl,
                            "implication",
                            false,
                            "");


                        std::string tempName = extractExpressionUniversalSpan(StrSpan(implName)).toStdString();
                        if (coreExprImpl != tempName)
                        {
                            definitions.implCounter--;
                        }


                        if (smoothenedSubexprsAfterGrooming.size() == 1)
                        {
                            returnExpression = implName;
                        }
                    }


                    ce::deleteTree(root);

                    overallChain.push_back(implName);
                }

                else if (startsWithSpan(StrSpan(subexpr), "!(>", 3))
                {
                    std::string exName;
                    LogicalEntity le;

                    ce::TreeNode1* root = ce::parseExpr(subexpr);

                    std::tuple<std::string, int, std::string> t = gl::ExpressionParser::renameLastRemoved(subexpr, definitions.variableCounter);
                    definitions.variableCounter++;
                    std::string renamedSubdef = std::get<0>(t);


                    // split into left/right
                    std::pair<std::string, std::string> lr = gl::DefinitionExpansion::extractValues(renamedSubdef, definitions.coreExpressionMap);
                    std::string leftExpr = lr.first;
                    std::string compiledLeftExpr = compileExpression(leftExpr);
                    std::string rightExpr = lr.second;

                    // toggle negation on right
                    if (!rightExpr.empty() && rightExpr[0] == '!') {
                        rightExpr = rightExpr.substr(1);
                    }
                    else {
                        rightExpr = std::string("!") + rightExpr;
                    };
                    std::string compiledRightExpr = compileExpression(rightExpr);


                    std::vector<EncodedExpression> simpleEncodedChain;
                    simpleEncodedChain.reserve(2);

                    simpleEncodedChain.push_back(EncodedExpression(compiledLeftExpr, "main"));
                    simpleEncodedChain.push_back(EncodedExpression(compiledRightExpr, "main"));

                    for (EncodedExpression& ee : simpleEncodedChain)
                    {
                        for (const std::string arg : root->arguments)
                        {
                            for (std::vector<std::string>& argumentDescription : ee.arguments)
                            {
                                if (arg == argumentDescription[1])
                                {
                                    argumentDescription[0] = "True";
                                }
                            }
                        }
                    }

                    if (isSimple and smoothenedSubexprsAfterGrooming.size() == 1)
                    {
                        exName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                            std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                            "",
                            "existence",
                            true,
                            expression);


                        if (exName != expression)
                        {
                            assert(false && "Should not happen");
                        }
                        // Existence mint hook (idempotent on a registry
                        // reuse): the two existence-implication compacts ride
                        // the entity (D-310).
                        existenceRuleCompiler.compileExistenceImplications(extractExpressionUniversalSpan(StrSpan(exName)).toStdString());
                        // Integration hook (idempotent on a registry reuse):
                        // the existence's integration rules ride its entry.
                        integrationRuleCompiler.compileIntegrationImplications(extractExpressionUniversalSpan(StrSpan(exName)).toStdString());

                        returnExpression = exName;
                    }
                    else
                    {
                        std::string coreExprEx = "existence" + std::to_string(definitions.existenceCounter);
                        definitions.existenceCounter++;

                        exName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                            std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                            coreExprEx,
                            "existence",
                            false,
                            "");
                        std::string tempName = extractExpressionUniversalSpan(StrSpan(exName)).toStdString();
                        if (coreExprEx != tempName)
                        {
                            definitions.existenceCounter--;
                        }
                        // Existence mint hook (idempotent on a registry
                        // reuse): the two existence-implication compacts ride
                        // the entity (D-310).
                        existenceRuleCompiler.compileExistenceImplications(tempName);
                        // Integration hook (idempotent on a registry reuse):
                        // the existence's integration rules ride its entry.
                        integrationRuleCompiler.compileIntegrationImplications(tempName);

                        if (smoothenedSubexprsAfterGrooming.size() == 1)
                        {
                            returnExpression = exName;
                        }
                    }



                    ce::deleteTree(root);

                    overallChain.push_back(exName);
                }
                else if (startsWithSpan(StrSpan(subexpr), "!(&", 3))
                {
                    // Negated AND = OR (de Morgan). Compile following existence handler pattern:
                    // !(&(left)(right)) -> (orN[args]) with elements = [compiled_left, compiled_right]
                    std::string orName;

                    ce::TreeNode1* root = ce::parseExpr(subexpr);

                    // split into left/right (the AND conjuncts = negated disjuncts)
                    std::pair<std::string, std::string> lr = gl::DefinitionExpansion::extractValues(subexpr, definitions.coreExpressionMap);
                    std::string compiledLeftExpr = compileExpression(lr.first);
                    std::string compiledRightExpr = compileExpression(lr.second);

                    std::vector<EncodedExpression> simpleEncodedChain;
                    simpleEncodedChain.reserve(2);
                    simpleEncodedChain.push_back(EncodedExpression(compiledLeftExpr, "main"));
                    simpleEncodedChain.push_back(EncodedExpression(compiledRightExpr, "main"));

                    // Mark bound args
                    for (EncodedExpression& ee : simpleEncodedChain)
                    {
                        for (const std::string arg : root->arguments)
                        {
                            for (std::vector<std::string>& argumentDescription : ee.arguments)
                            {
                                if (arg == argumentDescription[1])
                                {
                                    argumentDescription[0] = "True";
                                }
                            }
                        }
                    }

                    if (isSimple && smoothenedSubexprsAfterGrooming.size() == 1)
                    {
                        orName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                            std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                            "",
                            "or",
                            true,
                            expression);
                        assert(orName == expression);
                        returnExpression = orName;
                    }
                    else
                    {
                        std::string coreExprOr = "or" + std::to_string(definitions.orCounter);
                        definitions.orCounter++;

                        orName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                            std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                            coreExprOr,
                            "or",
                            false,
                            "");
                        std::string tempName = extractExpressionUniversalSpan(StrSpan(orName)).toStdString();
                        if (coreExprOr != tempName)
                        {
                            definitions.orCounter--;
                        }

                        if (smoothenedSubexprsAfterGrooming.size() == 1)
                        {
                            returnExpression = orName;
                        }
                    }

                    ce::deleteTree(root);
                    overallChain.push_back(orName);
                }
                else if (ce::expressionIsSimple(subexpr))
                {
                    std::string compiledExpr = compileExpression(subexpr);
                    assert(compiledExpr == subexpr && "Should not happen");
                    overallChain.push_back(compiledExpr);
                }
                else
                {
					assert(false && "Should not happen");
                }

            }

            if (smoothenedSubexprsAfterGrooming.size() > 1)
            {
                std::string andName;
                LogicalEntity le;

                std::vector<EncodedExpression> simpleEncodedChain;
                simpleEncodedChain.reserve(overallChain.size());

                for (const auto& elem : overallChain)
                {
                    simpleEncodedChain.push_back(EncodedExpression(elem, "main"));
                }

                for (EncodedExpression& ee : simpleEncodedChain)
                {
                    for (std::vector<std::string>& argumentDescription : ee.arguments)
                    {
                        argumentDescription[0] = "True";
                    }
                }

                if (isSimple)
                {
                    std::string coreExpression = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();
                    andName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                        std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                        "",
                        "and",
                        true,
                        expression);


                    if (andName != expression)
                    {
                        assert(false && "Should not happen");
                    }
                    // Integration hook: the and's integration rules ride
                    // its entry.
                    integrationRuleCompiler.compileIntegrationImplications(coreExpression);
                }
                else
                {
                    ce::TreeNode1* root = ce::parseExpr(expression);

                    std::string coreExprAnd = "and" + std::to_string(definitions.andCounter);
                    definitions.andCounter++;

                    andName = compilationReuse.excludeRepetitions(std::vector<EncodedExpression>(simpleEncodedChain.begin(), simpleEncodedChain.end()),
                        std::vector<EncodedExpression>(simpleEncodedChain.end(), simpleEncodedChain.end()),
                        coreExprAnd,
                        "and",
                        false,
                        "");
                    std::string tempName = extractExpressionUniversalSpan(StrSpan(andName)).toStdString();
                    if (coreExprAnd != tempName)
                    {
                        definitions.andCounter--;
                    }
                    // Integration hook (idempotent on a registry reuse): the
                    // and's integration rules ride its entry.
                    integrationRuleCompiler.compileIntegrationImplications(tempName);

                    ce::deleteTree(root);
                }

                returnExpression = andName;
            }

            if (isSimple)
            {
                returnExpression = inputExpression;
            }

            return returnExpression;
        }

/// @brief Compile configured definitions and derive their type metadata.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @return None.
void StructuralCompiler::compileCoreExpressionMap() {
            // The configured names are known before any definition compiles:
            // the integration hooks classify configured compounds while the
            // definitions are compiled.
            for (const auto& kv : definitions.coreExpressionMap) {
                definitions.expressionsFromConfig.insert(kv.first);
            }

            // ---------------------------------------------------------
            // 1. Compilation Phase
            // Must run FIRST so 'compiledExpressions' (LogicalEntities) exist.
            // The atomic definitions register before any compound: a
            // compound's integration set is compiled at its mint and resolves
            // every core of its definition tree, and a registry seeded from
            // the shared binary holds the compound definitions but no atomic
            // one (D-503).
            // ---------------------------------------------------------
            for (const auto& coreExpression : definitions.coreExpressionMap)
            {
                const std::string& signature = coreExpression.second.signature;
                if (definitionExpansion.expandExpr(signature) == signature)
                    compileExpression(signature);
            }
            for (auto coreExpression : definitions.coreExpressionMap)
            {
                compileExpression(coreExpression.second.signature);
            }

            compilationReuse.finalizeCompilationReplacements();

            // ---------------------------------------------------------
            // 2. Analysis Phase
            // Updates the existing LogicalEntities with 'definedSet' info.
            // ---------------------------------------------------------
            gl::ArgumentAnalyzer analyzer(definitions.coreExpressionMap);

            for (const auto& kv : definitions.coreExpressionMap) {
                const std::string& key = kv.first; // "identity", "interval", etc.
                const ce::CoreExpressionConfig& cfg = kv.second; // Already accessed, no need to look up

                std::string defString;
                if (std::holds_alternative<std::string>(cfg.definition)) {
                    defString = std::get<std::string>(cfg.definition);
                }

                if (!defString.empty()) {
                    // Build the tree
                    gl::AnalysisNode* root = analyzer.analyze(defString);
                    assert(root != nullptr && "Analyzer produced null for non-empty definition.");

                    // Use the instance 'analyzer' to call the static method (safer syntax)
                    analyzer.checkDefinitionConsistency(root, cfg, cfg.signature);

                    std::string setArg;
                    std::string defSet;

                    // Check if this definition describes a Set
                    if (analyzer.isSetDefinition(root, setArg, defSet)) {

                        // 1. Find the index of the set argument in the Config Signature
                        //    We use cfg.signature because 'setArg' matches the variable names used there.
                        std::vector<std::string> cfgArgs = ce::getArgs(cfg.signature);
                        auto itArg = std::find(cfgArgs.begin(), cfgArgs.end(), setArg);

                        // Strict assertion: The variable found in the definition MUST exist in the signature
                        assert(itArg != cfgArgs.end() && "Set argument from definition not found in configuration signature.");

                        int index = static_cast<int>(std::distance(cfgArgs.begin(), itArg));

                        // 2. Retrieve the compiled LogicalEntity using the loop key
                        auto leIt = definitions.compiledExpressions.find(key);
                        assert(leIt != definitions.compiledExpressions.end() && "LogicalEntity not found in compiled expressions.");

                        // 3. Extract the corresponding argument from the compiled signature
                        //    The names here (e.g., u_1, u_2) might differ from config, but the position (index) is invariant.
                        std::vector<std::string> compiledArgs = ce::getArgs(leIt->second.signature);
                        assert(index >= 0 && index < static_cast<int>(compiledArgs.size()) && "Argument index out of bounds in compiled signature.");

                        // 4. Update the definedSet field
                        leIt->second.definedSet = compiledArgs[index];

                        /*
                        std::cout << ">>> [Set Definition Detected] Key: " << key
                        << " | Variable: " << setArg
                        << " | Index: " << index
                        << " | Mapped to Sig Arg: " << leIt->second.definedSet << std::endl;
                        */

                    }

                    delete root;
                }
            }

            checkCompiledCoreExpressionMap();


            extendCoreExpressionMap();
        }

/// @brief Register configuration metadata for generated logical definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @return None.
void StructuralCompiler::extendCoreExpressionMap() {
            for (const auto& kv : definitions.compiledExpressions) {
                const std::string& key = kv.first;
                const LogicalEntity& le = kv.second;

                // Only process if missing from coreExpressionMap
                if (definitions.coreExpressionMap.find(key) == definitions.coreExpressionMap.end()) {
                    ce::CoreExpressionConfig cfg;
                    cfg.arity = le.arity;


                    const std::string cleanSig = gl::DefinitionExpansion::removeUPrefixFromArguments(le.signature);
                    cfg.signature = cleanSig;
                    cfg.definition = cleanSig; // As requested: fill definition same as signature

                    // Populate definitionSets by tracing arguments
                    std::vector<std::string> sigArgs = ce::getArgs(le.signature);
                    for (size_t i = 0; i < sigArgs.size(); ++i) {
                        std::string defSet = findDefinitionSetRecursive(sigArgs[i], le);
                        if (!defSet.empty()) {
                            // pair[0] = definitionSet, pair[1] = false
                            cfg.definitionSets[std::to_string(i + 1)] = std::make_pair(defSet, false);
                        }
                    }

                    definitions.coreExpressionMap[key] = cfg;
                }
            }
        }

/// @brief Compile a mail-bound implication into its compact named
        /// form `(implication<N>[args])`.
        ///
        /// @details
        /// Wrapper invoked at every site where a proved or internal
        /// implication is broadcast on the mail "implications" channel.
        /// It normalises the implication's free arguments and delegates to
        /// the existing `compileExpression` implication branch
        /// (the `(>` branch — the only branch reached for a top-level
        /// non-negated implication), returning the compact instance.
        ///
        /// Free arguments arrive in `u_`-prefixed template form for
        /// internal implications; a fully-bound proved theorem carries no
        /// `u_` at all (every variable is bound — see
        /// [D-75](../../docs/agentic_swdd/40_decisions.md)). `stripUPrefixAST` removes a
        /// leading `u_` from every argument token at every nesting level,
        /// yielding a uniform bare-argument `(>[]()())` string before
        /// compilation. A fully-bound theorem then has zero free arguments
        /// and compiles to `(implication<N>[])`; an implication with
        /// genuinely-free arguments compiles to `(implication<N>[args])`.
        ///
        /// Preparatory for ASIC 0.1: the returned compact form is
        /// additionally deposited as a mail expression; it has no effect on
        /// which theorems prove.
        ///
        /// No defensive fallback (CLAUDE.md Rule 19): an implication that
        /// reached the mail channel and fails to compile to a simple
        /// `(implication<N>[...])` is a bug and asserts at its origin — it
        /// never returns an empty / sentinel string the caller would have
        /// to test-and-skip.
        ///
        /// @param implication Canonical MPL implication string — the
        ///        original expanded `(>[]()())` as it enters the mail
        ///        channel (the 5th element of a `mailOut.implications`
        ///        tuple).
        /// @return The compact `(implication<N>[args])` form;
        ///         `(implication<N>[])` for a fully-bound theorem.
        /// @invariant The result is non-empty, satisfies
        ///            `ce::expressionIsSimple`, and its core name begins
        ///            with `"implication"`.
        /// @see stripUPrefixAST, compileExpression,
        ///      excludeRepetitions
std::string StructuralCompiler::compileImplicationToCompact(const std::string& implication) {
    // Normalise free args to bare form: u_-prefixed template params (internal
    // implications) and unprefixed bound vars (proved theorems) both reduce
    // to the same shape, so compileExpression's (>) branch sees a
    // uniform input. stripUPrefixAST walks the full parse tree (every nesting
    // level), unlike removeUPrefixFromArguments which only sees the first [...].
    const std::string stripped = gl::DefinitionExpansion::stripUPrefixAST(implication);

    // The (>) branch of compileExpression allocates / reuses the
    // implication<N> name and returns the compact instance. Counters are the
    // canonical this-> members so allocation stays continuous and (because
    // this runs only on the single-threaded post-join broadcast drain)
    // deterministic.
    const std::string compact = compileExpression(
        stripped);

    // CLAUDE.md Rule 19 — no defensive fallback. An implication that entered
    // mail MUST compile to a simple (implication<N>[...]); a miss is a bug we
    // trap here, with the stack intact, not a value the caller skips.
    assert(!compact.empty()
        && "compileImplicationToCompact: empty compile result for a mailed implication");
    assert(ce::expressionIsSimple(compact)
        && "compileImplicationToCompact: result is not a simple compact form");
    assert(startsWithSpan(StrSpan(extractExpressionUniversalSpan(StrSpan(compact)).toStdString()), "implication", 11)
        && "compileImplicationToCompact: result core name is not implication<N>");

    return compact;
}

/// @brief Perform precompileStructuralOperators using the registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param theorem Input to this compilation operation.
/// @return None.
void StructuralCompiler::precompileStructuralOperators(std::string& theorem) {
    auto compileAtMarker = [&](const char* marker /* "!(&" or "!(>" */) {
        while (true) {
            auto pos = theorem.find(marker);
            if (pos == std::string::npos) return;

            // Bracket-balanced scan starting at the '(' right after '!'.
            int depth = 0;
            std::size_t end = std::string::npos;
            for (std::size_t i = pos + 1; i < theorem.size(); ++i) {
                if (theorem[i] == '(') {
                    ++depth;
                } else if (theorem[i] == ')') {
                    --depth;
                    if (depth == 0) { end = i; break; }
                }
            }
            if (end == std::string::npos) return;  // malformed, bail

            std::string sub = theorem.substr(pos, end - pos + 1);
            std::string compiled = compileExpression(
                sub);

            if (compiled.empty() || compiled == sub) return;  // no progress
            theorem = theorem.substr(0, pos) + compiled + theorem.substr(end + 1);
        }
    };

    // Pass 1: compile all !(&...) — these become (or<N>[...]).
    compileAtMarker("!(&");
    // Pass 2: compile all !(>...) — these become (existence<N>[...]).
    compileAtMarker("!(>");
    supportingRuleCompiler.compileContrapositionFamilies();
    supportingRuleCompiler.compileDisintegrationPackages();
}

/// @brief Compile, once, the extensionality entity of a non-digit
        ///        definition set.
        ///
        /// @details
        /// Finds the membership operator (`in`, `in2`, `in3`) whose last slot
        /// carries @p label — asserts when none does (no extensionality for
        /// that type). Compiles its subset implication
        /// `(>[x..](m[x..,A])(m[x..,B]))` through the ordinary expression
        /// compiler, with non-numeric names so no bound variable meets a free
        /// one. An existing compact with the same body is reused: a numbered
        /// `implication<N>`, or the configured definition whose whole body is
        /// this implication (`subset[X,Y]` for sets in FTA), whose public name
        /// replaces every numbered equivalent (D-349) and is itself the rule's
        /// compact form (D-467). The compact instance's argument order fixes
        /// which formal is `A`. Runs only on the single-threaded registration
        /// seams, because compiling mints registry entries.
        ///
        /// @param label A definition set other than `(1)`.
        /// @invariant At most one entity per label.
        /// @see typedEqualityEntity, ensureTypedEqualityEntityForTheorem.
/// @return None.
void StructuralCompiler::ensureTypedEqualityEntity(StrSpan label) {
            assert(!equalSpans(label, StrSpan("(1)", 3))
                && "ensureTypedEqualityEntity: digits need no extensionality");
            if (definitions.typedEqualityEntity(label) != nullptr) return;

            static const char* const kMembership[] = { "in", "in2", "in3" };
            std::string op;
            int opArity = 0;
            for (const char* m : kMembership) {
                const ce::CoreExpressionConfig* cfg =
                    definitions.coreConfig(StrSpan(m, static_cast<int32_t>(std::strlen(m))));
                if (cfg == nullptr) continue;
                const std::pair<std::string, bool>* ds = gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, cfg->arity);
                if (ds != nullptr && equalSpans(StrSpan(ds->first), label)) {
                    op = m;
                    opArity = cfg->arity;
                    break;
                }
            }
            assert(!op.empty() && "ensureTypedEqualityEntity: no membership operator carries this definition set");

            std::string bound;
            for (int i = 1; i < opArity; ++i) {
                if (i > 1) bound += ",";
                bound += "ext_x" + std::to_string(i);
            }
            const std::string body = "(>[" + bound + "](" + op + "[" + bound + ",ext_A])("
                + op + "[" + bound + ",ext_B]))";
            // Not compileImplicationToCompact: its numbered-name contract
            // serves mailed implications, while a configured definition may
            // own this body (D-349).
            const std::string compact = compileExpression(body);
            assert(ce::expressionIsSimple(compact)
                && "ensureTypedEqualityEntity: subset implication did not compile to a simple compact");
            const LogicalEntity* subsetEntity = definitions.compiledEntity(extractExpressionSpan(StrSpan(compact)));
            assert(subsetEntity != nullptr && subsetEntity->category == "implication"
                && "ensureTypedEqualityEntity: subset compact is not a compiled implication");

            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(StrSpan(compact), args, ExecutionParameters::MAX_ARITY);
            assert(argN == 2 && "ensureTypedEqualityEntity: subset compact must take the two sets");
            const bool aFirst = equalSpans(args[0], StrSpan("ext_A", 5));
            assert((aFirst ? equalSpans(args[1], StrSpan("ext_B", 5))
                           : (equalSpans(args[0], StrSpan("ext_B", 5))
                              && equalSpans(args[1], StrSpan("ext_A", 5))))
                && "ensureTypedEqualityEntity: subset compact arguments are not the two sets");
            const std::string name = extractExpressionSpan(StrSpan(compact)).toStdString();
            const std::string subsetAB = "(" + name + (aFirst ? "[u_1,u_2])" : "[u_2,u_1])");
            const std::string subsetBA = "(" + name + (aFirst ? "[u_2,u_1])" : "[u_1,u_2])");

            CompiledDefinitionRegistry::TypedEqualityEntity typed;
            typed.label = label.toStdString();
            typed.entity = LogicalEntity("and", { subsetAB, subsetBA }, "(=[u_1,u_2])", 2);
            // The extensionality rule (both subset implications imply the
            // equality) is the entity's integration set, compiled at its
            // creation like every other entity's.
            typed.entity.integrationImplications =
                integrationRuleCompiler.compileIntegrationFamily(typed.entity, false);
            definitions.typedEqualityEntities.push_back(std::move(typed));
        }

/// @brief Verify that compiled implication heads resolve to registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @return None.
void StructuralCompiler::checkCompiledCoreExpressionMap() {
            // Local helper struct for recursion to verify no nested implications exist.
            struct DeepChecker {
                const CompiledExpressionMap& definitions;

                DeepChecker(const CompiledExpressionMap& defs) : definitions(defs) {}

                void check(const std::string& expression) const {
                    // 1. Identify the core entity name from the expression string
                    std::string core = extractExpressionUniversalSpan(StrSpan(expression)).toStdString();

                    auto it = definitions.find(core);

                    // Base Case 1: The core MUST be in our compiled map.
                    // If it is missing, we assert failure instead of treating it as a variable/primitive.
                    if (it == definitions.end()) {
                        std::cerr << "Assertion Failed: Core expression not found in definitions." << std::endl;
                        std::cerr << "  Expression: " << expression << std::endl;
                        std::cerr << "  Core: " << core << std::endl;
                        assert(false && "Core expression not found in compiled expressions.");
                    }

                    const LogicalEntity& le = it->second;

                    // 2. Nested implication: ACCEPTED — recursion stop.
                    // The historical "the prover does not support an implication
                    // as a constituent of another compiled expression" invariant
                    // is RETIRED per explicit user decision
                    // (see D-76). It predated
                    // top-level implication compaction: a top-level mail
                    // implication is now compiled to compact implication<N>
                    // form, so an implication<N> legitimately appears as a
                    // constituent — e.g. the head of a multi-premise theorem
                    // (whose inner > the compaction recurses), or a member of a
                    // hand-defined `and` operator like fXY / identity after
                    // cross-batch compact-name reuse. Treat it like an atomic
                    // leaf: stop here, do not recurse into its chain. The
                    // "core not found" guard above stays in force for the
                    // and/existence elements still walked below.
                    if (le.category == "implication") {
                        return;
                    }

                    // Base Case 2: Atomic entities stop the recursion.
                    if (le.category == "atomic") {
                        return;
                    }

                    // 3. Recursive Step
                    // For categories like "and", "existence", etc., we must deep dive into their elements.
                    for (const std::string& subElement : le.elements) {
                        check(subElement);
                    }
                }
            };

            DeepChecker checker(definitions.compiledExpressions);

            // Iterate through all compiled LogicalEntities
            for (auto it = definitions.compiledExpressions.begin(); it != definitions.compiledExpressions.end(); ++it) {
                const LogicalEntity& le = it->second;

                // We only initiate the deep check if the top-level entity is itself an implication.
                // We want to ensure that NONE of its constituents (body or head) contain an implication.
                if (le.category == "implication") {
                    checker.check(le.elements.back());
                }
            }
        }

/// @brief Resolve the definition set of a formal argument through compiled elements.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param var Formal argument to resolve.
/// @param le Compiled logical entity to inspect.
/// @return The result of the documented operation.
std::string StructuralCompiler::findDefinitionSetRecursive(const std::string& var, const LogicalEntity& le) {
            for (const std::string& elem : le.elements) {
                std::string core = ce::extractExpression(elem);
                std::vector<std::string> args = ce::getArgs(elem);

                for (size_t i = 0; i < args.size(); ++i) {
                    if (args[i] == var) {
                        // Case 1: Element is in coreExpressionMap (atomic/base operator)
                        if (definitions.coreExpressionMap.find(core) != definitions.coreExpressionMap.end()) {
                            std::string idx = std::to_string(i + 1);
                            const auto& ds = definitions.coreExpressionMap.at(core).definitionSets;
                            if (ds.find(idx) != ds.end()) {
                                return ds.at(idx).first;
                            }
                        }
                        // Case 2: Element is in compiledExpressions (recurse into sub-macro)
                        else if (definitions.compiledExpressions.find(core) != definitions.compiledExpressions.end()) {
                            const LogicalEntity& subLe = definitions.compiledExpressions.at(core);
                            std::vector<std::string> sigArgs = ce::getArgs(subLe.signature);
                            if (i < sigArgs.size()) {
                                std::string mappedVar = sigArgs[i];
                                std::string res = findDefinitionSetRecursive(mappedVar, subLe);
                                if (!res.empty()) return res;
                            }
                        }
                    }
                }
            }
            return "";
        }

} // namespace gl
