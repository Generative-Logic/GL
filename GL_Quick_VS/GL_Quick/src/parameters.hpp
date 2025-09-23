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

#pragma once
// Subset of parameters needed for quick mode scaffold.
// Values mirror GL/parameters.py (quick defaults).
namespace parameters {
    // from parameters.py
    static const int sizeAllBinariesAna = 10;

    static const int maxIterationNumberProof = 20;
	static const int numberIterationsConjectureFiltering = 1;

    static const int maxSizeDefSetMapping = 5;
    static const int maxSizeTargetSetMapping = 12;
    static const int maxNumberSecondaryVariables = 2;
	static const int sizeAllPermutationsAna = 7;
	static const int minNumOperatorsKey = 2;
    static const int maxIterationNumberVariable = 1;
    static const int  standardMaxSecondaryNumber = 1;
    static const bool trackHistory = true;
	static const int standardMaxAdmissionDepth = 0;
    static const int inductionMaxAdmissionDepth = 1;
    static const int inductionMaxSecondaryNumber = 2;
    static const int counterExampleBoundary = 6;
    
    static const bool debug = false;

}
