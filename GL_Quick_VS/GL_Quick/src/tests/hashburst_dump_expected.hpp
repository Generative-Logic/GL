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
#pragma once
namespace gl { namespace tests {
inline constexpr const char* expectedHashburstDumpParts[] = {
R"GL_DUMP(
=== HASHBURST ENTRY #1 | stmts=1 | origins=0 | hashMem=0 | toBeProved=1 ===
-- LB chain (innermost -> root):
  [0] (fixture[u_1])
  [1] 
-- encodedStatements:
  [0] (p[a]) | v=main
-- toBeProved (1):
  (q[a]) | v=main | auxies={0} | tags={}
-- exprOriginMap:
-- overallHashMemory.originals (0):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.encodedMap marker entries (0):
-- mailIn.statements (0):
-- mailIn.exprOriginMap (0):
-- compiledExpressions (3):
  orFixture | sig=(orFixture[u_1])
  p | sig=(p[u_1])
  q | sig=(q[u_1])
-- counters: startInt=7 startIntRepl=0 startIntPi=0 level=0 isActive=1 isPartOfRecursion=0 deltaNumberStatements=0 recursionCounter=0 contradictionIndex=-1 primedForContradiction=0
-- nameMap.idToName (6 entries; nextId=6):
  [0] 
  [1] main
  [2] p
  [3] (p[a])
  [4] a
  [5] (q[a])
-- nameMap.idToSub (1 entries; nextSubId=1):
  [0] 
-- nameMap.stackOfValidity (6):
  [0] {}
  [1] {}
  [2] {}
  [3] {}
  [4] {}
  [5] {}
-- nameMap.ancestorsOf (6):
  [0] {}
  [1] {1}
  [2] {2}
  [3] {3}
  [4] {4}
  [5] {5}
-- nameMap.pairMap (0 entries; size only)
-- statementLevelsMap (0):
-- localEncodedStatements (0):
-- localEncodedStatementsDelta (0):
-- intKnownStatements (0 packed (origId,validityId) keys):
-- equivalenceClassesMap (0):
-- integrationPrepared (0):
-- integrationPreparedMarker (0):
-- integrationStartIntMap (0):
-- expandedImplications (0):
-- weakVariables (0):
-- orAdmissionSet (0):
-- orBookkeeping (0):
-- orDisjunctCount (0):
-- orPendingBranches (0):
-- intValidityNamesToFilter (0):
-- pendingWipeScopes (0):
-- axedVariables (0):
-- intAxedVariables (0):
-- mailOut.statements (0):
-- mailOut.exprOriginMap (0):
-- sameIterationInternalMail.statements (0):
-- sameIterationInternalMail.exprOriginMap (0):
-- overallHashMemory.encodedMap (0 keys):
-- overallHashMemory.normalizedEncodedKeys (0 keys, set):
-- overallHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- overallHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.admissionMapIntegration (0):
-- overallHashMemory.admissionSetIntegration (0):
-- overallHashMemory.admissionMapIntegrationDouble (0):
-- overallHashMemory.triggersForAdmissionSetIntegration (0):
-- overallHashMemory.rejectedMap (0):
-- overallHashMemory.rejectedMapIntegration (0):
-- overallHashMemory.rejectedMapOrdis (0):
-- overallHashMemory.rejectedMapOrdis2 (0):
-- overallHashMemory.admissionMapOrdis2 (0):
-- overallHashMemory.rejectedMapSubproof (0):
-- overallHashMemory.varsInRejectedMapIntegrationKeys (0):
-- overallHashMemory.varsInAdmissionMapKeys (0):
-- overallHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- overallHashMemory.admissionStatusMap (0):
-- overallHashMemory.productsOfRecursion (0):
-- overallHashMemory.productsOfRecursionIds (0):
-- overallHashMemory.revisitInProgress (0):
-- overallHashMemory.maxKeyLength=0
-- localHashMemory.encodedMap (0 keys):
-- localHashMemory.normalizedEncodedKeys (0 keys, set):
-- localHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemory.admissionMap (0):
-- localHashMemory.admissionMapIntegration (0):
-- localHashMemory.admissionSetIntegration (0):
-- localHashMemory.admissionMapIntegrationDouble (0):
-- localHashMemory.triggersForAdmissionSetIntegration (0):
-- localHashMemory.rejectedMap (0):
-- localHashMemory.rejectedMapIntegration (0):
-- localHashMemory.rejectedMapOrdis (0):
-- localHashMemory.rejectedMapOrdis2 (0):
-- localHashMemory.admissionMapOrdis2 (0):
-- localHashMemory.rejectedMapSubproof (0):
-- localHashMemory.varsInRejectedMapIntegrationKeys (0):
-- localHashMemory.varsInAdmissionMapKeys (0):
-- localHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemory.admissionStatusMap (0):
-- localHashMemory.productsOfRecursion (0):
-- localHashMemory.productsOfRecursionIds (0):
-- localHashMemory.revisitInProgress (0):
-- localHashMemory.maxKeyLength=0
-- localHashMemoryDelta.encodedMap (0 keys):
-- localHashMemoryDelta.norma)GL_DUMP",
R"GL_DUMP(lizedEncodedKeys (0 keys, set):
-- localHashMemoryDelta.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemoryDelta.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemoryDelta.admissionMap (0):
-- localHashMemoryDelta.admissionMapIntegration (0):
-- localHashMemoryDelta.admissionSetIntegration (0):
-- localHashMemoryDelta.admissionMapIntegrationDouble (0):
-- localHashMemoryDelta.triggersForAdmissionSetIntegration (0):
-- localHashMemoryDelta.rejectedMap (0):
-- localHashMemoryDelta.rejectedMapIntegration (0):
-- localHashMemoryDelta.rejectedMapOrdis (0):
-- localHashMemoryDelta.rejectedMapOrdis2 (0):
-- localHashMemoryDelta.admissionMapOrdis2 (0):
-- localHashMemoryDelta.rejectedMapSubproof (0):
-- localHashMemoryDelta.varsInRejectedMapIntegrationKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemoryDelta.admissionStatusMap (0):
-- localHashMemoryDelta.productsOfRecursion (0):
-- localHashMemoryDelta.productsOfRecursionIds (0):
-- localHashMemoryDelta.revisitInProgress (0):
-- localHashMemoryDelta.maxKeyLength=0
-- workingMemory.encodedMap (0 keys):
-- workingMemory.normalizedEncodedKeys (0 keys, set):
-- workingMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- workingMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- workingMemory.admissionMap (0):
-- workingMemory.admissionMapIntegration (0):
-- workingMemory.admissionSetIntegration (0):
-- workingMemory.admissionMapIntegrationDouble (0):
-- workingMemory.triggersForAdmissionSetIntegration (0):
-- workingMemory.rejectedMap (0):
-- workingMemory.rejectedMapIntegration (0):
-- workingMemory.rejectedMapOrdis (0):
-- workingMemory.rejectedMapOrdis2 (0):
-- workingMemory.admissionMapOrdis2 (0):
-- workingMemory.rejectedMapSubproof (0):
-- workingMemory.varsInRejectedMapIntegrationKeys (0):
-- workingMemory.varsInAdmissionMapKeys (0):
-- workingMemory.varsInAdmissionMapIntegrationKeys (0):
-- workingMemory.admissionStatusMap (0):
-- workingMemory.productsOfRecursion (0):
-- workingMemory.productsOfRecursionIds (0):
-- workingMemory.revisitInProgress (0):
-- workingMemory.maxKeyLength=0
-- externalStatements (0):
-- intExternalStatements (0 — lockstep int mirror of externalStatements)

=== HASHBURST EXIT #1 | stmts=1 | origins=0 | hashMem=0 | toBeProved=1 ===
-- LB chain (innermost -> root):
  [0] (fixture[u_1])
  [1] 
-- encodedStatements:
  [0] (p[a]) | v=main
-- toBeProved (1):
  (q[a]) | v=main | auxies={0} | tags={}
-- exprOriginMap:
-- overallHashMemory.originals (0):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.encodedMap marker entries (0):
-- mailIn.statements (0):
-- mailIn.exprOriginMap (0):
-- counters: startInt=7 startIntRepl=0 startIntPi=0 level=0 isActive=1 isPartOfRecursion=0 deltaNumberStatements=0 recursionCounter=0 contradictionIndex=-1 primedForContradiction=0
-- nameMap.idToName (6 entries; nextId=6):
  [0] 
  [1] main
  [2] p
  [3] (p[a])
  [4] a
  [5] (q[a])
-- nameMap.idToSub (1 entries; nextSubId=1):
  [0] 
-- nameMap.stackOfValidity (6):
  [0] {}
  [1] {}
  [2] {}
  [3] {}
  [4] {}
  [5] {}
-- nameMap.ancestorsOf (6):
  [0] {}
  [1] {1}
  [2] {2}
  [3] {3}
  [4] {4}
  [5] {5}
-- nameMap.pairMap (0 entries; size only)
-- statementLevelsMap (0):
-- localEncodedStatements (0):
-- localEncodedStatementsDelta (0):
-- intKnownStatements (0 packed (origId,validityId) keys):
-- equivalenceClassesMap (0):
-- integrationPrepared (0):
-- integrationPreparedMarker (0):
-- integrationStartIntMap (0):
-- expandedImplications (0):
-- weakVariables (0):
-- orAdmissionSet (0):
-- orBookkeeping (0):
-- orDisjunctCount (0):
-- orPendingBranches (0):
-- intValidityNamesToFilter (0):
-- pendingWipeScopes (0):
-- axedVariables (0):
-- intAxedVariables (0):
-- mailOut.statements (0):
-- mailOut.exprOriginMap (0):
-- sameIterationInternalMail.statements (0):
-- sameIterationInternalMail.exprOriginMap (0):
-- overallHashMemory.encodedMap (0 keys):
-- overallHashMemory.normalizedEncodedKeys (0 keys, set):
-- overallHashMemory.normali)GL_DUMP",
R"GL_DUMP(zedEncodedSubkeys (0 keys, u_-signature record):
-- overallHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.admissionMapIntegration (0):
-- overallHashMemory.admissionSetIntegration (0):
-- overallHashMemory.admissionMapIntegrationDouble (0):
-- overallHashMemory.triggersForAdmissionSetIntegration (0):
-- overallHashMemory.rejectedMap (0):
-- overallHashMemory.rejectedMapIntegration (0):
-- overallHashMemory.rejectedMapOrdis (0):
-- overallHashMemory.rejectedMapOrdis2 (0):
-- overallHashMemory.admissionMapOrdis2 (0):
-- overallHashMemory.rejectedMapSubproof (0):
-- overallHashMemory.varsInRejectedMapIntegrationKeys (0):
-- overallHashMemory.varsInAdmissionMapKeys (0):
-- overallHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- overallHashMemory.admissionStatusMap (0):
-- overallHashMemory.productsOfRecursion (0):
-- overallHashMemory.productsOfRecursionIds (0):
-- overallHashMemory.revisitInProgress (0):
-- overallHashMemory.maxKeyLength=0
-- localHashMemory.encodedMap (0 keys):
-- localHashMemory.normalizedEncodedKeys (0 keys, set):
-- localHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemory.admissionMap (0):
-- localHashMemory.admissionMapIntegration (0):
-- localHashMemory.admissionSetIntegration (0):
-- localHashMemory.admissionMapIntegrationDouble (0):
-- localHashMemory.triggersForAdmissionSetIntegration (0):
-- localHashMemory.rejectedMap (0):
-- localHashMemory.rejectedMapIntegration (0):
-- localHashMemory.rejectedMapOrdis (0):
-- localHashMemory.rejectedMapOrdis2 (0):
-- localHashMemory.admissionMapOrdis2 (0):
-- localHashMemory.rejectedMapSubproof (0):
-- localHashMemory.varsInRejectedMapIntegrationKeys (0):
-- localHashMemory.varsInAdmissionMapKeys (0):
-- localHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemory.admissionStatusMap (0):
-- localHashMemory.productsOfRecursion (0):
-- localHashMemory.productsOfRecursionIds (0):
-- localHashMemory.revisitInProgress (0):
-- localHashMemory.maxKeyLength=0
-- localHashMemoryDelta.encodedMap (0 keys):
-- localHashMemoryDelta.normalizedEncodedKeys (0 keys, set):
-- localHashMemoryDelta.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemoryDelta.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemoryDelta.admissionMap (0):
-- localHashMemoryDelta.admissionMapIntegration (0):
-- localHashMemoryDelta.admissionSetIntegration (0):
-- localHashMemoryDelta.admissionMapIntegrationDouble (0):
-- localHashMemoryDelta.triggersForAdmissionSetIntegration (0):
-- localHashMemoryDelta.rejectedMap (0):
-- localHashMemoryDelta.rejectedMapIntegration (0):
-- localHashMemoryDelta.rejectedMapOrdis (0):
-- localHashMemoryDelta.rejectedMapOrdis2 (0):
-- localHashMemoryDelta.admissionMapOrdis2 (0):
-- localHashMemoryDelta.rejectedMapSubproof (0):
-- localHashMemoryDelta.varsInRejectedMapIntegrationKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemoryDelta.admissionStatusMap (0):
-- localHashMemoryDelta.productsOfRecursion (0):
-- localHashMemoryDelta.productsOfRecursionIds (0):
-- localHashMemoryDelta.revisitInProgress (0):
-- localHashMemoryDelta.maxKeyLength=0
-- workingMemory.encodedMap (0 keys):
-- workingMemory.normalizedEncodedKeys (0 keys, set):
-- workingMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- workingMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- workingMemory.admissionMap (0):
-- workingMemory.admissionMapIntegration (0):
-- workingMemory.admissionSetIntegration (0):
-- workingMemory.admissionMapIntegrationDouble (0):
-- workingMemory.triggersForAdmissionSetIntegration (0):
-- workingMemory.rejectedMap (0):
-- workingMemory.rejectedMapIntegration (0):
-- workingMemory.rejectedMapOrdis (0):
-- workingMemory.rejectedMapOrdis2 (0):
-- workingMemory.admissionMapOrdis2 (0):
-- workingMemory.rejectedMapSubproof (0):
-- workingMemory.varsInRejectedMapIntegrationKeys (0):
-- workingMemory.varsInAdmissionMapKeys (0):
-- workingMemory.varsInAdmissionMapIntegrationKeys (0))GL_DUMP",
R"GL_DUMP(:
-- workingMemory.admissionStatusMap (0):
-- workingMemory.productsOfRecursion (0):
-- workingMemory.productsOfRecursionIds (0):
-- workingMemory.revisitInProgress (0):
-- workingMemory.maxKeyLength=0
-- externalStatements (0):
-- intExternalStatements (0 — lockstep int mirror of externalStatements)
-- phase2.statementUpload (0 uploads, 0 sent, 0 withheld by the premise patterns):

=== HASHBURST ENTRY #2 | stmts=2 | origins=0 | hashMem=0 | toBeProved=1 ===
-- LB chain (innermost -> root):
  [0] (fixture[u_1])
  [1] 
-- encodedStatements:
  [0] (p[a]) | v=main
  [1] !(q[a]) | v=main
-- toBeProved (1):
  (q[a]) | v=main | auxies={0} | tags={}
-- exprOriginMap:
-- overallHashMemory.originals (0):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.encodedMap marker entries (0):
-- mailIn.statements (0):
-- mailIn.exprOriginMap (0):
-- counters: startInt=9 startIntRepl=0 startIntPi=0 level=0 isActive=1 isPartOfRecursion=0 deltaNumberStatements=0 recursionCounter=0 contradictionIndex=-1 primedForContradiction=0
-- nameMap.idToName (8 entries; nextId=8):
  [0] 
  [1] main
  [2] p
  [3] (p[a])
  [4] a
  [5] (q[a])
  [6] q
  [7] !(q[a])
-- nameMap.idToSub (1 entries; nextSubId=1):
  [0] 
-- nameMap.stackOfValidity (8):
  [0] {}
  [1] {}
  [2] {}
  [3] {}
  [4] {}
  [5] {}
  [6] {}
  [7] {}
-- nameMap.ancestorsOf (8):
  [0] {}
  [1] {1}
  [2] {2}
  [3] {3}
  [4] {4}
  [5] {5}
  [6] {6}
  [7] {7}
-- nameMap.pairMap (0 entries; size only)
-- statementLevelsMap (0):
-- localEncodedStatements (0):
-- localEncodedStatementsDelta (0):
-- intKnownStatements (0 packed (origId,validityId) keys):
-- equivalenceClassesMap (0):
-- integrationPrepared (0):
-- integrationPreparedMarker (0):
-- integrationStartIntMap (0):
-- expandedImplications (0):
-- weakVariables (0):
-- orAdmissionSet (0):
-- orBookkeeping (0):
-- orDisjunctCount (0):
-- orPendingBranches (0):
-- intValidityNamesToFilter (0):
-- pendingWipeScopes (0):
-- axedVariables (0):
-- intAxedVariables (0):
-- mailOut.statements (0):
-- mailOut.exprOriginMap (0):
-- sameIterationInternalMail.statements (0):
-- sameIterationInternalMail.exprOriginMap (0):
-- overallHashMemory.encodedMap (0 keys):
-- overallHashMemory.normalizedEncodedKeys (0 keys, set):
-- overallHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- overallHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.admissionMapIntegration (0):
-- overallHashMemory.admissionSetIntegration (0):
-- overallHashMemory.admissionMapIntegrationDouble (0):
-- overallHashMemory.triggersForAdmissionSetIntegration (0):
-- overallHashMemory.rejectedMap (0):
-- overallHashMemory.rejectedMapIntegration (0):
-- overallHashMemory.rejectedMapOrdis (0):
-- overallHashMemory.rejectedMapOrdis2 (0):
-- overallHashMemory.admissionMapOrdis2 (0):
-- overallHashMemory.rejectedMapSubproof (0):
-- overallHashMemory.varsInRejectedMapIntegrationKeys (0):
-- overallHashMemory.varsInAdmissionMapKeys (0):
-- overallHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- overallHashMemory.admissionStatusMap (0):
-- overallHashMemory.productsOfRecursion (0):
-- overallHashMemory.productsOfRecursionIds (0):
-- overallHashMemory.revisitInProgress (0):
-- overallHashMemory.maxKeyLength=0
-- localHashMemory.encodedMap (0 keys):
-- localHashMemory.normalizedEncodedKeys (0 keys, set):
-- localHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemory.admissionMap (0):
-- localHashMemory.admissionMapIntegration (0):
-- localHashMemory.admissionSetIntegration (0):
-- localHashMemory.admissionMapIntegrationDouble (0):
-- localHashMemory.triggersForAdmissionSetIntegration (0):
-- localHashMemory.rejectedMap (0):
-- localHashMemory.rejectedMapIntegration (0):
-- localHashMemory.rejectedMapOrdis (0):
-- localHashMemory.rejectedMapOrdis2 (0):
-- localHashMemory.admissionMapOrdis2 (0):
-- localHashMemory.rejectedMapSubproof (0):
-- localHashMemory.varsInRejectedMapIntegrationKeys (0):
-- localHashMemory.varsInAdmissionMapKeys (0):
-- localHashMemory.varsInAdmissionMapIntegrationKeys (0):
)GL_DUMP",
R"GL_DUMP(-- localHashMemory.admissionStatusMap (0):
-- localHashMemory.productsOfRecursion (0):
-- localHashMemory.productsOfRecursionIds (0):
-- localHashMemory.revisitInProgress (0):
-- localHashMemory.maxKeyLength=0
-- localHashMemoryDelta.encodedMap (0 keys):
-- localHashMemoryDelta.normalizedEncodedKeys (0 keys, set):
-- localHashMemoryDelta.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemoryDelta.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemoryDelta.admissionMap (0):
-- localHashMemoryDelta.admissionMapIntegration (0):
-- localHashMemoryDelta.admissionSetIntegration (0):
-- localHashMemoryDelta.admissionMapIntegrationDouble (0):
-- localHashMemoryDelta.triggersForAdmissionSetIntegration (0):
-- localHashMemoryDelta.rejectedMap (0):
-- localHashMemoryDelta.rejectedMapIntegration (0):
-- localHashMemoryDelta.rejectedMapOrdis (0):
-- localHashMemoryDelta.rejectedMapOrdis2 (0):
-- localHashMemoryDelta.admissionMapOrdis2 (0):
-- localHashMemoryDelta.rejectedMapSubproof (0):
-- localHashMemoryDelta.varsInRejectedMapIntegrationKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemoryDelta.admissionStatusMap (0):
-- localHashMemoryDelta.productsOfRecursion (0):
-- localHashMemoryDelta.productsOfRecursionIds (0):
-- localHashMemoryDelta.revisitInProgress (0):
-- localHashMemoryDelta.maxKeyLength=0
-- workingMemory.encodedMap (0 keys):
-- workingMemory.normalizedEncodedKeys (0 keys, set):
-- workingMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- workingMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- workingMemory.admissionMap (0):
-- workingMemory.admissionMapIntegration (0):
-- workingMemory.admissionSetIntegration (0):
-- workingMemory.admissionMapIntegrationDouble (0):
-- workingMemory.triggersForAdmissionSetIntegration (0):
-- workingMemory.rejectedMap (0):
-- workingMemory.rejectedMapIntegration (0):
-- workingMemory.rejectedMapOrdis (0):
-- workingMemory.rejectedMapOrdis2 (0):
-- workingMemory.admissionMapOrdis2 (0):
-- workingMemory.rejectedMapSubproof (0):
-- workingMemory.varsInRejectedMapIntegrationKeys (0):
-- workingMemory.varsInAdmissionMapKeys (0):
-- workingMemory.varsInAdmissionMapIntegrationKeys (0):
-- workingMemory.admissionStatusMap (0):
-- workingMemory.productsOfRecursion (0):
-- workingMemory.productsOfRecursionIds (0):
-- workingMemory.revisitInProgress (0):
-- workingMemory.maxKeyLength=0
-- externalStatements (0):
-- intExternalStatements (0 — lockstep int mirror of externalStatements)

=== HASHBURST EXIT #2 | stmts=2 | origins=0 | hashMem=0 | toBeProved=1 ===
-- LB chain (innermost -> root):
  [0] (fixture[u_1])
  [1] 
-- encodedStatements:
  [0] (p[a]) | v=main
  [1] !(q[a]) | v=main
-- toBeProved (1):
  (q[a]) | v=main | auxies={0} | tags={}
-- exprOriginMap:
-- overallHashMemory.originals (0):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.encodedMap marker entries (0):
-- mailIn.statements (0):
-- mailIn.exprOriginMap (0):
-- counters: startInt=9 startIntRepl=0 startIntPi=0 level=0 isActive=1 isPartOfRecursion=0 deltaNumberStatements=0 recursionCounter=0 contradictionIndex=-1 primedForContradiction=0
-- nameMap.idToName (8 entries; nextId=8):
  [0] 
  [1] main
  [2] p
  [3] (p[a])
  [4] a
  [5] (q[a])
  [6] q
  [7] !(q[a])
-- nameMap.idToSub (1 entries; nextSubId=1):
  [0] 
-- nameMap.stackOfValidity (8):
  [0] {}
  [1] {}
  [2] {}
  [3] {}
  [4] {}
  [5] {}
  [6] {}
  [7] {}
-- nameMap.ancestorsOf (8):
  [0] {}
  [1] {1}
  [2] {2}
  [3] {3}
  [4] {4}
  [5] {5}
  [6] {6}
  [7] {7}
-- nameMap.pairMap (0 entries; size only)
-- statementLevelsMap (0):
-- localEncodedStatements (0):
-- localEncodedStatementsDelta (0):
-- intKnownStatements (0 packed (origId,validityId) keys):
-- equivalenceClassesMap (0):
-- integrationPrepared (0):
-- integrationPreparedMarker (0):
-- integrationStartIntMap (0):
-- expandedImplications (0):
-- weakVariables (0):
-- orAdmissionSet (0):
-- orBookkeeping (0):
-- orDisjunctCount (0):
-- orPendingBranches (0):
-- intValidityNam)GL_DUMP",
R"GL_DUMP(esToFilter (0):
-- pendingWipeScopes (0):
-- axedVariables (0):
-- intAxedVariables (0):
-- mailOut.statements (0):
-- mailOut.exprOriginMap (0):
-- sameIterationInternalMail.statements (0):
-- sameIterationInternalMail.exprOriginMap (0):
-- overallHashMemory.encodedMap (0 keys):
-- overallHashMemory.normalizedEncodedKeys (0 keys, set):
-- overallHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- overallHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- overallHashMemory.admissionMap (0):
-- overallHashMemory.admissionMapIntegration (0):
-- overallHashMemory.admissionSetIntegration (0):
-- overallHashMemory.admissionMapIntegrationDouble (0):
-- overallHashMemory.triggersForAdmissionSetIntegration (0):
-- overallHashMemory.rejectedMap (0):
-- overallHashMemory.rejectedMapIntegration (0):
-- overallHashMemory.rejectedMapOrdis (0):
-- overallHashMemory.rejectedMapOrdis2 (0):
-- overallHashMemory.admissionMapOrdis2 (0):
-- overallHashMemory.rejectedMapSubproof (0):
-- overallHashMemory.varsInRejectedMapIntegrationKeys (0):
-- overallHashMemory.varsInAdmissionMapKeys (0):
-- overallHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- overallHashMemory.admissionStatusMap (0):
-- overallHashMemory.productsOfRecursion (0):
-- overallHashMemory.productsOfRecursionIds (0):
-- overallHashMemory.revisitInProgress (0):
-- overallHashMemory.maxKeyLength=0
-- localHashMemory.encodedMap (0 keys):
-- localHashMemory.normalizedEncodedKeys (0 keys, set):
-- localHashMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemory.admissionMap (0):
-- localHashMemory.admissionMapIntegration (0):
-- localHashMemory.admissionSetIntegration (0):
-- localHashMemory.admissionMapIntegrationDouble (0):
-- localHashMemory.triggersForAdmissionSetIntegration (0):
-- localHashMemory.rejectedMap (0):
-- localHashMemory.rejectedMapIntegration (0):
-- localHashMemory.rejectedMapOrdis (0):
-- localHashMemory.rejectedMapOrdis2 (0):
-- localHashMemory.admissionMapOrdis2 (0):
-- localHashMemory.rejectedMapSubproof (0):
-- localHashMemory.varsInRejectedMapIntegrationKeys (0):
-- localHashMemory.varsInAdmissionMapKeys (0):
-- localHashMemory.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemory.admissionStatusMap (0):
-- localHashMemory.productsOfRecursion (0):
-- localHashMemory.productsOfRecursionIds (0):
-- localHashMemory.revisitInProgress (0):
-- localHashMemory.maxKeyLength=0
-- localHashMemoryDelta.encodedMap (0 keys):
-- localHashMemoryDelta.normalizedEncodedKeys (0 keys, set):
-- localHashMemoryDelta.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- localHashMemoryDelta.remainingArgsNormalizedEncodedMap (0 key-sets):
-- localHashMemoryDelta.admissionMap (0):
-- localHashMemoryDelta.admissionMapIntegration (0):
-- localHashMemoryDelta.admissionSetIntegration (0):
-- localHashMemoryDelta.admissionMapIntegrationDouble (0):
-- localHashMemoryDelta.triggersForAdmissionSetIntegration (0):
-- localHashMemoryDelta.rejectedMap (0):
-- localHashMemoryDelta.rejectedMapIntegration (0):
-- localHashMemoryDelta.rejectedMapOrdis (0):
-- localHashMemoryDelta.rejectedMapOrdis2 (0):
-- localHashMemoryDelta.admissionMapOrdis2 (0):
-- localHashMemoryDelta.rejectedMapSubproof (0):
-- localHashMemoryDelta.varsInRejectedMapIntegrationKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapKeys (0):
-- localHashMemoryDelta.varsInAdmissionMapIntegrationKeys (0):
-- localHashMemoryDelta.admissionStatusMap (0):
-- localHashMemoryDelta.productsOfRecursion (0):
-- localHashMemoryDelta.productsOfRecursionIds (0):
-- localHashMemoryDelta.revisitInProgress (0):
-- localHashMemoryDelta.maxKeyLength=0
-- workingMemory.encodedMap (0 keys):
-- workingMemory.normalizedEncodedKeys (0 keys, set):
-- workingMemory.normalizedEncodedSubkeys (0 keys, u_-signature record):
-- workingMemory.remainingArgsNormalizedEncodedMap (0 key-sets):
-- workingMemory.admissionMap (0):
-- workingMemory.admissionMapIntegration (0):
-- workingMemory.admissionSetIntegration (0):
-- workingMemory.admissionMapIntegrationDouble (0):
-- workingMemory.triggersForAdmissionSetIntegration (0):
-- workingMemory.rejectedMap)GL_DUMP",
R"GL_DUMP( (0):
-- workingMemory.rejectedMapIntegration (0):
-- workingMemory.rejectedMapOrdis (0):
-- workingMemory.rejectedMapOrdis2 (0):
-- workingMemory.admissionMapOrdis2 (0):
-- workingMemory.rejectedMapSubproof (0):
-- workingMemory.varsInRejectedMapIntegrationKeys (0):
-- workingMemory.varsInAdmissionMapKeys (0):
-- workingMemory.varsInAdmissionMapIntegrationKeys (0):
-- workingMemory.admissionStatusMap (0):
-- workingMemory.productsOfRecursion (0):
-- workingMemory.productsOfRecursionIds (0):
-- workingMemory.revisitInProgress (0):
-- workingMemory.maxKeyLength=0
-- externalStatements (0):
-- intExternalStatements (0 — lockstep int mirror of externalStatements)
-- phase2.statementUpload (0 uploads, 0 sent, 0 withheld by the premise patterns):
)GL_DUMP"
};
}}
