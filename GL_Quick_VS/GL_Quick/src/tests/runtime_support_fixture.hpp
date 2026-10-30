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
#include "../logical_block/canonical_form_cache.hpp"
#include "../logical_block/collected_arena.hpp"
#include "../logical_block/new_variable_store.hpp"
#include "../logical_block/rejection_store.hpp"
#include "../logical_block/scanned_token_runs.hpp"
#include "../logical_block/witness_meta_store.hpp"
namespace gl::tests {
struct RuntimeSupportFixture {
template<class T> static decltype(auto) algebraKeys(T& value) { return (value.algebraKeys); }
template<class T> static decltype(auto) arena(T& value) { return (value.arena); }
template<class T> static decltype(auto) copies(T& value) { return (value.copies); }
template<class T> static decltype(auto) dirty(T& value) { return (value.dirty); }
template<class T> static decltype(auto) elems(T& value) { return (value.elems); }
template<class T> static decltype(auto) doubleKeys(T& value) { return (value.doubleKeys); }
template<class T> static decltype(auto) entries(T& value) { return (value.entries); }
template<class T> static decltype(auto) entryKeys(T& value) { return (value.entryKeys); }
template<class T> static decltype(auto) forceIntegration(T& value) { return (value.forceIntegration); }
template<class T> static decltype(auto) forms(T& value) { return (value.forms); }
template<class T> static decltype(auto) integrationDepth(T& value) { return (value.integrationDepth); }
template<class T> static decltype(auto) levs(T& value) { return (value.levs); }
template<class T> static decltype(auto) mb(T& value) { return (value.mb); }
template<class T> static decltype(auto) nextSlot(T& value) { return (value.nextSlot); }
template<class T> static decltype(auto) nodes(T& value) { return (value.nodes); }
template<class T> static decltype(auto) normalSource(T& value) { return (value.normalSource); }
template<class T> static decltype(auto) orSourceKeyIds(T& value) { return (value.orSourceKeyIds); }
template<class T> static decltype(auto) parkKeys(T& value) { return (value.parkKeys); }
template<class T> static decltype(auto) parks(T& value) { return (value.parks); }
template<class T> static decltype(auto) publishing(T& value) { return (value.publishing); }
template<class T> static decltype(auto) recs(T& value) { return (value.recs); }
template<class T> static decltype(auto) rules(T& value) { return (value.rules); }
template<class T> static decltype(auto) runs(T& value) { return (value.runs); }
template<class T> static decltype(auto) saved(T& value) { return (value.saved); }
template<class T> static decltype(auto) scan(T& value) { return (value.scan); }
template<class T> static decltype(auto) scopes(T& value) { return (value.scopes); }
template<class T> static decltype(auto) setKeys(T& value) { return (value.setKeys); }
template<class T> static decltype(auto) sharedNodes(T& value) { return (value.sharedNodes); }
template<class T> static decltype(auto) sibIds(T& value) { return (value.sibIds); }
template<class T> static decltype(auto) strings(T& value) { return (value.strings); }
template<class T> static decltype(auto) texts(T& value) { return (value.texts); }
template<class T> static decltype(auto) triggers(T& value) { return (value.triggers); }
template<class T> static decltype(auto) version(T& value) { return (value.version); }
};
}
