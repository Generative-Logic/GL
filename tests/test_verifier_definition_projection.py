"""Mandatory verifier-gate coverage for processed positive definition projections."""
from tests.test_harness import make_proof_line, make_state_minimal, register
from verifier import TAG_CHECKERS, _ORIGIN_EXEMPT_TAGS, check_definition_projection


def fixture(target='(in[p,N])', source='(outer[i,I,p,N])', namespace='main'):
    state = make_state_minimal()
    state.current_gl_binary = {
        'pair': {'category': 'and', 'signature': '(pair[u_1,u_2,u_3,u_4])',
                 'arity': 4, 'elements': ['(in[u_1,u_2])', '(in[u_3,u_4])']},
        'outer': {'category': 'and', 'signature': '(outer[u_1,u_2,u_3,u_4])',
                  'arity': 4, 'elements': ['(pair[u_1,u_2,u_3,u_4])', '(in[u_1,u_4])']}}
    origin = make_proof_line(source, namespace, 'task formulation')
    row = make_proof_line(target, namespace, 'definition projection', source, namespace)
    return row, [origin, row], state


@register
def registered_and_origin_checked():
    assert TAG_CHECKERS['definition projection'] is check_definition_projection
    assert 'definition projection' not in _ORIGIN_EXEMPT_TAGS


@register
def positive_named_nested_projection():
    assert check_definition_projection(*fixture()) is True


@register
def positive_explicit_conjunction():
    assert check_definition_projection(*fixture(source='(&(in[i,I])(in[p,N]))')) is True


@register
def positive_child_scope():
    assert check_definition_projection(*fixture(namespace='main_boundary_case')) is True


@register
def missing_source_rejected():
    row, chapter, state = fixture()
    assert check_definition_projection(row, [row], state) is False


@register
def foreign_scope_source_rejected():
    row, chapter, state = fixture()
    chapter[0].namespace = 'main_boundary_case'
    assert check_definition_projection(row, chapter, state) is False


@register
def cross_scope_citation_rejected():
    row, chapter, state = fixture()
    row.namespace = 'main_boundary_case'
    assert check_definition_projection(row, chapter, state) is False


@register
def malformed_dependency_pairs_rejected():
    for rest in ([], ['(outer[i,I,p,N])'],
                 ['(outer[i,I,p,N])', 'main', '(in[p,N])', 'main']):
        row, chapter, state = fixture()
        row.rest = rest
        assert check_definition_projection(row, chapter, state) is False


@register
def fixed_ports_and_polarity_rejected():
    for target in ('(in[q,N])', '(in[N,p])', '!(in[p,N])'):
        assert check_definition_projection(*fixture(target=target)) is False


@register
def identity_is_not_projection():
    assert check_definition_projection(*fixture(target='(outer[i,I,p,N])')) is False


@register
def no_elimination_through_negation_or_implication():
    for source in ('!(outer[i,I,p,N])', '(>[](in[p,I])(in[p,N]))'):
        assert check_definition_projection(*fixture(source=source)) is False


@register
def no_or_or_existential_or_implication_unfolding():
    for category in ('or', 'existence', 'implication'):
        row, chapter, state = fixture()
        state.current_gl_binary['pair']['category'] = category
        assert check_definition_projection(row, chapter, state) is False


@register
def negative_member_remains_opaque():
    row, chapter, state = fixture(target='!(in[p,N])')
    state.current_gl_binary['pair']['elements'][1] = '!(in[u_3,u_4])'
    assert check_definition_projection(row, chapter, state) is True
    row.expression = '(in[p,N])'
    assert check_definition_projection(row, chapter, state) is False


@register
def quantified_member_preserves_free_ports_and_capture():
    row, chapter, state = fixture(target='(>[z](P[z,x])(Q[x]))', source='(wrapped[x])')
    state.current_gl_binary = {'wrapped': {
        'category': 'and', 'signature': '(wrapped[u_1])', 'arity': 1,
        'elements': ['(>[x](P[x,u_1])(Q[u_1]))', '(R[u_1])']}}
    assert check_definition_projection(row, chapter, state) is True
    for target in ('(>[x](P[x,x])(Q[x]))', '(>[z](P[z,y])(Q[y]))'):
        row.expression = target
        assert check_definition_projection(row, chapter, state) is False


@register
def malformed_and_recursive_definitions_rejected():
    row, chapter, state = fixture()
    state.current_gl_binary['outer']['elements'] = ['(outer[u_1,u_2,u_3,u_4])']
    assert check_definition_projection(row, chapter, state) is False
    assert check_definition_projection(*fixture(source='(outer[i,I,p])')) is False
