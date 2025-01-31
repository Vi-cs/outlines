import typing
from collections import ChainMap, defaultdict
from copy import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Optional,
    Set,
    Tuple,
    List
)

import interegular
import regex
from interegular.fsm import FSM, anything_else
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.exceptions import (
    LexError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
)
from lark.indenter import PythonIndenter
from lark.lexer import BasicLexer, ContextualLexer, LexerState, Scanner
from lark.parsers.lalr_analysis import Shift
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import ParseConf, ParserState
from lark.utils import get_regexp_width

from outlines.params import Params

if TYPE_CHECKING:
    from lark.lexer import LexerThread

PartialParseState = Tuple[str, int]


class PartialTokenEOF(UnexpectedEOF):
    pass


class PartialScanner(Scanner):
    def __init__(self, scanner: Scanner):
        self.terminals = scanner.terminals
        self.g_regex_flags = scanner.g_regex_flags
        self.re_ = regex
        self.use_bytes = scanner.use_bytes
        self.match_whole = scanner.match_whole
        self.allowed_types = scanner.allowed_types
        self._mres = scanner._mres

    def match(self, text, pos) -> Optional[Tuple[str, Optional[str], bool]]:
        for mre in self._mres:
            m = mre.match(text, pos=pos, partial=True)
            if m:  # and ((not m.partial) or m.endpos == len(text)):
                return m.group(0), m.lastgroup, m.partial
        return None


class PartialBasicLexer(BasicLexer):
    def __init__(self, basic_lexer: BasicLexer):
        self.re = regex
        self.newline_types = basic_lexer.newline_types
        self.ignore_types = basic_lexer.ignore_types
        self.terminals = basic_lexer.terminals
        self.user_callbacks = basic_lexer.user_callbacks
        self.g_regex_flags = basic_lexer.g_regex_flags
        self.use_bytes = basic_lexer.use_bytes
        self.terminals_by_name = basic_lexer.terminals_by_name
        self.callback = getattr(basic_lexer, "callback", None)

        if basic_lexer._scanner is not None:
            self._scanner: Optional[PartialScanner] = PartialScanner(
                basic_lexer._scanner
            )
        else:
            self._scanner = None

        # This is used to determine the token type for partial matches
        self.terminal_to_regex = {}
        for name, terminal in self.terminals_by_name.items():
            self.terminal_to_regex[name] = self.re.compile(
                terminal.pattern.to_regexp(), self.g_regex_flags
            )

    def _build_scanner(self):
        super()._build_scanner()
        self._scanner = PartialScanner(self._scanner)

    def partial_matches(self, value, type_):
        partial_matches = set()

        # TODO: It's unfortunate that we have to do this costly search (again).
        # It would be better if we could *not* short-circuit the first time we
        # scan in the call to `self.match`.
        for term_name, term_regex in self.terminal_to_regex.items():
            if term_name == type_:
                # A standard lexed token result could actual indicate a partial
                # match
                regex_min, regex_max = get_regexp_width(term_regex.pattern)
                if regex_min <= len(value) < regex_max:
                    partial_matches.add(term_name)
            else:
                m = term_regex.match(value, partial=True)
                if m:
                    partial_matches.add(term_name)

        return partial_matches

    def next_token(self, lex_state: LexerState, parser_state: Any = None) -> Token:
        line_ctr = lex_state.line_ctr
        while line_ctr.char_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos)

            if not res:
                allowed = self.scanner.allowed_types - self.ignore_types
                if not allowed:
                    allowed = {"<END-OF-FILE>"}
                raise UnexpectedCharacters(
                    lex_state.text,
                    line_ctr.char_pos,
                    line_ctr.line,
                    line_ctr.column,
                    allowed=allowed,
                    token_history=lex_state.last_token and [lex_state.last_token],
                    state=parser_state,
                    terminals_by_name=self.terminals_by_name,
                )

            value, type_, partial = res

            # Don't advance the lexing state if we're at the end; there could
            # be ambiguous token types that aren't finished.
            if line_ctr.char_pos + len(value) >= len(lex_state.text):
                partial_matches = self.partial_matches(value, type_)
                if partial_matches or partial:
                    raise PartialTokenEOF(partial_matches)

            assert isinstance(self.callback, Dict)

            if type_ not in self.ignore_types:
                t = Token(
                    type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                line_ctr.feed(value, type_ in self.newline_types)
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.callback:
                    t = self.callback[t.type](t)
                    if not isinstance(t, Token):
                        raise LexError(
                            "Callbacks must return a token (returned %r)" % t
                        )
                lex_state.last_token = t
                return t

            if type_ in self.callback:
                t2 = Token(
                    type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                self.callback[type_](t2)

            line_ctr.feed(value, type_ in self.newline_types)

        raise EOFError(self)


class PartialPythonIndenter(PythonIndenter):
    """An `Indenter` that doesn't reset its state every time `process` is called."""

    def process(self, stream):
        return self._process(stream)

    def _process(self, stream):
        for token in stream:
            # These were previously *after* the `yield`, but that makes the
            # state tracking unnecessarily convoluted.
            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                if self.paren_level < 0:
                    raise UnexpectedToken(token, [])

            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

        # while len(self.indent_level) > 1:
        #     self.indent_level.pop()
        #     yield Token(self.DEDENT_type, "")

    def __copy__(self):
        res = type(self)()
        res.paren_level = self.paren_level
        res.indent_level = copy(self.indent_level)
        return res


def copy_lexer_thread(lexer_thread: "LexerThread") -> "LexerThread":
    res = copy(lexer_thread)
    res.lexer = copy(res.lexer)

    if getattr(res.lexer, "postlexer", None):
        if isinstance(res.lexer.postlexer, PythonIndenter) and not isinstance(
                res.lexer.postlexer, PartialPythonIndenter
        ):
            # Patch these methods so that the post lexer keeps its state
            # XXX: This won't really work in generality.
            postlexer = PartialPythonIndenter()
            postlexer.paren_level = res.lexer.postlexer.paren_level
            postlexer.indent_level = res.lexer.postlexer.indent_level
            res.lexer.postlexer = postlexer
        else:
            res.lexer.postlexer = copy(res.lexer.postlexer)

    # Patch/replace the lexer objects so that they support partial matches
    context_lexer = res.lexer

    if not isinstance(context_lexer, ContextualLexer):
        # XXX: The layouts change with the grammars
        context_lexer = context_lexer.lexer
        assert isinstance(context_lexer, ContextualLexer)

    if not isinstance(context_lexer.root_lexer, PartialBasicLexer):
        context_lexer.root_lexer = PartialBasicLexer(context_lexer.root_lexer)

        basic_lexers = context_lexer.lexers
        for idx, lexer in basic_lexers.items():
            basic_lexers[idx] = PartialBasicLexer(lexer)

    return res


def copy_parser_state(parser_state: ParserState) -> ParserState:
    res = copy(parser_state)
    res.lexer = copy_lexer_thread(res.lexer)

    return res


def copy_ip(ip: "InteractiveParser") -> "InteractiveParser":
    res = copy(ip)
    res.lexer_thread = copy_lexer_thread(res.lexer_thread)
    return res


def parse_to_end(parser_state: ParserState) -> Tuple[ParserState, Set[str]]:
    """Continue parsing from the current parse state and return partial next tokens.

    .. warning::
        The parse state `parser_state` is updated in-place and must be patched
        to work with this function.  Either patch it manually or use
        `copy_parser_state` before calling this.

    """

    expected_next_tokens: Set[str] = set()
    try:
        for token in parser_state.lexer.lex(parser_state):
            parser_state.feed_token(token)
    except PartialTokenEOF as e:
        expected_next_tokens = e.expected

    return parser_state, expected_next_tokens


def find_partial_matches(
        fsm: FSM, input_strings: List[str], start_state: Optional[int] = None, activate_log: bool = False
) -> Set[Tuple[Optional[int], Tuple[int, ...]]]:
    """Find the states in the finite state machine `fsm` that accept `input_string`.

    This will consider all possible states in the finite state machine (FSM)
    that accept the beginning of `input_string` as starting points, unless a
    specific `start_state` is provided.

    Parameters
    ----------
    fsm
        The finite state machine.
    input_string
        The string for which we generate partial matches.
    start_state
        A single fixed starting state to consider.  For example, if this value
        is set to `fsm.initial`, it attempt to read `input_string` from the
        beginning of the FSM/regular expression.

    Returns
    -------
    A set of tuples corresponding to each valid starting state in the FSM.  The
    first element of each tuple contains either ``None`` or an integer
    indicating the position in `input_string` at which the FSM terminated.  The
    second element is the tuple of states visited during execution of the FSM
    plus the next, unvisited transition state.

    """
    if Params.verbose and activate_log:
        print('#### BEGIN find_partial_matches Input : ')
        print(
            f'fsm: too long - input_strings:{input_strings} - len(input_string[-1]):{len(input_strings[-1])} - start_state:{start_state}')

    # TODO: We could probably reuse parts of the computed paths when computing
    # results for multiple starting points.
    def _partial_match(
            trans: Dict[int, int],
            input_string: str
    ) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
        fsm_map = ChainMap({fsm.initial: trans}, fsm.map)
        state = fsm.initial
        accepted_states: Tuple[int, ...] = ()

        if Params.verbose and activate_log:
            print('#### BEGIN find_partial_matches _partial_match - Input:')
            print(f'fsm_map:toolong state:{state} accepted_states:{accepted_states}')

        for i, symbol in enumerate(input_string):
            if Params.verbose and activate_log:
                print(f'for {i}, {symbol} in enumerate({input_string}):')

                print(f'if {anything_else} in fsm.alphabet and {symbol} not in fsm.alphabet:')
            if anything_else in fsm.alphabet and symbol not in fsm.alphabet:
                symbol = anything_else

            trans_key = fsm.alphabet[symbol]
            if Params.verbose and activate_log:
                print(f'trans_key = fsm.alphabet[symbol]:{trans_key}')

            if not (state in fsm_map and trans_key in fsm_map[state]):
                if state in fsm.finals:
                    i -= 1
                    break
                return None, None

            state = fsm_map[state][trans_key]
            if Params.verbose and activate_log:
                print(f'state = fsm_map[state][trans_key]:{state}')

            accepted_states += (state,)

        terminated = state in fsm.finals
        if not terminated and state == fsm.initial:
            return None, None

        return None if not terminated else i, accepted_states

    def _is_string_available_at_that_state(
            string: str, state :int, contains_optional_space3: bool

    ):
        if string.replace(" ","")!="" and string[0]!=' ' and state!=0 and contains_optional_space3:
            if Params.verbose and activate_log:
                print(f'string.replace(" ",""):{string.replace(" ","")} and string[0]:{string[0]} and state:{state} and contains_optional_space3:{contains_optional_space3}')
                print(f'\'{string}\' not available at state {state}')
            return False
        #commented as i think this filtering have been done before
        #if string[0]==' ' and state==0 and contains_optional_space:
        #    return False;
        return True

    def _execute(
            fsm: FSM, input_string: str, start_state: Optional[int] = None, res: typing.Any = None, contains_optional_space2: bool = False
    ):



        trans_key = fsm.alphabet[input_string[0]]
        transition_maps = (
            fsm.map if start_state is None else {start_state: fsm.map[start_state]}
        )

        if Params.verbose and activate_log:
            print(f'trans_key:{trans_key} - input_string:{input_string} - start_state:{start_state} - transition_maps.items():{transition_maps.items()} - fsm.alphabet:too long')

        # we get to the state and ..
        for state, trans in transition_maps.items():
            # if the trans_key (vocab key of the first letter of the input_string) equals the first letter of the state (trans)
            # and if the string is available at that state
            if trans_key in trans and _is_string_available_at_that_state(input_string, state, contains_optional_space3=contains_optional_space2):
                if Params.verbose and activate_log:
                    print(
                        f'for state, trans in transition_maps.items(): state:{state} - trans:{trans}')

                n_matched, path = _partial_match(trans,input_string )
                if Params.verbose and activate_log:
                    print(f'n_matched:{n_matched} - path:{path}')
                if path is not None:
                    res.add((n_matched, (state,) + path,input_string))
        if Params.verbose and activate_log:
            print('Output : ')
            print(res)
            print('#### END find_partial_matches')
        return res

    res = set()
    contains_optional_space=False

    #TODO merge with _is_string_available_at_that_state
    if isinstance(input_strings, list):
        # keep the info that each input_string is a version of a token that could contains space at the beginning
        if len(input_strings) >1:
            contains_optional_space = True
        #if it is not the first token
        if not (start_state == fsm.initial or start_state is None):
            input_strings = [input_strings[-1]]
        #if it is the first token (and not the gms calculation)
        elif not start_state is None:
            input_strings = input_strings[:1]
        else:
            input_strings=input_strings
            ## we keep both is it is for vocab
    else:
        input_strings=[input_strings]

    if Params.verbose and activate_log:
        print(f'for input_strings: {input_strings}')

    for string in input_strings:

        if not(len(string) == 0 or string[0] not in fsm.alphabet):
            res = _execute(fsm, string, start_state, res, contains_optional_space2=contains_optional_space)


        if Params.verbose and activate_log:
            print(f'res for {string}: {res}')
        # if (start_state == fsm.initial or start_state is None) and \

    return res


def terminals_to_fsms(lp: Lark) -> Dict[str, FSM]:
    """Construct a ``dict`` mapping terminal symbol names to their finite state machines."""

    symbol_names_and_fsms = {}
    for terminal in lp.terminals:
        pattern = interegular.parse_pattern(terminal.pattern.to_regexp())
        # TODO: Use `pyparser.terminals[0].pattern.flags`?
        try:
            fsm = pattern.to_fsm().reduce()
        except Unsupported:
            fsm = None

        symbol_names_and_fsms[terminal.name] = fsm

    return symbol_names_and_fsms


def map_partial_states_to_vocab(
        vocabulary: Iterable[List[str]],
        terminals_to_fsms_map: Dict[str, FSM],
        partial_match_filter: Callable[
            [str, Optional[int], Tuple[int, ...], List[str]], bool
        ] = lambda *args: True,
        final_state_string: Optional[str] = None,
) -> Tuple[
    DefaultDict[PartialParseState, Set[int]], Dict[str, DefaultDict[int, Set[int]]]
]:
    """Construct a map from partial parse states to subsets of `vocabulary`.

    The subsets of `vocabulary` consist of elements that are accepted by--or
    transition to--the corresponding partial parse states.

    Parameters
    ----------
    vocabulary
        The vocabulary composed of strings.
    terminals_to_fsms_map
        Terminal symbol names mapped to FSMs, as provided by `terminals_to_fsms`.
    partial_match_filter
        A callable that determines which partial matches to keep.  The first
        argument is the string being match, the rest are the unpacked partial
        match return values of `find_partial_matches`.
    final_state_string
        A string from `vocabulary` that is to be added to all the final states
        in the FSM.
    """

    final_state_string_idx = None

    # Partial parse states to the subsets of the vocabulary that accept them
    pstate_to_vocab = defaultdict(set)
    possible_paths = {}
    log_index = 0
    # symbol_name = REGEX
    for symbol_name, fsm in terminals_to_fsms_map.items():
        if log_index == 0 and False:
            activate_log = True
        else:
            activate_log = False
        log_index = log_index + 1
        if Params.verbose and activate_log:
            print(f'for symbol_name, fsm in terminals_to_fsms_map.items(): symbol_name:{symbol_name} - fsm: too long - terminals_to_fsms_map:too long')
        terminal_possible_paths = defaultdict(set)
        # iterate on vocab string, at index i in the vocab
        for i, vocab_string in enumerate(vocabulary):
            if vocab_string[0] == final_state_string:
                final_state_string_idx = i
            if False and (i == 7559000 or i==100):
                activate_log = True
            else:
                activate_log = False
            if Params.verbose and activate_log:
                print(
                    f'-- for i, vocab_string in enumerate(vocabulary): i:{i} - vocab_string:{vocab_string} - len(vocab_string):{len(vocab_string)}')
            '''The first element of each tuple contains either ``None`` or an integer
            indicating the position in `input_string` at which the FSM terminated.  The
            second element is the tuple of states visited during execution of the FSM
            plus the next, unvisited transition state.'''
            for end_idx, state_seq, string in find_partial_matches(fsm=fsm, input_strings=vocab_string, activate_log=activate_log):
                if Params.verbose and activate_log:
                    print(
                        f'---- for end_idx, state_seq in find_partial_matches(fsm, vocab_string): end_idx:{end_idx} - state_seq:{state_seq}')
                    print(
                        f'---- if partial_match_filter(vocab_string, end_idx, state_seq): {partial_match_filter(string, end_idx, state_seq)}')
                ''' def partial_match_filter(string, end_idx, state_seq):
                    if end_idx is not None and end_idx < len(string) - 1:
                        return False
                    return True'''
                if partial_match_filter(string, end_idx, state_seq, vocab_string):
                    terminal_possible_paths[state_seq[0]].add(state_seq[-1])
                    if Params.verbose and activate_log:
                        print(
                            f'------ terminal_possible_paths[state_seq[0]].add(state_seq[-1]): state_seq[0]:{state_seq[0]} - state_seq[-1]:{state_seq[-1]}')
                    pstate_to_vocab[(symbol_name, state_seq[0])].add(i)
                    if Params.verbose and activate_log:
                        print(f'------ pstate_to_vocab[(symbol_name, state_seq[0])].add(i):')

        possible_paths[symbol_name] = terminal_possible_paths

    if final_state_string_idx is not None:
        # Allow transitions to EOS from all terminals FSM states
        if Params.verbose:
            print(f'terminals_to_fsms_map.items():too long')
        for symbol_name, fsm in terminals_to_fsms_map.items():
            print(f'for symbol_name:{symbol_name}, fsm: too long, fsm.finals: too long')
            for state in fsm.finals:
                print(f'state:{state}')
                pstate_to_vocab[(symbol_name, state)].add(final_state_string_idx)
    # print('Output : ')
    # print(pstate_to_vocab)
    # print(possible_paths)

    if Params.verbose:
        print('#### END map_partial_states_to_vocab. Output:')
        print(f'pstate_to_vocab:{pstate_to_vocab}, possible_paths:{possible_paths}')
    return pstate_to_vocab, possible_paths


def terminals_to_lalr_states(lp: Lark) -> DefaultDict[str, Set[int]]:
    terminals_to_states = defaultdict(set)
    parse_table = lp.parser.parser.parser.parse_table
    for state, tokens_to_ops in parse_table.states.items():
        for token, op in tokens_to_ops.items():
            if op[0] == Shift:
                # `op[1]` is the state we shift to when `token` is observed
                terminals_to_states[token].add(op[1])

    return terminals_to_states


def create_pmatch_parser_states(
        lp: Lark,
        terminals_to_states: Dict[str, Set[int]],
        term_type: str,
        ptoken: str,
        pmatch: Tuple[int, Tuple[int, ...]],
) -> Tuple[ParserState, ...]:
    parse_table = lp.parser.parser.parser.parse_table

    # TODO: We need to effectively disable the callbacks that build the
    # trees, because we aren't actually parsing a valid state that can, say,
    # be reduced
    def noop(*args, **kwargs):
        pass

    callbacks = {rule: noop for rule, cb in lp._callbacks.items()}
    parse_conf = ParseConf(parse_table, callbacks, lp.options.start[0])
    lexer_thread = lp.parser._make_lexer_thread(ptoken)
    lexer_state = lexer_thread.state
    lexer_state.line_ctr.char_pos = pmatch[0] + 1
    lexer_state.last_token = Token(term_type, "")
    res = tuple(
        ParserState(parse_conf, lexer_thread, [state], None)
        for state in terminals_to_states[term_type]
    )
    return res
