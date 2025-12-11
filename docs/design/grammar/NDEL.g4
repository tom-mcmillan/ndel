/*
 * NDEL Grammar Specification
 * Version: 0.1.0
 * 
 * This ANTLR4 grammar defines the Non-Deterministic Expression Language (NDEL).
 * NDEL combines deterministic expression structure with non-deterministic value resolution.
 */

grammar NDEL;

// =============================================================================
// PARSER RULES
// =============================================================================

// Entry point
program
    : statement* EOF
    ;

statement
    : domainDeclaration
    | expression ';'?
    ;

// Domain declaration
domainDeclaration
    : '@domain' '(' STRING_LITERAL ')'
    ;

// Expressions
expression
    : conditionalExpression
    ;

conditionalExpression
    : logicalOrExpression ('?' expression ':' expression)?
    ;

logicalOrExpression
    : logicalAndExpression ('||' logicalAndExpression)*
    ;

logicalAndExpression
    : equalityExpression ('&&' equalityExpression)*
    ;

equalityExpression
    : relationalExpression (('==' | '!=') relationalExpression)*
    ;

relationalExpression
    : additiveExpression (('<' | '<=' | '>' | '>=') additiveExpression)*
    ;

additiveExpression
    : multiplicativeExpression (('+' | '-') multiplicativeExpression)*
    ;

multiplicativeExpression
    : unaryExpression (('*' | '/' | '%') unaryExpression)*
    ;

unaryExpression
    : ('!' | '-' | '+')? fuzzyExpression
    ;

fuzzyExpression
    : postfixExpression (fuzzyOperator fuzzyValue)?
    ;

fuzzyOperator
    : 'is'
    | 'shows'
    | 'approximately'
    | 'roughly'
    ;

fuzzyValue
    : STRING_LITERAL
    | FUZZY_STRING
    ;

postfixExpression
    : primaryExpression (
        '.' IDENTIFIER                           // Member access
        | '[' expression ']'                     // Index access
        | '(' argumentList? ')'                  // Function call
    )*
    ;

primaryExpression
    : literal
    | IDENTIFIER
    | '(' expression ')'
    | listLiteral
    | mapLiteral
    | structLiteral
    | confidenceExpression
    | 'has' '(' expression '.' IDENTIFIER ')'
    ;

literal
    : NUMBER_LITERAL
    | STRING_LITERAL
    | BOOLEAN_LITERAL
    | NULL_LITERAL
    ;

listLiteral
    : '[' (expression (',' expression)*)? ']'
    ;

mapLiteral
    : '{' (mapEntry (',' mapEntry)*)? '}'
    ;

mapEntry
    : (IDENTIFIER | STRING_LITERAL) ':' expression
    ;

structLiteral
    : IDENTIFIER '{' (structField (',' structField)*)? '}'
    ;

structField
    : IDENTIFIER ':' expression
    ;

confidenceExpression
    : 'confidence' '(' ')'
    | 'with_confidence' '(' NUMBER_LITERAL ',' expression ')'
    | 'alternatives' '(' ')'
    ;

argumentList
    : expression (',' expression)*
    ;

// =============================================================================
// LEXER RULES
// =============================================================================

// Keywords
DOMAIN      : '@domain';
IS          : 'is';
SHOWS       : 'shows';
APPROXIMATELY : 'approximately';
ROUGHLY     : 'roughly';
HAS         : 'has';
IN          : 'in';
IF          : 'if';
THEN        : 'then';
ELSE        : 'else';
AND         : 'and';
OR          : 'or';
NOT         : 'not';
TRUE        : 'true';
FALSE       : 'false';
NULL        : 'null';
CONFIDENCE  : 'confidence';
WITH_CONFIDENCE : 'with_confidence';
ALTERNATIVES : 'alternatives';

// Operators
PLUS        : '+';
MINUS       : '-';
MULTIPLY    : '*';
DIVIDE      : '/';
MODULO      : '%';
EQUAL       : '==';
NOT_EQUAL   : '!=';
LESS        : '<';
LESS_EQUAL  : '<=';
GREATER     : '>';
GREATER_EQUAL : '>=';
LOGICAL_AND : '&&';
LOGICAL_OR  : '||';
LOGICAL_NOT : '!';
QUESTION    : '?';
COLON       : ':';
DOT         : '.';
COMMA       : ',';
SEMICOLON   : ';';
LPAREN      : '(';
RPAREN      : ')';
LBRACKET    : '[';
RBRACKET    : ']';
LBRACE      : '{';
RBRACE      : '}';

// Literals
BOOLEAN_LITERAL
    : TRUE
    | FALSE
    ;

NULL_LITERAL
    : NULL
    ;

NUMBER_LITERAL
    : INT_LITERAL
    | FLOAT_LITERAL
    ;

INT_LITERAL
    : DECIMAL_LITERAL
    | HEX_LITERAL
    | OCTAL_LITERAL
    | BINARY_LITERAL
    ;

DECIMAL_LITERAL
    : '0'
    | [1-9] DIGIT*
    ;

HEX_LITERAL
    : '0' [xX] HEX_DIGIT+
    ;

OCTAL_LITERAL
    : '0' [oO] OCTAL_DIGIT+
    ;

BINARY_LITERAL
    : '0' [bB] BINARY_DIGIT+
    ;

FLOAT_LITERAL
    : DIGIT+ '.' DIGIT* EXPONENT?
    | '.' DIGIT+ EXPONENT?
    | DIGIT+ EXPONENT
    ;

STRING_LITERAL
    : '"' (ESC_SEQUENCE | ~["\\])* '"'
    | '\'' (ESC_SEQUENCE | ~['\\])* '\''
    ;

// Special fuzzy string literal (could have special handling)
FUZZY_STRING
    : 'fuzzy' STRING_LITERAL
    ;

// Identifiers
IDENTIFIER
    : LETTER (LETTER | DIGIT | '_')*
    ;

// Comments
SINGLE_LINE_COMMENT
    : '//' ~[\r\n]* -> skip
    ;

MULTI_LINE_COMMENT
    : '/*' .*? '*/' -> skip
    ;

// Whitespace
WS
    : [ \t\r\n]+ -> skip
    ;

// =============================================================================
// FRAGMENTS (Lexer helpers)
// =============================================================================

fragment DIGIT
    : [0-9]
    ;

fragment HEX_DIGIT
    : [0-9a-fA-F]
    ;

fragment OCTAL_DIGIT
    : [0-7]
    ;

fragment BINARY_DIGIT
    : [01]
    ;

fragment LETTER
    : [a-zA-Z]
    | [\u0080-\uFFFF]  // Unicode support
    ;

fragment EXPONENT
    : [eE] [+-]? DIGIT+
    ;

fragment ESC_SEQUENCE
    : '\\' [btnfr"'\\]
    | '\\' 'u' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT
    | '\\' 'U' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT
    | '\\' [0-3] OCTAL_DIGIT OCTAL_DIGIT
    | '\\' OCTAL_DIGIT OCTAL_DIGIT?
    ;

// =============================================================================
// ERROR HANDLING
// =============================================================================

// Catch-all for unrecognized characters
ERROR_CHAR
    : .
    ;
