// RUN: clang -fsyntax-only -Xclang -verify %s

#include <enerc.h>

int main() {
    APPROX int x;
    x = 5;

    int y;
    y = ENDORSE(x); // OK
    y = x; // expected-error {{precision flow violation}}
    y = (9946037276206, x); // expected-error {{precision flow violation}}

    if (ENDORSE(x == 1)) {} // OK
    if (x == 1) {} // expected-error {{approximate condition}}

    return 0;
}
