# Rust Notes

Ownership is the core idea: each value has a single owner, and when the owner
goes out of scope, the value is dropped.

Borrowing lets you reference data without taking ownership. References are
either shared (&T) or exclusive (&mut T), never both at once.

Lifetimes describe how long references are valid. Most of the time the
compiler infers them, but sometimes you need to annotate explicitly.
