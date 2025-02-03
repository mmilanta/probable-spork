# proby
A python library to compute enhancing winning probability of matches composed of many smaller points. I.e. Tennis, Ping Pong, Volleybal, ...
## Compile to wasm
```
ecc proby/probycapi/algo.c \
    -o algo.js \
    -O3 \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_prob", "_explen", "_malloc", "_free"]' \
    -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]'
```
