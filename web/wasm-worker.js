// 1) Load the Emscripten-generated code:
importScripts('algo.js');

// 2) Wait for Emscripten’s runtime to initialize:
Module['onRuntimeInitialized'] = function() {

  // Wrap C functions with cwrap (return type, arg types):
  const c_prob = Module.cwrap('prob', 'number', ['number', 'number', 'number']);
  
  // Listen for messages from the main thread:
  onmessage = (e) => {
    // Expect a message of the form:
    // { requestId, graph, ps, startIndex }
    const { requestId, graph, ps, startIndex } = e.data;

    // Convert JS arrays to typed arrays if needed (depends on how you pass them from the main thread)
    const graphArray = (graph instanceof Uint32Array) ? graph : new Uint32Array(graph);
    const psArray    = (ps instanceof Float64Array)  ? ps   : new Float64Array(ps);

    // Allocate memory in the WASM heap
    const sizeGraphBytes = graphArray.length * graphArray.BYTES_PER_ELEMENT; // 4 bytes each
    const ptrGraph = Module._malloc(sizeGraphBytes);
    Module.HEAPU32.set(graphArray, ptrGraph >> 2);

    const sizePsBytes = psArray.length * psArray.BYTES_PER_ELEMENT; // 8 bytes each
    const ptrPs = Module._malloc(sizePsBytes);
    Module.HEAPF64.set(psArray, ptrPs >> 3);

    // Call the compiled C function
    // prob(unsigned int *graph, double *ps, int index)
    // returns a double
    const probability = c_prob(ptrGraph, ptrPs, startIndex);

    // Free WASM memory
    Module._free(ptrGraph);
    Module._free(ptrPs);

    // Send result back to main thread
    postMessage({
      requestId,
      probability
    });
  };
};