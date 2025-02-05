// Emscripten adds a global "Module" object when "algo.js" loads.
// We'll wait until it's ready to call any compiled C functions.

Module.onRuntimeInitialized = function() {
  // 1) Wrap the C function "prob" using cwrap.
  //    'prob' returns a double, and takes (unsigned int*, double*, int) as parameters.
  const c_prob = Module.cwrap('prob', 'number', ['number', 'number', 'number']);
  const c_explen = Module.cwrap('explen', 'number', ['number', 'number', 'number']);
  /**
   * High-level JS helper: calls the C 'prob' function.
   * @param {number[]} graphArr - Array of positive integers (unsigned int in C)
   * @param {number[]} psArr    - Array of floats (double in C)
   * @param {number} [startIndex=2] - Start node index
   * @returns {number} Probability
   */
  function probability(graphArr, psArr, startIndex = 2) {
    // Convert JS arrays to typed arrays
    const graphTyped = new Uint32Array(graphArr);
    const psTyped    = new Float64Array(psArr);

    // Allocate memory in the WASM heap
    const graphBytes = graphTyped.length * graphTyped.BYTES_PER_ELEMENT;
    const graphPtr   = Module._malloc(graphBytes);
    Module.HEAPU32.set(graphTyped, graphPtr >> 2);

    const psBytes = psTyped.length * psTyped.BYTES_PER_ELEMENT;
    const psPtr   = Module._malloc(psBytes);
    Module.HEAPF64.set(psTyped, psPtr >> 3);

    // Call the C function
    const out = c_prob(graphPtr, psPtr, startIndex);

    // Free memory
    Module._free(graphPtr);
    Module._free(psPtr);

    return out;
  }  
  function expected_length(graphArr, psArr, startIndex = 2) {
    // Convert JS arrays to typed arrays
    const graphTyped = new Uint32Array(graphArr);
    const psTyped    = new Float64Array(psArr);

    // Allocate memory in the WASM heap
    const graphBytes = graphTyped.length * graphTyped.BYTES_PER_ELEMENT;
    const graphPtr   = Module._malloc(graphBytes);
    Module.HEAPU32.set(graphTyped, graphPtr >> 2);

    const psBytes = psTyped.length * psTyped.BYTES_PER_ELEMENT;
    const psPtr   = Module._malloc(psBytes);
    Module.HEAPF64.set(psTyped, psPtr >> 3);

    // Call the C function
    const out = c_explen(graphPtr, psPtr, startIndex);

    // Free memory
    Module._free(graphPtr);
    Module._free(psPtr);

    return out;
  }
  const N = 1000;
  const DELTA = 0.001;
  const ps = Array.from({ length: N }, (_, i) => i / N);

  const ctx = document.getElementById('p_chart');
  let pChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ps, // X-Axis labels
      datasets: [] // Initially empty, datasets will be added dynamically
    },
    options: {
        responsive: true,
        scales: {
            x: { beginAtZero: true },
            y: { beginAtZero: true }
        },
        
    }
  });  
  const ctx2 = document.getElementById('length_fairness_chart');

  const x_ds = Array.from({ length: N }, (_, i) => 15 * (i / N));
  const dataPoints = x_ds.map(x => ({ x: x, y: x * x }));
  let lenFairnessChart = new Chart(ctx2, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Optimality threshold',
        data: dataPoints,
        showLine: true,          // Draw a connecting line
        fill: false,
        borderColor: 'blue',
        backgroundColor: 'blue',
        pointRadius: 0,
      },]
    },
    
    options: {
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          min: 0,
          max: 15,
          title: {
            display: true,
            text: 'Fairness',
          },
        },
        y: {
          type: 'linear',
          min: 0,
          max: 225,
          title: {
            display: true,
            text: 'Expected length',
          },
        },
      },
        
    }
  });
  // 2) When the button is clicked, read user input, call `probability`, show the result
  document.getElementById('run').addEventListener('click', function() {

    // Get the user input strings
    const userCode = document.getElementById('code').value; // e.g. "3,5,10,42,99"
  
    const graphData = __BRYTHON__.imported.runner.run_code(userCode);

    // Optionally, choose a start node. We'll just do 2 for example:
    const startNode = 2;

    // Call the Wasm function
    const vs = ps.map(p => probability(graphData, [p], 2));


    const dvs = [0.5, 0.5 + DELTA].map(p => probability(graphData, [p], 2));
    const fairness = (dvs[1] - dvs[0]) / DELTA;
    const el = expected_length(graphData, [0.5], 2);
    document.getElementById('output').textContent = `Fairness (Derivative P(p) with p = 0.5): ${fairness.toFixed(2)}\nExpected length (E[length of the match] with p = 0.5): ${el.toFixed(2)}`;

    const dataset = {
        label: 'P(p)',
        data: vs,
        pointRadius: 0,
    };
    if (pChart.data.datasets.length > 0) {
      pChart.data.datasets.pop();
    }
    pChart.data.datasets.push(dataset);
    pChart.update();

    if (lenFairnessChart.data.datasets.length > 1) {
      lenFairnessChart.data.datasets.pop();
    }
    lenFairnessChart.data.datasets.push({
        data: [{
            x: fairness,
            y: el,
        }],
        fill: true,
        borderColor: 'red',
        backgroundColor: 'red',
        label: 'your match',
    })
    lenFairnessChart.update();
    console.log(lenFairnessChart.data.datasets);
    // Display the result
  });





};