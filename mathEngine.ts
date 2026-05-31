/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActivationType, Dataset, TokenizerVocab, NetworkState } from '../types';

// Standard activation functions and their math descriptions
export const ACTIVATIONS = {
  sigmoid: {
    func: (x: number) => 1 / (1 + Math.exp(-x)),
    deriv: (fx: number) => fx * (1 - fx),
    formula: 'f(x) = \\frac{1}{1 + e^{-x}}',
    description: 'Squashes any real-valued number into a strict probability range between 0 and 1.'
  },
  relu: {
    func: (x: number) => Math.max(0, x),
    deriv: (fx: number) => (fx > 0 ? 1 : 0),
    formula: 'f(x) = \\max(0, x)',
    description: 'Passes positive inputs directly; outputs absolute zero for all negative inputs. Solves vanishing gradients.'
  },
  tanh: {
    func: (x: number) => Math.tanh(x),
    deriv: (fx: number) => 1 - fx * fx,
    formula: 'f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}',
    description: 'Outputs zero-centered values between -1 and 1, facilitating faster convergence in deep layers.'
  }
};

/**
 * Parses raw text input into a standard dataset structured as numerical matrices.
 * Supports tokenizing unstructured sentiment texts or parsing structured tabular formats.
 */
export function parseTextDataset(rawInput: string, format: 'text' | 'tabular', targetColumnName?: string): Dataset {
  if (format === 'tabular') {
    // Basic CSV parser
    const lines = rawInput.trim().split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) {
      throw new Error('Not enough tabular data. Need at least a header row and 1 data row.');
    }

    // Split headers
    const delimiter = rawInput.includes(';') ? ';' : ',';
    const headers = lines[0].split(delimiter).map(h => h.trim().replace(/['"„“]/g, ''));
    
    // Parse records
    const rawRows: Array<Record<string, string | number>> = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(delimiter).map(p => p.trim().replace(/['"„“]/g, ''));
      if (parts.length !== headers.length) continue;
      
      const record: Record<string, string | number> = {};
      headers.forEach((header, colIdx) => {
        const val = parts[colIdx];
        const num = Number(val);
        record[header] = isNaN(num) ? val : num;
      });
      rawRows.push(record);
    }

    if (rawRows.length === 0) {
      throw new Error('Could not parse any valid entries from the CSV/tabular source.');
    }

    // Determine target column (default to last column if unspecified)
    const targetCol = targetColumnName && headers.includes(targetColumnName) 
      ? targetColumnName 
      : headers[headers.length - 1];

    const featureHeaders = headers.filter(h => h !== targetCol);

    // Prepare min/max scaling values for numerical indicators
    const featureMinMax = featureHeaders.map(fh => {
      let min = Infinity;
      let max = -Infinity;
      rawRows.forEach(row => {
        const val = Number(row[fh]);
        if (!isNaN(val)) {
          if (val < min) min = val;
          if (val > max) max = val;
        }
      });
      if (min === Infinity) { min = 0; max = 1; }
      if (min === max) { max = min + 1; } // Avoid zero division
      return { min, max };
    });

    let targetMin = Infinity;
    let targetMax = -Infinity;
    rawRows.forEach(row => {
      const val = Number(row[targetCol]);
      if (!isNaN(val)) {
        if (val < targetMin) targetMin = val;
        if (val > targetMax) targetMax = val;
      }
    });
    if (targetMin === Infinity) { targetMin = 0; targetMax = 1; }
    if (targetMin === targetMax) { targetMax = targetMin + 1; }

    // Map each row to a feature vector
    const X: number[][] = [];
    const y: number[] = [];

    rawRows.forEach(row => {
      const xVec: number[] = [];
      featureHeaders.forEach((fh, fIdx) => {
        const rawVal = row[fh];
        const numVal = Number(rawVal);
        if (!isNaN(numVal)) {
          // Continuous numeric min-max scale
          const mm = featureMinMax[fIdx];
          xVec.push((numVal - mm.min) / (mm.max - mm.min));
        } else {
          // Categorical mapping (hashing or simple index)
          const strVal = String(rawVal).toLowerCase();
          let hashSum = 0;
          for (let charIdx = 0; charIdx < strVal.length; charIdx++) {
            hashSum += strVal.charCodeAt(charIdx);
          }
          xVec.push((hashSum % 10) / 10); // Standardize categorical 0-1
        }
      });
      X.push(xVec);

      const targetVal = String(row[targetCol]).trim().toLowerCase();
      const numTarget = Number(targetVal);
      if (!isNaN(numTarget)) {
        // Continuous target normalisation
        y.push((numTarget - targetMin) / (targetMax - targetMin));
      } else if (targetVal === 'positive' || targetVal === 'yes' || targetVal === 'true' || targetVal === '1' || targetVal === 'high') {
        y.push(1);
      } else if (targetVal === 'negative' || targetVal === 'no' || targetVal === 'false' || targetVal === '0' || targetVal === 'low') {
        y.push(0);
      } else {
        // String classification fallback
        const codeSum = targetVal.length > 0 ? targetVal.charCodeAt(0) : 0;
        y.push(codeSum % 2 === 0 ? 0 : 1);
      }
    });

    const isClassification = y.every(val => val === 0 || val === 1);

    return {
      headers: featureHeaders,
      rawRows,
      X,
      y,
      featureMinMax,
      targetMinMax: { min: targetMin, max: targetMax },
      taskType: isClassification ? 'classification' : 'regression'
    };
  } else {
    // Unstructured Text Processing Format
    // Format: sentence | sentiment label (positive/negative)
    const lines = rawInput.trim().split('\n').filter(Boolean);
    const textRows: Array<{ text: string; label: string }> = [];

    lines.forEach(l => {
      const lastTab = l.lastIndexOf('\t');
      const lastComma = l.lastIndexOf(',');
      const sep = lastTab !== -1 ? lastTab : (lastComma !== -1 ? lastComma : l.lastIndexOf('|'));
      
      if (sep !== -1) {
        const text = l.slice(0, sep).trim().replace(/['"„“]/g, '');
        const label = l.slice(sep + 1).trim().replace(/['"„“]/g, '');
        textRows.push({ text, label });
      } else {
        // Automatic sentiment analyzer or split first/last words
        const words = l.split(/\s+/);
        const label = words[words.length - 1];
        const text = words.slice(0, words.length - 1).join(' ');
        textRows.push({ text, label });
      }
    });

    // Extract vocabulary dictionary
    const vocab: TokenizerVocab = {};
    let wordCounter = 0;

    textRows.forEach(row => {
      const tokens = tokenizeText(row.text);
      tokens.forEach(tok => {
        if (vocab[tok] === undefined) {
          vocab[tok] = wordCounter++;
        }
      });
    });

    const vocabSize = Math.max(1, wordCounter);
    const X: number[][] = [];
    const y: number[] = [];
    const rawRows: Array<Record<string, string | number>> = [];

    textRows.forEach((row, rIdx) => {
      const vec = new Array(vocabSize).fill(0);
      const tokens = tokenizeText(row.text);
      tokens.forEach(tok => {
        const vocabId = vocab[tok];
        if (vocabId !== undefined) {
          vec[vocabId] += 1; // Basic Bag of Words metric
        }
      });
      X.push(vec);

      const normalizedLabel = row.label.toLowerCase();
      const isPositive = normalizedLabel.includes('posit') || normalizedLabel.includes('good') || normalizedLabel.includes('yes') || normalizedLabel.includes('1') || normalizedLabel.includes('true') || normalizedLabel.includes('рад') || normalizedLabel.includes('класс') || normalizedLabel.includes('да');
      y.push(isPositive ? 1 : 0);

      rawRows.push({
        'Text Input': row.text,
        'Original Label': row.label,
        'Target Vector': isPositive ? 1 : 0
      });
    });

    // Make clean vocabulary headers
    const headers = new Array(vocabSize);
    Object.entries(vocab).forEach(([word, id]) => {
      headers[id] = `"${word}" frequency`;
    });

    return {
      headers,
      rawRows,
      X,
      y,
      featureMinMax: headers.map(() => ({ min: 0, max: 10 })),
      targetMinMax: { min: 0, max: 1 },
      vocab,
      taskType: 'classification'
    };
  }
}

export function tokenizeText(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"“”']/g, '')
    .split(/\s+/)
    .filter(Boolean);
}

/**
 * Initializes model weights and biases securely based on layer specifications.
 * He / Xavier standard initialization prevents neuron saturation.
 */
export function initializeWeights(layerSizes: number[]): NetworkState {
  const weights: number[][][] = []; // indices: [layerX][toNeuron][fromNeuron]
  const biases: number[][] = [];    // indices: [layerX][toNeuron]

  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fromSize = layerSizes[l];
    const toSize = layerSizes[l + 1];

    // Initialize Weights
    const layerW: number[][] = [];
    const stdDev = Math.sqrt(2 / fromSize); // He weight init
    for (let j = 0; j < toSize; j++) {
      const neuronW: number[] = [];
      for (let i = 0; i < fromSize; i++) {
        // Random normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1 || 0.0001)) * Math.cos(2.0 * Math.PI * u2);
        neuronW.push(z0 * stdDev);
      }
      layerW.push(neuronW);
    }
    weights.push(layerW);

    // Initialize Biases
    const layerB = new Array(toSize).fill(0.1); // Small positive activation offsets
    biases.push(layerB);
  }

  return { weights, biases };
}

/**
 * Runs forward propagation on an input, caching weights activation records
 * for math visualizations and backpropagation.
 */
export function forwardPropagate(
  inputVec: number[],
  network: NetworkState,
  activation: ActivationType
) {
  const activations: number[][] = [inputVec];
  const preActivations: number[][] = []; // Raw network input before activation (Z values)

  const actF = ACTIVATIONS[activation].func;

  for (let l = 0; l < network.weights.length; l++) {
    const W = network.weights[l];
    const b = network.biases[l];
    const prevA = activations[l];

    const currentZ: number[] = [];
    const currentA: number[] = [];

    for (let j = 0; j < W.length; j++) {
      // z = sum(w * x) + b
      let z = b[j];
      for (let i = 0; i < W[j].length; i++) {
        z += W[j][i] * prevA[i];
      }
      currentZ.push(z);
      // For final layers in binary classification, sigmoid handles thresholds perfectly
      const finalLayer = l === network.weights.length - 1;
      const val = finalLayer ? ACTIVATIONS.sigmoid.func(z) : actF(z);
      currentA.push(val);
    }

    preActivations.push(currentZ);
    activations.push(currentA);
  }

  return { activations, preActivations };
}

/**
 * Executes a single backpropagation training iteration on a given single row,
 * updating core parameters via gradient descent.
 */
export function backpropagateStep(
  inputVec: number[],
  target: number,
  network: NetworkState,
  activation: ActivationType,
  learningRate: number
): { computedOutput: number; error: number; updated: NetworkState } {
  // 1. Forward propagation
  const { activations, preActivations } = forwardPropagate(inputVec, network, activation);
  const output = activations[activations.length - 1][0]; // Assuming 1 output node
  const error = target - output;

  // Clone network parameters to maintain deep purity
  const W_updated = network.weights.map(layer => layer.map(n => [...n]));
  const b_updated = network.biases.map(layer => [...layer]);

  const numLayers = network.weights.length;
  const dActivations = new Array(numLayers + 1);
  const dPreActivations = new Array(numLayers);

  // Gradient of output node (MSE loss paired with Sigmoid output derivative)
  // J = 1/2 (y - y_hat)^2 => dJ/dy_hat = -(y - y_hat) = (y_hat - y)
  // dy_hat/dz = y_hat * (1 - y_hat)
  const finalZ = preActivations[preActivations.length - 1][0];
  const finalA = activations[activations.length - 1][0];
  const dLoss_dA = finalA - target;
  const dA_dZ = ACTIVATIONS.sigmoid.deriv(finalA);
  const dZ_final = dLoss_dA * dA_dZ;

  dPreActivations[numLayers - 1] = [dZ_final];

  // 2. Compute error vectors back propagated through hidden layers
  const actDeriv = ACTIVATIONS[activation].deriv;

  for (let l = numLayers - 2; l >= 0; l--) {
    const W_next = W_updated[l + 1];
    const dZ_next = dPreActivations[l + 1];
    const currentZ = preActivations[l];
    const currentA = activations[l + 1];
    const layerDZ: number[] = [];

    for (let i = 0; i < W_updated[l].length; i++) {
      let gradAccum = 0;
      for (let j = 0; j < W_next.length; j++) {
        gradAccum += W_next[j][i] * dZ_next[j];
      }
      const dZ = gradAccum * actDeriv(currentA[i]);
      layerDZ.push(dZ);
    }
    dPreActivations[l] = layerDZ;
  }

  // 3. Update active weights and biases using computed gradient matrices
  for (let l = 0; l < numLayers; l++) {
    const dZ = dPreActivations[l];
    const aPrev = activations[l];

    for (let j = 0; j < W_updated[l].length; j++) {
      // bias update: delta_bias = learning_rate * dZ
      b_updated[l][j] -= learningRate * dZ[j];

      for (let i = 0; i < W_updated[l][j].length; i++) {
        // weight update: delta_weight = learning_rate * dZ * a_prev
        W_updated[l][j][i] -= learningRate * dZ[j] * aPrev[i];
      }
    }
  }

  return {
    computedOutput: output,
    error,
    updated: {
      weights: W_updated,
      biases: b_updated
    }
  };
}

/**
 * Builds standalone HTML code, embodying the model's architecture, trained parameter vectors, 
 * vocabulary datasets, and fully styled web UI metrics for clean offline usage.
 */
export function generateExporthtml(
  dataset: Dataset,
  network: NetworkState,
  activation: ActivationType,
  hiddenLayersLength: number[],
  modelName: string
): string {
  const serializedWeights = JSON.stringify(network.weights);
  const serializedBiases = JSON.stringify(network.biases);
  const serializedVocab = JSON.stringify(dataset.vocab || null);
  const serializedFeatureMM = JSON.stringify(dataset.featureMinMax);
  const serializedTargetMinMax = JSON.stringify(dataset.targetMinMax);
  const isClassificationText = dataset.vocab ? 'true' : 'false';
  const headersListStr = JSON.stringify(dataset.headers);
  const numInputs = dataset.X[0]?.length || 0;

  const titleSafe = modelName ? modelName.replace(/['"„“]/g, '') : 'Математический Экземпляр ИИ';

  return `<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${titleSafe} - Экспортированная Модель ИИ</title>
    <!-- Tailwind Engine -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background-color: #0b0f17;
            color: #f1f5f9;
        }
        .code-font {
            font-family: 'JetBrains Mono', monospace;
        }
    </style>
</head>
<body class="p-6 md:p-12">
    <div class="max-w-4xl mx-auto space-y-8">
        
        <!-- Header -->
        <div class="border-b border-slate-800 pb-6">
            <span class="text-xs font-semibold uppercase tracking-wider text-teal-400 code-font">Standalone Inference Module</span>
            <h1 class="text-3xl md:text-4xl font-extrabold text-slate-100 hover:text-white transition">
                ИИ Модель: ${titleSafe}
            </h1>
            <p class="text-xs md:text-sm text-slate-400 mt-2">
                Полностью автономный модуль математической нейросети, скомпилированный из вашей сессии обучения.
            </p>
        </div>

        <!-- Model Configuration Data Matrix -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5">
                <h2 class="text-md font-semibold text-teal-400 mb-3 code-font">Спецификация Модели</h2>
                <div class="space-y-2 text-xs md:text-sm text-slate-300">
                    <div class="flex justify-between border-b border-slate-800 pb-2">
                        <span>Архитектура:</span>
                        <span class="text-teal-300 code-font">[${numInputs} ➔ ${hiddenLayersLength.join(' ➔ ')} ➔ 1]</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-800 pb-2">
                        <span>Функция активации:</span>
                        <span class="text-yellow-400 code-font uppercase font-medium">${activation}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-800 pb-2">
                        <span>Тип классификатора:</span>
                        <span class="text-purple-400 font-medium">${dataset.vocab ? 'Анализ Текста (NLP - Bag of Words)' : 'Табличные данные (Числовые Фичи)'}</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Статус:</span>
                        <span class="text-green-400 font-bold flex items-center gap-1.5">
                            <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            Готова к предсказаниям
                        </span>
                    </div>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
                <div>
                    <h2 class="text-md font-semibold text-yellow-400 mb-1 code-font">Принцип Мышления</h2>
                    <p class="text-xs text-slate-400 leading-relaxed">
                        ИИ выполняет взвешенное суммирование по формуле <code class="bg-slate-950 px-1.5 py-0.5 rounded text-teal-300 pointer">z = W·a + b</code> на каждом из слоёв перед применением активации <code class="bg-slate-950 px-1 py-0.5 rounded text-yellow-300">${activation}</code>.
                    </p>
                </div>
                <div class="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 code-font">
                    Углы наклона весов зафиксированы в коде на основе SGD.
                </div>
            </div>
        </div>

        <!-- Testing Playground Interfaces -->
        <div class="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-6">
            <h2 class="text-lg font-bold text-slate-100 flex items-center gap-2">
                <span class="flex items-center justify-center w-6 h-6 rounded bg-teal-500/10 text-teal-400 text-xs code-font">▸</span>
                Математический Тест Весовой Модели
            </h2>

            <div class="space-y-4">
                <div id="textModeContainer" class="${dataset.vocab ? '' : 'hidden'} space-y-2">
                    <label class="block text-xs font-semibold text-slate-400 code-font uppercase">Текст для Анализа Сентемента</label>
                    <textarea id="nlpInput" class="w-full h-24 bg-slate-950 border border-slate-800 rounded-lg p-3 text-slate-200 text-sm focus:outline-none focus:border-teal-500 font-mono" placeholder="Введите текст..."></textarea>
                    <span class="text-[11px] text-slate-500">Система преобразует текст в числовую матрицу согласно словарю vocabulary ({wordCount: ${Object.keys(dataset.vocab || {}).length}}).</span>
                </div>

                <div id="tabularModeContainer" class="${dataset.vocab ? 'hidden' : ''} space-y-4">
                    <label class="block text-xs font-semibold text-slate-400 code-font uppercase">Спецификация Фичей (Числа)</label>
                    <div id="featuresGrid" class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <!-- Dynamic Feature Inputs -->
                    </div>
                </div>

                <button onclick="runInference()" class="w-full bg-gradient-to-r from-teal-500 to-emerald-600 font-bold text-slate-950 rounded-lg py-3 hover:opacity-90 active:scale-[0.99] transition duration-200 text-sm cursor-pointer code-font uppercase">
                    Вычислить Результат
                </button>
            </div>
        </div>

        <!-- Predict Output Display Card -->
        <div id="outputCard" class="hidden bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
                <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-400 code-font">Результат Математического Перемножения</h3>
                <span id="outputValueBadge" class="text-xs bg-teal-500/10 text-teal-300 font-mono px-2 py-1 rounded border border-teal-500/10">0.0000</span>
            </div>

            <div class="flex flex-col sm:flex-row items-center gap-6 py-2">
                <!-- Large Visual Dial Indicator -->
                <div class="relative w-28 h-28 flex items-center justify-center bg-slate-950 rounded-full border border-slate-800/80">
                    <div class="absolute inset-2 rounded-full border border-dashed border-teal-500/25 animate-spin"></div>
                    <span id="outputPercentage" class="text-2xl font-black text-teal-400 code-font">0%</span>
                </div>

                <div class="flex-1 space-y-2 text-center sm:text-left">
                    <h4 id="classificationLabel" class="text-lg font-bold text-slate-100">Позитивное суждение</h4>
                    <p id="classificationMath" class="text-xs text-slate-400 leading-relaxed max-w-md">
                        Взвешенные узлы выдали оценку x. Ошибка дельта стремится к минимуму. Синапсы зафиксировали высокую корреляцию.
                    </p>
                </div>
            </div>

            <!-- Trace Math Log Area -->
             <div class="pt-4 border-t border-slate-800/80 space-y-2">
                <span class="block text-xs font-semibold text-slate-400 uppercase tracking-wider code-font">Калькуляционная выписка (Внутренние тензоры):</span>
                <div id="mathLog" class="bg-slate-950 rounded-lg p-4 text-[11px] leading-relaxed text-slate-400 code-font whitespace-pre-wrap max-h-56 overflow-y-auto">
                    <!-- Logs go here -->
                </div>
             </div>
        </div>

    </div>

    <!-- Scripting Infrastructure -->
    <script>
        // Core Static Neural Net Engine Details
        const weights = ${serializedWeights};
        const biases = ${serializedBiases};
        const vocab = ${serializedVocab};
        const featureMinMax = ${serializedFeatureMM};
        const targetMinMax = ${serializedTargetMinMax};
        const isNLP = ${isClassificationText};
        const headers = ${headersListStr};
        const actName = '${activation}';

        // Mathematics Activation Implementations
        const acts = {
            sigmoid: x => 1 / (1 + Math.exp(-x)),
            relu: x => Math.max(0, x),
            tanh: x => Math.tanh(x)
        };

        // Render input inputs for numerical indicators
        if (!isNLP) {
            const grid = document.getElementById('featuresGrid');
            headers.forEach((h, idx) => {
                const mm = featureMinMax[idx] || { min: 0, max: 100 };
                const step = (mm.max - mm.min) / 100 || 1;
                grid.innerHTML += \`
                    <div class="space-y-1.5 p-3.5 bg-slate-950 rounded-lg border border-slate-800/70">
                        <label class="block text-xs text-slate-400 font-semibold truncate hover:text-slate-200 transition">\${h}</label>
                        <div class="flex gap-2 items-center">
                            <input type="range" id="f_range_\${idx}" min="\${mm.min}" max="\${mm.max}" step="\${step}" value="\${(mm.min+mm.max)/2}" oninput="document.getElementById('f_val_\${idx}').value=this.value; runInference()" class="w-full accent-teal-500 h-1 rounded cursor-pointer bg-slate-800">
                            <input type="number" id="f_val_\${idx}" min="\${mm.min}" max="\${mm.max}" step="\${step}" value="\${((mm.min+mm.max)/2).toFixed(2)}" oninput="document.getElementById('f_range_\${idx}').value=this.value; runInference()" class="w-16 h-7 bg-slate-900 border border-slate-800 rounded font-mono text-center text-xs focus:outline-none focus:border-teal-500">
                        </div>
                        <span class="text-[10px] text-slate-500 font-mono">Шкала: \${mm.min} \${isNaN(mm.min) ? '' : '➔'} \${mm.max}</span>
                    </div>
                \`;
            });
        }

        function tokenizeText(text) {
            return text.toLowerCase()
                .replace(/[.,\\/#!$%\\^&\\*;:{}=\\-_\\x60~()?"“”']/g, '')
                .split(/\\s+/)
                .filter(Boolean);
        }

        function runInference() {
            let xVec = [];
            let mathLogStr = '=== Инициализация Прямого Распространения (Forward Pass) ===\\n';

            if (isNLP) {
                const text = document.getElementById('nlpInput').value || '';
                const tokens = tokenizeText(text);
                const vocabSize = Object.keys(vocab || {}).length;
                xVec = new Array(vocabSize).fill(0);
                
                mathLogStr += \`Вводный текст: "\${text}"\\nТокены после нормализации: [\${tokens.join(', ')}]\\n\\n\`;
                
                tokens.forEach(tok => {
                    if (vocab[tok] !== undefined) {
                        xVec[vocab[tok]] += 1;
                    }
                });

                mathLogStr += 'Результирующий вектор частот (Bag-of-Words):\\n[';
                xVec.forEach((val, id) => {
                    if (val > 0) {
                        const wordsArr = Object.keys(vocab);
                        const matchedWord = wordsArr.find(k => vocab[k] === id);
                        mathLogStr += \` "\${matchedWord}": \${val}, \`;
                    }
                });
                mathLogStr += ']\\n\\n';
            } else {
                headers.forEach((h, idx) => {
                    const rawVal = Number(document.getElementById('f_val_' + idx).value) || 0;
                    const mm = featureMinMax[idx] || { min: 0, max: 1 };
                    const norm = (rawVal - mm.min) / (mm.max - mm.min || 1);
                    xVec.push(norm);
                    mathLogStr += \`Фича [\\\`\${h}\\\`]: введено \${rawVal.toFixed(2)}, нормировано ➔ \${norm.toFixed(4)}\\n\`;
                });
                mathLogStr += '\\n';
            }

            // Math Layer loops feedforward
            let currentActivations = [...xVec];
            const actF = acts[actName] || acts.sigmoid;

            for (let l = 0; l < weights.length; l++) {
                const W = weights[l];
                const b = biases[l];
                const nextActivations = [];
                const isOutput = l === weights.length - 1;

                mathLogStr += \`--- Слой \${l + 1} (Переход \${currentActivations.length} нейронов ➔ \${W.length} нейронов) ---\\n\`;

                for (let j = 0; j < W.length; j++) {
                    let z = b[j];
                    let sumTerm = \`b[\${j}](\${b[j].toFixed(4)})\`;
                    for (let i = 0; i < W[j].length; i++) {
                        z += W[j][i] * currentActivations[i];
                        if (W[j][i] !== 0 && currentActivations[i] !== 0 && W[j].length < 15) {
                            sumTerm += \` + w[\${j}][\${i}](\${W[j][i].toFixed(3)})·a[\${i}](\${currentActivations[i].toFixed(2)})\`;
                        }
                    }
                    if (W[j].length >= 15) sumTerm += ' + ...векторное умножение весов...';
                    
                    const actVal = isOutput ? acts.sigmoid(z) : actF(z);
                    nextActivations.push(actVal);

                    mathLogStr += \`  Узел \${j}: z = \${z.toFixed(5)} ➔ a(\${isOutput ? 'sigmoid' : actName}) = \${actVal.toFixed(6)}\\n\`;
                }
                currentActivations = nextActivations;
                mathLogStr += '\\n';
            }

            const rawVal = currentActivations[0];
            mathLogStr += \`=== ИТОГОВОЕ ЗНАЧЕНИЕ СЕТИ: \${rawVal.toFixed(6)} ===\\n\`;

            // Render Output details
            const outCard = document.getElementById('outputCard');
            outCard.classList.remove('hidden');

            document.getElementById('outputValueBadge').innerText = rawVal.toFixed(5);
            const percentage = Math.round(rawVal * 100);
            document.getElementById('outputPercentage').innerText = percentage + '%';

            const classificationLabel = document.getElementById('classificationLabel');
            const classificationMath = document.getElementById('classificationMath');

            if (isNLP) {
                if (rawVal > 0.5) {
                    classificationLabel.innerHTML = 'Позитивная тональность (Positive Label)';
                    classificationLabel.className = 'text-lg font-bold text-teal-400';
                    classificationMath.innerText = \`Математическая нейросеть предсказывает позитивный окрас с вероятностью \${percentage}%. Индексы слов сложились с положительным вектором.\`;
                } else {
                    classificationLabel.innerHTML = 'Негативная тональность (Negative Label)';
                    classificationLabel.className = 'text-lg font-bold text-red-400';
                    classificationMath.innerText = \`Математическая нейросеть предсказывает негативный окрас с вероятностью \${100 - percentage}%. Взвешенные весы дали отрицательную сумму.\`;
                }
            } else {
                // Quantitative regression or threshold scaling
                const actMetric = rawVal * (targetMinMax.max - targetMinMax.min) + targetMinMax.min;
                classificationLabel.innerHTML = \`Предсказание: \${actMetric.toFixed(3)}\`;
                classificationLabel.className = 'text-lg font-bold text-yellow-400';
                classificationMath.innerText = \`Результирующий сигнал (\${rawVal.toFixed(4)}) пересчитан в исходные шкалы [\${targetMinMax.min.toFixed(1)} до \${targetMinMax.max.toFixed(1)}], что дает математическое значение \${actMetric.toFixed(4)}.\`;
            }

            document.getElementById('mathLog').innerText = mathLogStr;
        }

        // Active run automatic tracking
        if (isNLP) {
            document.getElementById('nlpInput').addEventListener('input', runInference);
            // Default input text sentiments
            document.getElementById('nlpInput').value = "математика у тебя очень классная и мощная";
            runInference();
        } else {
            runInference();
        }
    </script>
</body>
</html>`;
}
