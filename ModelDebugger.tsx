/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { PlayCircle, HelpCircle, Code, ChevronRight, Hash } from 'lucide-react';
import { Dataset, NetworkState, ActivationType } from '../types';
import { forwardPropagate, ACTIVATIONS } from '../utils/mathEngine';

interface ModelDebuggerProps {
  dataset: Dataset | null;
  network: NetworkState | null;
  activation: ActivationType;
}

export function ModelDebugger({ dataset, network, activation }: ModelDebuggerProps) {
  const [selectedRowId, setSelectedRowId] = useState<number>(0);
  const [calcLogs, setCalcLogs] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);

  if (!dataset || !network) return null;

  const handleTrace = () => {
    const inputVec = dataset.X[selectedRowId];
    if (!inputVec) return;

    const { activations, preActivations } = forwardPropagate(inputVec, network, activation);

    const logs: string[] = [];
    logs.push(`Входящие тензоры (x): [${inputVec.map(v => v.toFixed(4)).join(', ')}]`);

    // Trace through layers dynamically
    for (let l = 0; l < network.weights.length; l++) {
      const W = network.weights[l];
      const b = network.biases[l];
      const prevA = activations[l];
      const isOutput = l === network.weights.length - 1;
      const actName = isOutput ? 'sigmoid' : activation;

      logs.push(`\n=== СЛОЙ ${l + 1} (${isOutput ? 'Выходной' : 'Скрытый'}) ===`);

      W.forEach((toWeights, neuronId) => {
        let formulaStr = `z = b[${neuronId}](${b[neuronId].toFixed(4)})`;
        let sumAccum = b[neuronId];

        toWeights.forEach((wVal, fromId) => {
          const product = wVal * prevA[fromId];
          sumAccum += product;
          
          if (wVal !== 0 && prevA[fromId] !== 0) {
            formulaStr += ` + w_${neuronId}_${fromId}(${wVal.toFixed(3)}) * a_${fromId}(${prevA[fromId].toFixed(3)})`;
          }
        });

        const actVal = isOutput ? ACTIVATIONS.sigmoid.func(sumAccum) : ACTIVATIONS[activation].func(sumAccum);
        formulaStr += ` = ${sumAccum.toFixed(5)}`;
        
        logs.push(`Узел #${neuronId + 1}:`);
        logs.push(`  Формула взвешивания: ${formulaStr}`);
        logs.push(`  Активация a = ${actName}(z) = ${actVal.toFixed(6)}`);
      });
    }

    const finalOutput = activations[activations.length - 1][0];
    const target = dataset.y[selectedRowId];
    const rawMSE = 0.5 * Math.pow(target - finalOutput, 2);

    logs.push(`\n=== ФИНАЛЬНАЯ КОМПИЛЯЦИЯ ПРЕДСКАЗАНИЯ ===`);
    logs.push(`Полученный выход сети (y_hat): ${finalOutput.toFixed(6)}`);
    logs.push(`Ожидаемое значение (y): ${target.toFixed(2)}`);
    logs.push(`Индивидуальная ошибка MSE: ${rawMSE.toFixed(8)}`);

    setCalcLogs(logs);
    setPrediction(finalOutput);
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 space-y-6">
      
      {/* Header Info */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-teal-500/10 text-teal-400">
            <Code className="w-4 h-4" />
          </span>
          <h2 className="text-lg font-bold text-slate-100 font-sans">Шаг 4: Математический Трейсинг (Debugger)</h2>
        </div>
        <span className="text-[10px] bg-slate-950 border border-slate-850 px-2.5 py-0.5 rounded-full text-slate-400 code-font">
          Без API-ключей
        </span>
      </div>

      <div className="space-y-4">
        
        {/* Row selection dial */}
        <div className="flex flex-col sm:flex-row items-center gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-850/60 text-xs">
          <div className="flex-1 space-y-1">
            <span className="font-bold text-slate-200">Выберите элемент датасета для сканирования:</span>
            <p className="text-[10px] text-slate-500 leading-relaxed">Пошагово вычислим значения для выбранной строки и покажем формулы.</p>
          </div>

          <div className="flex gap-2 w-full sm:w-auto shrink-0">
            <select
              value={selectedRowId}
              onChange={(e) => {
                setSelectedRowId(Number(e.target.value));
                setPrediction(null);
                setCalcLogs([]);
              }}
              className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-slate-300 font-mono text-xs focus:outline-none focus:border-teal-500 transition cursor-pointer flex-1"
            >
              {dataset.X.map((_, idx) => (
                <option key={idx} value={idx}>
                  Строка #{idx} (Цель: {dataset.y[idx].toFixed(2)})
                </option>
              ))}
            </select>

            <button
              type="button"
              onClick={handleTrace}
              className="bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold px-4 py-2 rounded-lg active:scale-95 transition flex items-center justify-center gap-1.5 cursor-pointer text-xs uppercase tracking-wide code-font"
            >
              <PlayCircle className="w-4 h-4" /> ВЫЧИСЛИТЬ
            </button>
          </div>
        </div>

        {/* Visual Outputs Trace */}
        {prediction !== null && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            
            {/* Predictions comparison card */}
            <div className="bg-slate-950 rounded-xl border border-slate-850 p-4 flex flex-col justify-between">
              <div>
                <span className="block text-[10px] text-slate-500 uppercase tracking-wider font-mono">Прогноз ИИ vs Цель</span>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-2xl font-black text-teal-400 font-mono">{prediction.toFixed(4)}</span>
                  <span className="text-slate-600 font-mono">/</span>
                  <span className="text-sm text-slate-400 font-mono">{dataset.y[selectedRowId].toFixed(2)}</span>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-850/80 text-[10px] text-slate-400 leading-relaxed">
                {prediction > 0.5 && dataset.y[selectedRowId] === 1 ? (
                  <span className="text-emerald-400 font-bold">✓ Успех: Сигналы совпали (True Positive)</span>
                ) : prediction <= 0.5 && dataset.y[selectedRowId] === 0 ? (
                  <span className="text-emerald-400 font-bold">✓ Успех: Сигналы совпали (True Negative)</span>
                ) : (
                  <span className="text-yellow-400">В процессе схождения ошибки (Эпохи уменьшают зазор)</span>
                )}
              </div>
            </div>

            {/* Calculations logs trace terminal */}
            <div className="md:col-span-2 bg-slate-950 border border-slate-850 rounded-xl p-4 space-y-2 flex flex-col justify-between">
              <div className="flex items-center gap-1.5">
                <Hash className="w-3.5 h-3.5 text-yellow-500" />
                <span className="text-xs font-bold text-slate-350 code-font uppercase">Формулярный Журнал Умножений</span>
              </div>
              <div className="bg-slate-900 border border-slate-850 rounded-lg p-3 text-[10px] leading-relaxed font-mono text-slate-400 whitespace-pre-wrap max-h-48 overflow-y-auto mt-2 select-text shadow-inner">
                {calcLogs.map((logLine, lineId) => (
                  <div key={lineId} className="hover:text-slate-200 transition">{logLine}</div>
                ))}
              </div>
            </div>

          </div>
        )}

      </div>

    </div>
  );
}
