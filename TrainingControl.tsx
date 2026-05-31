/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Play, Pause, RotateCcw, LineChart, Target, AlertCircle } from 'lucide-react';
import { ModelConfig, MetricPoint, TrainingState, Dataset } from '../types';

interface TrainingControlProps {
  config: ModelConfig;
  onChangeConfig: (config: ModelConfig) => void;
  dataset: Dataset | null;
  history: MetricPoint[];
  trainingState: TrainingState;
  onStartTraining: () => void;
  onPauseTraining: () => void;
  onResetNN: () => void;
}

export function TrainingControl({
  config,
  onChangeConfig,
  dataset,
  history,
  trainingState,
  onStartTraining,
  onPauseTraining,
  onResetNN
}: TrainingControlProps) {
  const [lrInput, setLrInput] = useState(config.learningRate);

  const handleLrChange = (val: number) => {
    const parsed = Math.max(0.001, Math.min(1.0, val));
    setLrInput(parsed);
    onChangeConfig({ ...config, learningRate: parsed });
  };

  const handleEpochsChange = (val: number) => {
    const parsed = Math.max(10, Math.min(2000, val));
    onChangeConfig({ ...config, epochs: parsed });
  };

  const renderLossChart = () => {
    if (history.length === 0) {
      return (
        <div className="h-44 flex flex-col items-center justify-center border border-dashed border-slate-800 rounded-xl bg-slate-950/40 text-xs text-slate-500 font-mono">
          <LineChart className="w-5 h-5 mb-1.5 opacity-60 text-teal-400" />
          <span>Матрица Потерь пуста. Начните обучение.</span>
        </div>
      );
    }

    const w = 500;
    const h = 180;
    const padding = 20;

    const maxLoss = Math.max(...history.map(pt => pt.loss), 0.1);
    const coords = history.map((pt, index) => {
      // Map epochs to x coordinate, loss to y coordinate
      const x = padding + (index / (history.length - 1 || 1)) * (w - padding * 2);
      const y = h - padding - (pt.loss / maxLoss) * (h - padding * 2);
      return { x, y };
    });

    let pathD = `M ${coords[0].x} ${coords[0].y}`;
    for (let i = 1; i < coords.length; i++) {
      pathD += ` L ${coords[i].x} ${coords[i].y}`;
    }

    // Build area path for visual fill gradient
    const areaD = `${pathD} L ${coords[coords.length - 1].x} ${h - padding} L ${coords[0].x} ${h - padding} Z`;

    const latestLoss = history[history.length - 1].loss;
    const initialLoss = history[0].loss;

    return (
      <div className="bg-slate-950 rounded-xl border border-slate-850 p-4 space-y-3 relative overflow-hidden">
        <div className="flex justify-between items-center text-xs">
          <span className="font-semibold text-slate-400 font-mono flex items-center gap-1.5">
            <LineChart className="w-3.5 h-3.5 text-teal-400 animate-pulse" />
            График Снижения Ошибки (Loss Trend)
          </span>
          <span className="font-mono text-slate-500 text-[10px]">
            Старт: {initialLoss.toFixed(4)} ➔ Актуал: {latestLoss.toFixed(4)}
          </span>
        </div>

        <div className="w-full overflow-x-auto pr-1">
          <svg className="w-[100%] h-[180px] min-w-[300px]" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
            <defs>
              <linearGradient id="gradientArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0d9488" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#0d9488" stopOpacity="0.0"/>
              </linearGradient>
            </defs>

            {/* Grid Lines */}
            <line x1={padding} y1={padding} x2={w - padding} y2={padding} stroke="#1e293b" strokeWidth="0.5" strokeDasharray="2,2" />
            <line x1={padding} y1={h / 2} x2={w - padding} y2={h / 2} stroke="#1e293b" strokeWidth="0.5" strokeDasharray="2,2" />
            <line x1={padding} y1={h - padding} x2={w - padding} y2={h - padding} stroke="#334155" strokeWidth="1" />

            {/* Render Filled Area */}
            <path d={areaD} fill="url(#gradientArea)" />

            {/* Render Connection Stroke */}
            <path d={pathD} fill="none" stroke="#2dd4bf" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

            {/* Highlights latest dot */}
            <circle cx={coords[coords.length - 1].x} cy={coords[coords.length - 1].y} r="4.5" fill="#2dd4bf" stroke="#000" strokeWidth="1.5" />
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 space-y-6">
      
      {/* Target Title */}
      <div className="flex items-center gap-2">
        <span className="flex items-center justify-center w-6 h-6 rounded-md bg-teal-500/10 text-teal-400">
          <Target className="w-4 h-4" />
        </span>
        <h2 className="text-lg font-bold text-slate-100 font-sans">Шаг 3: Математическое Обучение (SGD)</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Param Adjustment dials */}
        <div className="space-y-4 pt-1">
          <span className="block text-xs font-semibold text-slate-400 code-font uppercase">Параметры Сессии</span>
          
          {/* Learning Rate slider input */}
          <div className="space-y-2 p-3 bg-slate-950/60 rounded-xl border border-slate-850/60">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">Коэффициент обучения (η)</span>
              <span className="text-teal-400 font-mono font-semibold">{config.learningRate}</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={config.learningRate}
              onChange={(e) => handleLrChange(Number(e.target.value))}
              disabled={trainingState.isRunning}
              className="w-full accent-teal-500 h-1 rounded cursor-pointer bg-slate-800 disabled:opacity-40"
            />
            <p className="text-[9px] text-slate-500 leading-relaxed leading-tighter">Скорость шага градиентного спуска. Маленькая η — аккуратно, но долго; большая — быстрее, но может взлететь ошибка.</p>
          </div>

          {/* Epochs count slider */}
          <div className="space-y-2 p-3 bg-slate-950/60 rounded-xl border border-slate-850/60">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">Эпохи (Итерации)</span>
              <span className="text-teal-400 font-mono font-semibold">{config.epochs}</span>
            </div>
            <input
              type="range"
              min="20"
              max="1500"
              step="20"
              value={config.epochs}
              onChange={(e) => handleEpochsChange(Number(e.target.value))}
              disabled={trainingState.isRunning}
              className="w-full accent-teal-500 h-1 rounded cursor-pointer bg-slate-800 disabled:opacity-40"
            />
            <p className="text-[9px] text-slate-500 leading-relaxed">Длина проходов всего датасета. Оптимум лежит в области схождения тренда.</p>
          </div>
        </div>

        {/* Live Loss Chart output */}
        <div className="md:col-span-2 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-850">
              <span className="block text-[10px] text-slate-500 uppercase tracking-wider font-mono">Эпоха</span>
              <span className="text-xl font-bold font-mono text-slate-100 mt-1 block">
                {trainingState.currentEpoch} <span className="text-xs text-slate-500 font-normal">/ {config.epochs}</span>
              </span>
            </div>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-850">
              <span className="block text-[10px] text-slate-500 uppercase tracking-wider font-mono">Loss (MSE)</span>
              <span className={`text-xl font-bold font-mono mt-1 block ${history.length > 0 ? 'text-yellow-400' : 'text-slate-500'}`}>
                {history.length > 0 ? history[history.length - 1].loss.toFixed(6) : 'N/A'}
              </span>
            </div>
          </div>

          {renderLossChart()}
        </div>

      </div>

      {/* Action triggers */}
      <div className="flex flex-col sm:flex-row gap-3 pt-2">
        {trainingState.isRunning ? (
          <button
            type="button"
            onClick={onPauseTraining}
            className="flex-1 bg-yellow-500 hover:bg-yellow-400 text-slate-950 font-bold py-3 px-4 rounded-xl active:scale-[0.99] transition duration-150 inline-flex items-center justify-center gap-2 cursor-pointer text-xs uppercase tracking-wider font-mono"
          >
            <Pause className="w-4 h-4" /> Приостановить Обучение
          </button>
        ) : (
          <button
            type="button"
            onClick={onStartTraining}
            disabled={!dataset}
            className="flex-1 bg-gradient-to-r from-teal-500 to-emerald-600 hover:from-teal-400 hover:to-emerald-500 text-slate-950 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 font-bold py-3 px-4 rounded-xl active:scale-[0.99] transition duration-150 inline-flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed text-xs uppercase tracking-wider font-mono"
          >
            <Play className="w-4 h-4" /> Начать Математическое Обучение
          </button>
        )}

        <button
          type="button"
          onClick={onResetNN}
          className="bg-slate-950 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 py-3 px-5 rounded-xl active:scale-[0.99] transition duration-150 inline-flex items-center justify-center gap-2 cursor-pointer text-xs uppercase tracking-wider font-mono"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Сбросить веса (Reset)
        </button>
      </div>

    </div>
  );
}
