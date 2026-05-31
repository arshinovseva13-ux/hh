/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Layers, Plus, Trash2, Sliders, Info } from 'lucide-react';
import { ActivationType, LayerConfig, ModelConfig, NetworkState } from '../types';
import { ACTIVATIONS } from '../utils/mathEngine';

interface NetworkBuilderProps {
  config: ModelConfig;
  onChangeConfig: (newConfig: ModelConfig) => void;
  network: NetworkState | null;
  numFeatures: number;
}

export function NetworkBuilder({ config, onChangeConfig, network, numFeatures }: NetworkBuilderProps) {
  const addLayer = () => {
    if (config.hiddenLayers.length >= 3) return; // Prevent layout clutter
    const newLayer: LayerConfig = {
      id: Math.random().toString(36).substring(2, 9),
      neurons: 4
    };
    onChangeConfig({
      ...config,
      hiddenLayers: [...config.hiddenLayers, newLayer]
    });
  };

  const removeLayer = (id: string) => {
    onChangeConfig({
      ...config,
      hiddenLayers: config.hiddenLayers.filter(l => l.id !== id)
    });
  };

  const updateNeurons = (id: string, count: number) => {
    const validatedCount = Math.max(1, Math.min(8, count));
    onChangeConfig({
      ...config,
      hiddenLayers: config.hiddenLayers.map(l => (l.id === id ? { ...l, neurons: validatedCount } : l))
    });
  };

  const updateActivation = (type: ActivationType) => {
    onChangeConfig({
      ...config,
      activation: type
    });
  };

  // Helper to render static svg graph icon of activation functions
  const renderActivationPlot = (type: ActivationType) => {
    let path = '';
    if (type === 'sigmoid') {
      // S shape curve
      path = 'M 5 35 Q 25 35, 25 20 T 45 5';
    } else if (type === 'relu') {
      // flat then diagonal
      path = 'M 5 35 L 25 35 L 45 5';
    } else {
      // tanh (more aggressive S curve)
      path = 'M 5 35 C 15 35, 15 5, 25 5 C 35 5, 35 5, 45 5';
      path = 'M 5 35 Q 20 35, 25 20 T 45 5'; // simplified S shape
    }

    return (
      <svg className="w-12 h-10 border border-slate-800 rounded bg-slate-950 p-1 shrink-0" viewBox="0 0 50 40">
        <line x1="5" y1="20" x2="45" y2="20" stroke="#334155" strokeWidth="1" strokeDasharray="2,2" />
        <line x1="25" y1="5" x2="25" y2="35" stroke="#334155" strokeWidth="1" strokeDasharray="2,2" />
        <path d={path} fill="none" stroke="#2dd4bf" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    );
  };

  // Render highly advanced interactive synaptic graph of the live model
  const renderNetworkGraph = () => {
    if (!network) return null;

    // Define layers structure visually
    const layers = [
      Math.min(8, numFeatures), // Input layer (cap at 8 nodes for visual safety)
      ...config.hiddenLayers.map(l => l.neurons),
      1 // Output layer
    ];

    const svgWidth = 640;
    const svgHeight = 240;
    const paddingX = 50;
    const paddingY = 25;

    // Calculate node coordinates
    const layerSpacingX = (svgWidth - paddingX * 2) / (layers.length - 1);
    const nodes: Array<Array<{ x: number; y: number; layerIdx: number; nodeIdx: number }>> = [];

    layers.forEach((nodesSize, layerIdx) => {
      const x = paddingX + layerIdx * layerSpacingX;
      let layerSpaceY = (svgHeight - paddingY * 2) / (nodesSize - 1 || 1);
      if (nodesSize === 1) {
        layerSpaceY = 0;
      }

      const layerNodes = [];
      for (let i = 0; i < nodesSize; i++) {
        const y = nodesSize === 1 
          ? svgHeight / 2 
          : paddingY + i * layerSpaceY;
        layerNodes.push({ x, y, layerIdx, nodeIdx: i });
      }
      nodes.push(layerNodes);
    });

    return (
      <div className="bg-slate-950/80 rounded-xl border border-slate-850 p-4 space-y-2 relative overflow-hidden">
        <div className="absolute top-3 right-3 flex items-center gap-1 text-[10px] text-slate-500 font-mono">
          <span className="w-2 h-2 rounded-full bg-teal-500"></span> Вес (+) | <span className="w-2 h-2 rounded-full bg-orange-500"></span> Вес (-)
        </div>
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide font-mono mb-1.5">Генеральная Карта Синапсов (SVG)</h4>
        
        <div className="w-full overflow-x-auto pr-1">
          <svg className="w-[600px] h-[240px] mx-auto block" viewBox="0 0 640 240">
            {/* Draw weights connections first */}
            {network.weights.map((weightMatrix, lIdx) => {
              const currentLayerNodes = nodes[lIdx];
              const nextLayerNodes = nodes[lIdx + 1];

              return weightMatrix.map((toNeuronWeights, toIdx) => {
                const toNode = nextLayerNodes[toIdx];
                if (!toNode) return null;

                return toNeuronWeights.map((weightVal, fromIdx) => {
                  const fromNode = currentLayerNodes[fromIdx];
                  if (!fromNode) return null;

                  // Weight visuals options
                  const isPositive = weightVal >= 0;
                  const absW = Math.min(6, Math.abs(weightVal) * 3);
                  const color = isPositive 
                    ? `rgba(45, 212, 191, ${Math.min(0.8, 0.15 + absW / 5)})` 
                    : `rgba(249, 115, 22, ${Math.min(0.8, 0.15 + absW / 5)})`;
                  const strokeWidth = 0.5 + absW;

                  return (
                    <line
                      key={`${lIdx}-${toIdx}-${fromIdx}`}
                      x1={fromNode.x}
                      y1={fromNode.y}
                      x2={toNode.x}
                      y2={toNode.y}
                      stroke={color}
                      strokeWidth={strokeWidth}
                      className="transition-all duration-300"
                    />
                  );
                });
              });
            })}

            {/* Draw nodes circles */}
            {nodes.map((layerNodes, lIdx) => {
              const isInput = lIdx === 0;
              const isOutput = lIdx === nodes.length - 1;
              const fill = isInput 
                ? '#1e293b' 
                : isOutput 
                  ? '#0d9488' 
                  : '#111827';
              const stroke = isInput 
                ? '#475569' 
                : isOutput 
                  ? '#2dd4bf' 
                  : '#3b82f6';

              return layerNodes.map((n, nIdx) => (
                <g key={`${lIdx}-${nIdx}`}>
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={8}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth="2.5"
                    className="hover:r-10 transition duration-150"
                  />
                  <text
                    x={n.x}
                    y={n.y + 19}
                    textAnchor="middle"
                    className="fill-slate-500 font-mono text-[8px] uppercase tracking-tighter"
                  >
                    {isInput ? `x${nIdx + 1}` : isOutput ? 'output' : `h${lIdx}_${nIdx + 1}`}
                  </text>
                </g>
              ));
            })}
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 space-y-6">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-850/60 pb-4">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-teal-500/10 text-teal-400 font-bold">
            <Layers className="w-4 h-4" />
          </span>
          <h2 className="text-lg font-bold text-slate-100 font-sans">Шаг 2: Математический Конструктор ИИ</h2>
        </div>

        {/* Dynamic Model Name Config */}
        <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-850 w-full sm:w-auto">
          <span className="text-[10px] text-slate-500 font-mono uppercase font-bold pl-2">НАЗВАНИЕ:</span>
          <input
            type="text"
            value={config.modelName || 'Моя Нейросеть'}
            onChange={(e) => onChangeConfig({ ...config, modelName: e.target.value })}
            placeholder="Имя вашей модели..."
            className="w-full sm:w-44 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-teal-400 font-mono font-bold focus:outline-none focus:border-teal-500 transition"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Left column: Layout Config */}
        <div className="space-y-4">
          <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-850">
            <span className="text-xs font-semibold text-slate-400 code-font uppercase flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-teal-400" />
              Слои Нейросети
            </span>
            <button
              type="button"
              onClick={addLayer}
              disabled={config.hiddenLayers.length >= 3}
              className="text-[10px] bg-teal-500/10 hover:bg-teal-500/20 text-teal-400 font-bold px-2 py-1 rounded inline-flex items-center gap-1 transition-all hover:scale-105 disabled:opacity-40 disabled:scale-100 cursor-pointer code-font"
            >
              <Plus className="w-3 h-3" /> СЛОЙ (MAX 3)
            </button>
          </div>

          <div className="space-y-3">
            {/* Input layer label */}
            <div className="flex items-center justify-between p-3 bg-slate-950/60 rounded-xl border border-slate-850/60 text-xs">
              <div>
                <span className="font-bold text-slate-200">Входной Слой (Input Layer)</span>
                <p className="text-[10px] text-slate-500 mt-0.5">Картирует размер фичей датасета.</p>
              </div>
              <span className="bg-slate-900 border border-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono font-bold">
                {numFeatures} фич
              </span>
            </div>

            {/* Hidden layers iteration */}
            {config.hiddenLayers.map((layer, index) => (
              <div key={layer.id} className="flex items-center justify-between p-3 bg-slate-950/60 rounded-xl border border-slate-850 text-xs gap-4 relative group">
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1.5 h-6 bg-blue-500 rounded-r"></div>
                <div>
                  <span className="font-bold text-slate-200 pl-2">Скрытый Слой #{index + 1}</span>
                  <p className="text-[10px] text-slate-500 mt-0.5 pl-2">Отвечает за обучение сложным нелинейным закономерностям.</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded px-1">
                    <button
                      type="button"
                      onClick={() => updateNeurons(layer.id, layer.neurons - 1)}
                      className="text-slate-400 hover:text-slate-200 px-1 font-bold font-mono text-xs cursor-pointer"
                    >
                      -
                    </button>
                    <span className="text-xs text-slate-200 font-mono font-bold px-1.5">{layer.neurons}</span>
                    <button
                      type="button"
                      onClick={() => updateNeurons(layer.id, layer.neurons + 1)}
                      className="text-slate-400 hover:text-slate-200 px-1 font-bold font-mono text-xs cursor-pointer"
                    >
                      +
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeLayer(layer.id)}
                    className="text-red-400 hover:text-red-300 p-1 hover:bg-red-950/30 rounded transition cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}

            {/* Output layer label */}
            <div className="flex items-center justify-between p-3 bg-slate-950/60 rounded-xl border border-slate-850/60 text-xs">
              <div>
                <span className="font-bold text-slate-200">Выходной Слой (Output Layer)</span>
                <p className="text-[10px] text-slate-500 mt-0.5">Взвешивает и форматирует финальный ответ.</p>
              </div>
              <span className="bg-teal-500/10 border border-teal-500/20 text-teal-400 px-2 py-0.5 rounded font-mono font-bold">
                1 нейрон
              </span>
            </div>
          </div>
        </div>

        {/* Right column: Activation Functions Selector */}
        <div className="space-y-4">
          <span className="text-xs font-semibold text-slate-400 code-font uppercase flex items-center gap-1.5 p-1 bg-slate-950/40 border border-slate-850 rounded">
            <Info className="w-4 h-4 text-yellow-400" />
            Математическая активация (f(z))
          </span>

          <div className="space-y-2.5">
            {Object.entries(ACTIVATIONS).map(([key, item]) => {
              const active = config.activation === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => updateActivation(key as ActivationType)}
                  className={`w-full flex gap-3 text-left p-2.5 rounded-xl border transition duration-200 cursor-pointer ${
                    active 
                      ? 'bg-teal-500/5 border-teal-500/40 text-teal-300' 
                      : 'bg-slate-950/50 border-slate-850 hover:border-slate-800 text-slate-350'
                  }`}
                >
                  {renderActivationPlot(key as ActivationType)}
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold uppercase code-font">{key}</span>
                      <code className="text-[10px] text-yellow-400 font-mono">{item.formula}</code>
                    </div>
                    <p className="text-[9.5px] text-slate-400 leading-relaxed font-sans">{item.description}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

      </div>

      {renderNetworkGraph()}

    </div>
  );
}
