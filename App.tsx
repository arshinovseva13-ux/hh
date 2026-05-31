/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Brain, Sparkles, Binary, CheckCircle, Download } from 'lucide-react';
import { Dataset, ModelConfig, NetworkState, MetricPoint, TrainingState } from './types';
import { DatasetUploader } from './components/DatasetUploader';
import { NetworkBuilder } from './components/NetworkBuilder';
import { TrainingControl } from './components/TrainingControl';
import { ModelDebugger } from './components/ModelDebugger';
import { ExportPanel } from './components/ExportPanel';
import { AiAssistant } from './components/AiAssistant';
import { initializeWeights, backpropagateStep, parseTextDataset } from './utils/mathEngine';

const INITIAL_TEXT_PRESET = `ИИ мыслит невероятно красиво и математически точно | positive
мне очень не нравится сложный и запутаный код | negative
великолепный калькулятор с простой логикой | positive
ужасно долго грузится и выдает ошибки | negative
математика это лучшее изобретение человечества | positive
скучный нудный интерфейс и плохой дизайн | negative
обожаю строить нейронные сети своими руками | positive
это очень трудно сложно и неинтересно | negative`;

export default function App() {
  // 1. Initial Dataset Parsing (Immediate load)
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [rawText, setRawText] = useState(INITIAL_TEXT_PRESET);
  const [format, setFormat] = useState<'text' | 'tabular'>('text');
  
  // 2. Initial Model Configurations
  const [config, setConfig] = useState<ModelConfig>({
    learningRate: 0.15,
    epochs: 400,
    hiddenLayers: [
      { id: 'h1', neurons: 4 }
    ],
    activation: 'sigmoid',
    taskType: 'classification',
    modelName: 'Моя Нейросеть',
    externalAiType: 'disabled',
    externalAiKey: '',
    externalAiUrl: 'http://localhost:11434/v1/chat/completions',
    externalModelName: 'gemini-3.5-flash'
  });

  // 3. Network Matrices (Weights & Biases)
  const [network, setNetwork] = useState<NetworkState | null>(null);

  // 4. Session details
  const [history, setHistory] = useState<MetricPoint[]>([]);
  const [trainingState, setTrainingState] = useState<TrainingState>({
    currentEpoch: 0,
    history: [],
    isRunning: false,
    isPaused: false,
    speedMs: 25 // Time step sleep
  });

  const trainingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Automatically load the default sentiment preset on boot so visual elements render immediately
  useEffect(() => {
    try {
      const parsed = parseTextDataset(INITIAL_TEXT_PRESET, 'text');
      setDataset(parsed);
    } catch (e) {
      console.error('Ошибка инициализации дефолтного датасета:', e);
    }
  }, []);

  // Initialize weights whenever the layer architecture or input parameters change
  useEffect(() => {
    if (!dataset) return;
    const numInputNodes = dataset.X[0]?.length || 0;
    const layerStructure = [
      numInputNodes,
      ...config.hiddenLayers.map(l => l.neurons),
      1 // Output layer node
    ];
    const initialNetwork = initializeWeights(layerStructure);
    setNetwork(initialNetwork);
    setHistory([]);
    setTrainingState(prev => ({ ...prev, currentEpoch: 0, isRunning: false }));
  }, [dataset, config.hiddenLayers, config.activation]);

  const handleResetNN = () => {
    if (!dataset) return;
    const numInputNodes = dataset.X[0]?.length || 0;
    const layerStructure = [
      numInputNodes,
      ...config.hiddenLayers.map(l => l.neurons),
      1
    ];
    setNetwork(initializeWeights(layerStructure));
    setHistory([]);
    setTrainingState(prev => ({ ...prev, currentEpoch: 0, isRunning: false }));
  };

  // Safe training execution flow
  useEffect(() => {
    if (trainingState.isRunning && dataset && network) {
      const runEpoch = () => {
        setTrainingState(prev => {
          if (prev.currentEpoch >= config.epochs || !prev.isRunning) {
            if (trainingTimerRef.current) clearInterval(trainingTimerRef.current);
            return { ...prev, isRunning: false };
          }

          const nextEpoch = prev.currentEpoch + 1;
          let currentNet = { ...network };
          let totalSquaredError = 0;

          // Process SGD over entire data rows
          for (let r = 0; r < dataset.X.length; r++) {
            const rowX = dataset.X[r];
            const rowY = dataset.y[r];

            const stepResult = backpropagateStep(
              rowX,
              rowY,
              currentNet,
              config.activation,
              config.learningRate
            );

            currentNet = stepResult.updated;
            totalSquaredError += Math.pow(rowY - stepResult.computedOutput, 2);
          }

          const epochMSE = totalSquaredError / dataset.X.length;
          
          setNetwork(currentNet);
          setHistory(prevHist => {
            const updated = [...prevHist, { epoch: nextEpoch, loss: epochMSE }];
            // Limit length down to fits standard charting nicely
            if (updated.length > 500) {
              return updated.filter((_, id) => id % 2 === 0);
            }
            return updated;
          });

          return {
            ...prev,
            currentEpoch: nextEpoch
          };
        });
      };

      // Set up interval for stepping forward safely
      trainingTimerRef.current = setInterval(runEpoch, trainingState.speedMs);
    }

    return () => {
      if (trainingTimerRef.current) clearInterval(trainingTimerRef.current);
    };
  }, [trainingState.isRunning, dataset, network, config.epochs, config.learningRate, config.activation]);

  const handleStartTraining = () => {
    if (!dataset) return;
    if (trainingState.currentEpoch >= config.epochs) {
      // Automatic complete loop reset
      setHistory([]);
      setTrainingState(prev => ({ ...prev, currentEpoch: 0, isRunning: true }));
    } else {
      setTrainingState(prev => ({ ...prev, isRunning: true }));
    }
  };

  const handlePauseTraining = () => {
    setTrainingState(prev => ({ ...prev, isRunning: false }));
  };

  const [constructorDownloaded, setConstructorDownloaded] = useState(false);

  const handleDownloadWholeConstructor = async () => {
    try {
      const response = await fetch('./index.html');
      let text = '';
      if (response.ok) {
        text = await response.text();
      } else {
        const fallbackResponse = await fetch('./');
        text = await fallbackResponse.text();
      }

      if (!text || text.trim().length === 0) {
        throw new Error('Файл пуст или не получен');
      }

      const blob = new Blob([text], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'nn-constructor-open-source.html';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      setConstructorDownloaded(true);
      setTimeout(() => setConstructorDownloaded(false), 3000);
    } catch (e) {
      console.error('Ошибка скачивания приложения:', e);
    }
  };

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 flex flex-col font-sans selection:bg-teal-500/20 selection:text-teal-300">
      
      {/* Top Navigation Panel Header */}
      <header className="border-b border-slate-900 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-teal-500 rounded-lg blur opacity-30 animate-pulse"></div>
            <span className="relative flex items-center justify-center w-10 h-10 bg-gradient-to-br from-teal-500 to-emerald-600 rounded-lg text-slate-950 shadow-lg shadow-teal-500/10">
              <Brain className="w-5 h-5" />
            </span>
          </div>
          <div>
            <h1 className="text-md md:text-lg font-black tracking-tight flex items-center gap-2">
              ИИ Модель: <span className="text-teal-400 font-mono font-black">{config.modelName || 'Новая Нейросеть'}</span>
              <span className="text-[10px] bg-slate-900 border border-slate-800 text-teal-400 font-mono px-2 py-0.5 rounded-full font-bold uppercase tracking-widest hidden md:inline">
                Математический Движок
              </span>
            </h1>
            <p className="text-[10px] md:text-xs text-slate-400 mt-0.5 leading-none">
              Интерактивный Backpropagation симулятор весов нейросетей прямо в вашем браузере
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <button
            type="button"
            onClick={handleDownloadWholeConstructor}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-bold text-[10px] transition duration-150 active:scale-95 cursor-pointer font-mono ${
              constructorDownloaded
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'bg-purple-600 hover:bg-purple-550 text-slate-950 shadow-lg shadow-purple-500/10'
            }`}
            title="Скачать весь этот интерактивный конструктор как один полностью автономный HTML файл"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{constructorDownloaded ? 'СКАЧАНО (1 HTML) ✓' : 'ОТКРЫТЫЙ КОД (СКАЧАТЬ ВЕСЬ КОНСТРУКТОР)'}</span>
          </button>
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 bg-slate-900/60 border border-slate-850 rounded-lg font-mono text-[10px] text-slate-400">
            <Binary className="w-3.5 h-3.5 text-teal-400" />
            <span>ДВИЖОК: JS BACKPROPAGATION</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-lg font-semibold text-[10px]">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>100% OFFLINE</span>
          </div>
        </div>
      </header>

      {/* Main Workspace Frame Layout */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        
        {/* Left Grid Section: Data preparation & network shape configs */}
        <div className="space-y-6">
          <DatasetUploader
            onDatasetLoaded={setDataset}
            currentDataset={dataset}
            rawText={rawText}
            setRawText={setRawText}
            format={format}
            setFormat={setFormat}
          />

          <AiAssistant
            config={config}
            onChangeConfig={setConfig}
            dataset={dataset}
            network={network}
            onStartTraining={handleStartTraining}
            onPauseTraining={handlePauseTraining}
            onResetNN={handleResetNN}
            onLoadGeneratedDataset={(newRaw, newFormat) => {
              setRawText(newRaw);
              setFormat(newFormat);
              try {
                const parsed = parseTextDataset(newRaw, newFormat);
                setDataset(parsed);
              } catch (e) {
                console.error('Ошибка автоматического применения ИИ датасета:', e);
              }
            }}
          />

          <NetworkBuilder
            config={config}
            onChangeConfig={setConfig}
            network={network}
            numFeatures={dataset?.X[0]?.length || 0}
          />
        </div>

        {/* Right Grid Section: SGD parameters, charts, tracking debug logs & browser exporter */}
        <div className="space-y-6">
          <TrainingControl
            config={config}
            onChangeConfig={setConfig}
            dataset={dataset}
            history={history}
            trainingState={trainingState}
            onStartTraining={handleStartTraining}
            onPauseTraining={handlePauseTraining}
            onResetNN={handleResetNN}
          />
          
          <ModelDebugger
            dataset={dataset}
            network={network}
            activation={config.activation}
          />

          <ExportPanel
            dataset={dataset}
            network={network}
            activation={config.activation}
            hiddenLayersLength={config.hiddenLayers.map(l => l.neurons)}
            modelName={config.modelName}
          />
        </div>

      </main>

      {/* Humble Footer */}
      <footer className="border-t border-slate-950 py-5 bg-slate-950/20 text-center text-[10px] text-slate-500 font-mono">
        <div>Математический алгоритм стохастического градиентного спуска (SGD) и векторизатор текста NLP ведут автономные расчеты.</div>
        <div className="mt-1">Google AI Studio Build &bull; 2026</div>
      </footer>

    </div>
  );
}
