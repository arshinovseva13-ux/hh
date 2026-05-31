/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ActivationType = 'sigmoid' | 'relu' | 'tanh';

export interface TokenizerVocab {
  [word: string]: number;
}

export interface Dataset {
  headers: string[];
  rawRows: Array<Record<string, string | number>>;
  X: number[][]; // Numerical feature matrix
  y: number[];   // Numerical target vector (0 or 1 for classification, or normalized floats for regression)
  featureMinMax: Array<{ min: number; max: number }>;
  targetMinMax: { min: number; max: number };
  vocab?: TokenizerVocab; // For text-based vectorization
  taskType: 'classification' | 'regression';
}

export interface LayerConfig {
  id: string;
  neurons: number;
}

export interface ModelConfig {
  learningRate: number;
  epochs: number;
  hiddenLayers: LayerConfig[];
  activation: ActivationType;
  taskType: 'classification' | 'regression';
  modelName: string;
  externalAiType: 'disabled' | 'gemini' | 'openai' | 'custom';
  externalAiKey: string;
  externalAiUrl: string;
  externalModelName: string;
}

// Weights and biases for each layer transition
export interface NetworkState {
  weights: number[][][]; // weights[layerIndex][toNeuron][fromNeuron]
  biases: number[][];    // biases[layerIndex][toNeuron]
}

export interface MetricPoint {
  epoch: number;
  loss: number;
  accuracy?: number;
}

export interface TrainingState {
  currentEpoch: number;
  history: MetricPoint[];
  isRunning: boolean;
  isPaused: boolean;
  speedMs: number; // Interval speed for stepping
}
