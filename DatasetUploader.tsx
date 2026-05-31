/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Upload, FileSpreadsheet, Sparkles, Hash, AlertTriangle, Globe, Download, Loader2 } from 'lucide-react';
import { Dataset } from '../types';
import { parseTextDataset } from '../utils/mathEngine';

interface DatasetUploaderProps {
  onDatasetLoaded: (df: Dataset) => void;
  currentDataset: Dataset | null;
  rawText: string;
  setRawText: (val: string) => void;
  format: 'text' | 'tabular';
  setFormat: (val: 'text' | 'tabular') => void;
}

// Built-in presets for instant visual mathematical experimentation
export const PRESETS = {
  sentiment: {
    title: 'Отзывы об ИИ (NLP Текст)',
    description: 'Обучите ИИ понимать человеческую тональность без API-ключей, анализируя частоту слов в предложениях.',
    format: 'text' as const,
    data: `ИИ мыслит невероятно красиво и математически точно | positive
мне очень не нравится сложный и запутаный код | negative
великолепный калькулятор с простой логикой | positive
ужасно долго грузится и выдает ошибки | negative
математика это лучшее изобретение человечества | positive
скучный нудный интерфейс и плохой дизайн | negative
обожаю строить нейронные сети своими руками | positive
это очень трудно сложно и неинтересно | negative`
  },
  credit: {
    title: 'Оценка Доступа к Кредиту (Таблица)',
    description: 'Классическое табличное кредитование. Прогнозирует одобрение на основе доходов, кредитного рейтинга и возраста.',
    format: 'tabular' as const,
    data: `Доход, Возраст, Рейтинг, Одобрено
120000, 34, 780, 1
45000, 22, 600, 0
95000, 48, 710, 1
32000, 19, 580, 0
85000, 29, 690, 1
150000, 45, 820, 1
28000, 25, 500, 0
65000, 31, 640, 0`
  },
  parabola: {
    title: 'Математический Синус (Регрессия)',
    description: 'Математический аппроксиматор. Обучает нейросеть предсказывать вещественную функцию на основе двух переменных.',
    format: 'tabular' as const,
    data: `ПеременнаяX1, ПеременнаяX2, РезультатY
0.1, 0.9, 0.95
0.4, 0.8, 0.82
0.8, 0.3, 0.31
0.2, 0.2, 0.18
0.9, 0.9, 0.99
0.3, 0.6, 0.58
0.7, 0.1, 0.08
0.5, 0.5, 0.49`
  }
};

export function DatasetUploader({
  onDatasetLoaded,
  currentDataset,
  rawText,
  setRawText,
  format,
  setFormat
}: DatasetUploaderProps) {
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  // Internet Dataset Download form
  const [downloadUrl, setDownloadUrl] = useState('');
  const [downloadFormat, setDownloadFormat] = useState<'text' | 'tabular'>('tabular');
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (!downloadUrl) return;
    setIsDownloading(true);
    setErrorMsg(null);
    try {
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error(`Статус HTTP ${response.status}`);
      }
      const text = await response.text();
      if (!text || text.trim().length === 0) {
        throw new Error('Загруженный файл оказался пустым.');
      }
      setRawText(text);
      setFormat(downloadFormat);
      const parsed = parseTextDataset(text, downloadFormat);
      onDatasetLoaded(parsed);
    } catch (e: any) {
      console.error(e);
      setErrorMsg(`Сброс: Не удалось закачать по URL. Причина: CORS или неверный адрес. Вы можете скопировать данные вручную или загрузить файл напрямую в поле ввода.`);
    } finally {
      setIsDownloading(false);
    }
  };

  const handlePresetSelect = (key: keyof typeof PRESETS) => {
    const preset = PRESETS[key];
    setFormat(preset.format);
    setRawText(preset.data);
    setErrorMsg(null);
    try {
      const parsed = parseTextDataset(preset.data, preset.format);
      onDatasetLoaded(parsed);
    } catch (e: any) {
      setErrorMsg(e.message);
    }
  };

  const handleParse = () => {
    try {
      setErrorMsg(null);
      const parsed = parseTextDataset(rawText, format);
      onDatasetLoaded(parsed);
    } catch (e: any) {
      setErrorMsg(e.message || 'Ошибка парсинга. Проверьте правильность формата.');
    }
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 space-y-6">
      
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-md bg-teal-500/10 text-teal-400">
              <Sparkles className="w-4 h-4" />
            </span>
            <h2 className="text-lg font-bold text-slate-100 font-sans">Шаг 1: Конструктор Датасета</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Загрузите свои данные для обучения. Тексты преобразуются в частотные векторы, числа нормируются.
          </p>
        </div>

        {/* Format Selectors */}
        <div className="flex gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            type="button"
            onClick={() => {
              setFormat('text');
              setRawText(PRESETS.sentiment.data);
            }}
            className={`px-3 py-1.5 rounded-md font-medium transition cursor-pointer ${
              format === 'text' 
                ? 'bg-teal-500/10 text-teal-300 border border-teal-500/10' 
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            Произвольный Текст (Sentiment)
          </button>
          <button
            type="button"
            onClick={() => {
              setFormat('tabular');
              setRawText(PRESETS.credit.data);
            }}
            className={`px-3 py-1.5 rounded-md font-medium transition cursor-pointer ${
              format === 'tabular' 
                ? 'bg-teal-500/10 text-teal-300 border border-teal-500/10' 
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            Таблица CSV / Числа
          </button>
        </div>
      </div>

      {/* Preset cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {Object.entries(PRESETS).map(([key, preset]) => (
          <button
            key={key}
            type="button"
            onClick={() => handlePresetSelect(key as keyof typeof PRESETS)}
            className="flex flex-col text-left p-3.5 bg-slate-950/60 hover:bg-slate-950 border border-slate-800/80 hover:border-slate-700 rounded-xl transition duration-200 cursor-pointer group"
          >
            <div className="flex items-center gap-2 mb-1.5">
              {preset.format === 'text' ? (
                <Upload className="w-3.5 h-3.5 text-teal-400 group-hover:scale-110 transition" />
              ) : (
                <FileSpreadsheet className="w-3.5 h-3.5 text-yellow-400 group-hover:scale-110 transition" />
              )}
              <span className="text-xs font-bold text-slate-200">{preset.title}</span>
            </div>
            <p className="text-[10px] text-slate-400 leading-relaxed truncate-3-lines">{preset.description}</p>
          </button>
        ))}
      </div>

      {/* Internet Dataset Link Downloader */}
      <div className="bg-slate-950/40 p-4.5 rounded-xl border border-slate-850 space-y-3">
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-slate-300 font-mono uppercase tracking-wider">
            <Globe className="w-4 h-4 text-teal-400 animate-pulse" />
            Закачка датасета по ссылке из интернета
          </div>
          <span className="text-[10px] text-slate-500 font-mono">Стриминг и парсинг в память</span>
        </div>

        <div className="flex flex-col sm:flex-row gap-2">
          <div className="flex-1 flex gap-2 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden px-3 items-center">
            <input
              type="url"
              value={downloadUrl}
              onChange={(e) => setDownloadUrl(e.target.value)}
              placeholder="Вставьте прямую ссылку на RAW .csv или .txt файл..."
              className="w-full bg-transparent border-none text-xs text-slate-200 py-2 focus:outline-none placeholder:text-slate-600 font-mono"
            />
          </div>

          <div className="flex gap-2">
            <select
              value={downloadFormat}
              onChange={(e) => setDownloadFormat(e.target.value as 'text' | 'tabular')}
              className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-350 focus:outline-none"
            >
              <option value="tabular">CSV (Числа)</option>
              <option value="text">TXT (Тексты NLP)</option>
            </select>

            <button
              type="button"
              onClick={handleDownload}
              disabled={isDownloading || !downloadUrl}
              className="bg-teal-500 hover:bg-teal-400 text-slate-950 disabled:opacity-40 font-bold px-4.5 py-1.5 rounded-lg text-xs transition duration-150 active:scale-95 cursor-pointer flex items-center justify-center gap-1.5 shrink-0"
            >
              {isDownloading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Загрузка...
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5" /> Закачать
                </>
              )}
            </button>
          </div>
        </div>

        {/* Templates quick selector */}
        <div className="flex flex-wrap gap-2 items-center text-[10px]">
          <span className="text-slate-500 font-mono">Примеры RAW-датасетов:</span>
          <button
            type="button"
            onClick={() => {
              setDownloadUrl('https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv');
              setDownloadFormat('tabular');
            }}
            className="bg-slate-950 hover:bg-slate-900 border border-slate-850 hover:border-slate-800 text-slate-400 px-2.5 py-1 rounded transition cursor-pointer font-mono"
          >
            💵 Классификация Банкнот (CSV)
          </button>
          <button
            type="button"
            onClick={() => {
              setDownloadUrl('https://gist.githubusercontent.com/ready-ai/db3f9bfdbf9b/raw/sample_sentiment.txt');
              setDownloadFormat('text');
            }}
            className="bg-slate-950 hover:bg-slate-900 border border-slate-850 hover:border-slate-800 text-slate-400 px-2.5 py-1 rounded transition cursor-pointer font-mono"
          >
            💬 Анализ Эмоций Текстов (TXT)
          </button>
        </div>
      </div>

      {/* Text Area Input */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide font-mono flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
            Редактор Сырых Данных
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            {format === 'text' ? 'Формат: Текст предложения | Метка (positive / negative)' : 'Формат: Колонки через запятую, последняя — цель'}
          </span>
        </div>

        <textarea
          value={rawText}
          onChange={(e) => setRawText(e.target.value)}
          className="w-full h-44 bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs font-mono text-slate-300 focus:outline-none focus:border-teal-500 transition focus:ring-1 focus:ring-teal-500/20 shadow-inner"
          placeholder={
            format === 'text'
              ? 'Я обожаю эту программу | positive\nКод не работает и виснет | negative'
              : 'Фича1, Фича2, РезультатY\n10, 4.5, 1\n4, 1.2, 0'
          }
        />

        {errorMsg && (
          <div className="flex items-center gap-2 p-3 bg-red-950/40 border border-red-500/20 text-red-400 rounded-lg text-xs leading-relaxed">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        <button
          type="button"
          onClick={handleParse}
          className="w-full bg-slate-950 hover:bg-slate-900 text-slate-300 hover:text-white border border-slate-800 hover:border-slate-700 font-bold text-xs py-2.5 rounded-lg active:scale-[0.99] transition duration-150 cursor-pointer uppercase tracking-wider font-mono flex items-center justify-center gap-1.5"
        >
          Применить и Векторизовать
        </button>
      </div>

      {/* Numerical Matrix Visualization */}
      {currentDataset && (
        <div className="pt-4 border-t border-slate-800/80 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold text-teal-400 uppercase tracking-wide font-mono flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5" />
              Трансляция в Тензорную Матрицу (X, y)
            </h3>
            <span className="text-[10px] bg-slate-950 border border-slate-800/80 text-slate-400 py-0.5 px-2 rounded-full font-mono">
              Длина вектора фичей: {currentDataset.X[0]?.length || 0} | Строк: {currentDataset.X.length}
            </span>
          </div>

          {/* Table representing actual mathematical tensors ready for NN */}
          <div className="overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950/60">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 font-mono text-[10px] uppercase">
                  <th className="py-2.5 px-4 font-semibold text-center border-r border-slate-800">Индекс Row</th>
                  <th className="py-2.5 px-4 font-semibold text-center border-r border-slate-800">Вектор Фичей (X)</th>
                  <th className="py-2.5 px-4 font-semibold text-center">Целевой вектор (y)</th>
                </tr>
              </thead>
              <tbody className="font-mono text-[11px] text-slate-300 divide-y divide-slate-850">
                {currentDataset.X.slice(0, 5).map((rowX, rowIdx) => (
                  <tr key={rowIdx} className="hover:bg-slate-900/40 transition">
                    <td className="py-2 px-4 text-center border-r border-slate-800 text-slate-500 font-medium">{rowIdx}</td>
                    <td className="py-2 px-4 border-r border-slate-800">
                      <div className="flex flex-wrap gap-1 max-w-xl">
                        <span className="text-slate-500 font-bold">[</span>
                        {rowX.map((val, colIdx) => (
                          <span key={colIdx} title={currentDataset.headers[colIdx]} className="bg-slate-900 border border-slate-800/60 rounded px-1.5 py-0.5 text-teal-400 font-medium">
                            {val.toFixed(3)}
                          </span>
                        ))}
                        <span className="text-slate-500 font-bold">]</span>
                      </div>
                    </td>
                    <td className="py-2 px-4 text-center font-bold">
                      <span className={`px-2 py-0.5 rounded-md ${
                        currentDataset.y[rowIdx] === 1 
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/10' 
                          : currentDataset.y[rowIdx] === 0 
                            ? 'bg-red-500/10 text-red-400 border border-red-500/10' 
                            : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/10'
                      }`}>
                        {currentDataset.y[rowIdx].toFixed(4)}
                      </span>
                    </td>
                  </tr>
                ))}
                {currentDataset.X.length > 5 && (
                  <tr>
                    <td colSpan={3} className="py-2.5 px-4 text-center text-slate-500 italic bg-slate-950/30">
                      ... и ещё {currentDataset.X.length - 5} строк скрыто в превью для экономии места ...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Tokens detail overlay for NLP */}
          {currentDataset.vocab && (
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide font-mono">Словарь токенов (Word Vocab):</h4>
              <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
                {Object.entries(currentDataset.vocab).map(([word, id]) => (
                  <span key={word} className="bg-slate-900 text-slate-400 border border-slate-800 text-[10px] px-2 py-1 rounded-md font-mono flex gap-1">
                    <span className="text-slate-500">#{id}:</span>
                    <strong className="text-teal-400">"{word}"</strong>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
