/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Download, FileCode, Check, Copy } from 'lucide-react';
import { Dataset, NetworkState, ActivationType } from '../types';
import { generateExporthtml } from '../utils/mathEngine';

interface ExportPanelProps {
  dataset: Dataset | null;
  network: NetworkState | null;
  activation: ActivationType;
  hiddenLayersLength: number[];
  modelName: string;
}

export function ExportPanel({ dataset, network, activation, hiddenLayersLength, modelName }: ExportPanelProps) {
  const [downloaded, setDownloaded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [appDownloaded, setAppDownloaded] = useState(false);

  if (!dataset || !network) return null;

  const codeHtml = generateExporthtml(dataset, network, activation, hiddenLayersLength, modelName);

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
      
      setAppDownloaded(true);
      setTimeout(() => setAppDownloaded(false), 3000);
    } catch (e) {
      console.error('Ошибка скачивания приложения:', e);
    }
  };

  const handleDownload = () => {
    try {
      const blob = new Blob([codeHtml], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'ai-mathematical-model.html';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 2500);
    } catch (e) {
      console.error('Ошибка записи файла экспорта:', e);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(codeHtml);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 space-y-6">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-md bg-teal-500/10 text-teal-400">
              <FileCode className="w-4 h-4" />
            </span>
            <h2 className="text-lg font-bold text-slate-100 font-sans">Шаг 5: Экспорт Автономного ИИ</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Пакетный экспорт в один HTML/JS-файл. Перенесите готовую обученную модель в любой браузер без внешних серверов.
          </p>
        </div>

        {/* Copy/Download triggers */}
        <div className="flex flex-wrap gap-2 w-full sm:w-auto justify-end">
          <button
            type="button"
            onClick={handleDownloadWholeConstructor}
            className={`flex-1 sm:flex-initial font-bold px-4 py-2.5 rounded-xl text-xs flex items-center justify-center gap-1.5 transition active:scale-95 cursor-pointer font-mono ${
              appDownloaded
                ? 'bg-purple-950 text-purple-400 border border-purple-900/60'
                : 'bg-purple-900/20 hover:bg-purple-900/30 text-purple-300 border border-purple-900/40 hover:border-purple-900/60'
            }`}
            title="Скачать весь этот интерактивный конструктор как один полностью автономный HTML файл (Open Source)"
          >
            {appDownloaded ? (
              <>
                <Check className="w-3.5 h-3.5 text-purple-400" /> КОНСТРУКТОР СКАЧАН!
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" /> СКАЧАТЬ ВЕСЬ КОНСТРУКТОР (1 HTML)
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleCopy}
            className="flex-1 sm:flex-initial bg-slate-950 hover:bg-slate-900 text-slate-350 border border-slate-800 hover:border-slate-700 font-bold px-4 py-2.5 rounded-xl text-xs flex items-center justify-center gap-1.5 transition active:scale-95 cursor-pointer font-mono"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" /> СКОПИРОВАНО
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" /> КОПИРОВАТЬ КОД
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleDownload}
            className="flex-1 sm:flex-initial bg-gradient-to-r from-teal-500 to-emerald-600 hover:opacity-90 text-slate-950 font-bold px-5 py-2.5 rounded-xl text-xs flex items-center justify-center gap-1.5 transition active:scale-95 cursor-pointer font-mono"
          >
            {downloaded ? (
              <>
                <Check className="w-4 h-4 text-emerald-950" /> СОХРАНЕНО!
              </>
            ) : (
              <>
                <Download className="w-4 h-4" /> СКАЧАТЬ HTML-ИИ
              </>
            )}
          </button>
        </div>
      </div>

      {/* Standout Export Value Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850/80 space-y-2">
          <h3 className="text-xs font-bold text-teal-400 code-font uppercase">Что внутри экспортированного модуля?</h3>
          <ul className="text-slate-450 text-[11px] leading-relaxed list-disc list-inside space-y-1">
            <li><strong className="text-slate-200">Параметрическая матрица</strong> (веса и смещения всех слоев).</li>
            <li><strong className="text-slate-200">Спецификация токенизации</strong> (прямое преобразования текста во фреймы).</li>
            <li><strong className="text-slate-200">Offline-Inference код</strong> (чистый JS форвардинга без зависимостей).</li>
            <li><strong className="text-slate-200">Отзывчивый UI интерфейс</strong> (форма ввода, весовые бары и логи).</li>
          </ul>
        </div>

        <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-850/80 flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold text-yellow-500 code-font uppercase mb-1">Принцип Независимости / Без API</h3>
            <p className="text-[10.5px] leading-relaxed text-slate-400">
              Этот файл полностью самодостаточен. Ему не нужен доступ к Интернету, серверам или API-ключам Google/OpenAI. Запуск математических расчетов происходит напрямую в движке V8 вашего браузера.
            </p>
          </div>
          <span className="text-[9.5px] text-slate-500 font-mono mt-3">
            Вес скомпилированного файла: ~15 КБ.
          </span>
        </div>

      </div>

    </div>
  );
}
