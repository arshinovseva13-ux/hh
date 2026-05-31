/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, Key, HelpCircle, HardDrive, RefreshCw, Cpu, Brain, Check, Play, 
  AlertCircle, MessageSquare, Send, BookOpen, Layers, Zap, Pause, Award
} from 'lucide-react';
import { Dataset, ModelConfig, NetworkState } from '../types';
import { PRESETS } from './DatasetUploader';

interface AiAssistantProps {
  config: ModelConfig;
  onChangeConfig: (newConfig: ModelConfig) => void;
  dataset: Dataset | null;
  network: NetworkState | null;
  onStartTraining: () => void;
  onPauseTraining: () => void;
  onResetNN: () => void;
  onLoadGeneratedDataset: (rawText: string, format: 'text' | 'tabular') => void;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
}

export function AiAssistant({
  config,
  onChangeConfig,
  dataset,
  network,
  onStartTraining,
  onPauseTraining,
  onResetNN,
  onLoadGeneratedDataset
}: AiAssistantProps) {
  // Tabs: 'chat', 'actions', 'tutorial', 'api'
  const [activeTab, setActiveTab] = useState<'chat' | 'actions' | 'tutorial' | 'api'>('chat');
  
  // Custom API configurations
  const [provider, setProvider] = useState<'disabled' | 'gemini' | 'openai' | 'custom'>(config.externalAiType);
  const [apiKey, setApiKey] = useState(config.externalAiKey);
  const [customUrl, setCustomUrl] = useState(config.externalAiUrl || 'http://localhost:11434/v1/chat/completions');
  const [targetModel, setTargetModel] = useState(config.externalModelName || 'gemini-3.5-flash');
  
  // Interactive Chat State
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: 'Привет! Я твой интерактивный Копилот и гид по нейросетям. 🧠\n\nЯ умею объяснять теорию своими словами и готов САМ настроить калькулятор весов за тебя! Выбери вкладку "⚡ ДЕЙСТВИЯ ИИ", чтобы поручить мне конфигурацию, или задай любой вопрос об обучении нейросетей.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const [loading, setLoading] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const saveSettings = () => {
    onChangeConfig({
      ...config,
      externalAiType: provider,
      externalAiKey: apiKey,
      externalAiUrl: customUrl,
      externalModelName: targetModel
    });
    setSuccessMsg('Настройки связи с ИИ успешно обновлены!');
    setTimeout(() => setSuccessMsg(null), 3000);
  };

  const handleProviderChange = (newProvider: 'disabled' | 'gemini' | 'openai' | 'custom') => {
    setProvider(newProvider);
    if (newProvider === 'gemini') {
      setTargetModel('gemini-3.5-flash');
      setCustomUrl('');
    } else if (newProvider === 'openai') {
      setTargetModel('gpt-4o-mini');
      setCustomUrl('https://api.openai.com/v1/chat/completions');
    } else if (newProvider === 'custom') {
      setTargetModel('llama3');
      setCustomUrl('http://localhost:11434/v1/chat/completions');
    }
  };

  // Pre-compiled comprehensive educational responses in Russian if offline
  const EDUCATIONAL_QA: Record<string, string> = {
    'weights': '🛠️ **Синаптические веса (Weights)** — это числовые коэффициенты связи между нейронами. Они определяют «важность» признаков. Когда сеть обучается с помощью метода обратного распространения ошибки, она шаг за шагом корректирует эти веса, чтобы минимизировать погрешность предсказания. Увеличение веса делает связь сильнее, а отрицательный вес — инвертирует сигнал.',
    'bias': '⚖️ **Смещение (Bias)** — это дополнительный весомый параметр нейрона, независимый от входных признаков. Он позволяет сдвинуть график функции активации влево или вправо, чтобы нейрон мог срабатывать (активироваться) даже при нулевых сигналах на входе.',
    'backprop': '🔄 **Метод обратного распространения (Backpropagation)** — это основа обучения. Модель сравнивает свой результат со значением учителя (целевым вектором), получая ошибку MSE. Затем этот сигнал ошибки «прогоняется» в обратном направлении от выхода ко входу. Используя производные функций, алгоритм рассчитывает градиент и плавно меняет веса.',
    'activation': '📈 **Активация (Activation)** задает математическое сито. Без активации нейросеть превратится в простую линейную регрессию, неспособную понимать нелинейные паттерны. **Sigmoid** сжимает значения от 0 до 1 (идеально для бинарного разделения), а **ReLU** зануляет всё, что меньше нуля, предотвращая насыщение.',
    'hidden': '🧠 **Скрытые слои (Hidden Layers)** извлекают абстрактные фичи. Первый слой видит простые линии или отдельные слова, второй сочетает их в формы или предложения. Чем сложнее и нелинейнее задача, тем больше нейронов в скрытом слое потребуется!',
    'underfitting': '🔴 **Ошибка велика и не падает?** Это недообучение. Попробуйте: 1) Отрегулировать градиент и увеличить скорость обучения (Learning Rate), 2) Добавить нейроны в скрытые слои, 3) Увеличить количество эпох обучения до 400.'
  };

  // Perform prompt execution inside the chat
  const handleSendMessage = async (textToSend: string) => {
    if (!textToSend.trim()) return;

    const userMsg: ChatMessage = {
      id: Math.random().toString(),
      sender: 'user',
      text: userMsgTextCleaning(textToSend),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setChatInput('');
    setLoading(true);
    setErrorMsg(null);

    // 1. Check if we correspond to predefined interactive buttons
    const queryLower = textToSend.toLowerCase();
    let matchedOfflineAnswer = '';
    
    if (queryLower.includes('вес') || queryLower.includes('weight')) {
      matchedOfflineAnswer = EDUCATIONAL_QA.weights;
    } else if (queryLower.includes('смещен') || queryLower.includes('bias')) {
      matchedOfflineAnswer = EDUCATIONAL_QA.bias;
    } else if (queryLower.includes('backpropagation') || queryLower.includes('обратн') || queryLower.includes('как учится')) {
      matchedOfflineAnswer = EDUCATIONAL_QA.backprop;
    } else if (queryLower.includes('активац') || queryLower.includes('activation')) {
      matchedOfflineAnswer = EDUCATIONAL_QA.activation;
    } else if (queryLower.includes('скрыт') || queryLower.includes('слой') || queryLower.includes('архитект')) {
      matchedOfflineAnswer = EDUCATIONAL_QA.hidden;
    } else if (queryLower.includes('настрой') || queryLower.includes('ошибк') || queryLower.includes('не учит')) {
      matchedOfflineAnswer = EDUCATIONAL_QA.underfitting;
    }

    // 2. If API is not configured or offline is preferred
    if (provider === 'disabled' || !apiKey) {
      setTimeout(() => {
        let finalReply = '';
        if (matchedOfflineAnswer) {
          finalReply = matchedOfflineAnswer;
        } else {
          finalReply = `🤖 **Информационный Спутник**: Я распознал ваш вопрос! Наш браузерный симулятор использует математически точную реализацию **Feedforward-Forward** и **SGD Backpropagation** на чистом TypeScript.

Для получения персональных советов по вашим данным вы можете настроить API-ключ в сопредельной вкладке настроек ИИ. На данный момент рекомендую попробовать вкладку **"⚡ ДЕЙСТВИЯ ИИ"** — там я могу перестроить архитектуру нейросети в один клик!`;
        }

        const replyMsg: ChatMessage = {
          id: Math.random().toString(),
          sender: 'assistant',
          text: finalReply,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setMessages(prev => [...prev, replyMsg]);
        setLoading(false);
      }, 800);
      return;
    }

    // 3. Real External Model Call
    try {
      let prompt = `Ты — профессиональный ИИ-Спутник и Копилот в интерактивном визуализаторе нейронных сетей "Конструктор ИИ Весов". 
Пользователь спрашивает: "${textToSend}".

Дай понятный, супер-краткий (до 5 предложений), профессиональный математический ответ на русском языке. Будь дружелюбен, используй логичные смайлики и объясняй теорию простыми терминами (нейроны, градиенты, минимизация дельты).`;

      let responseText = '';

      if (provider === 'gemini') {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${targetModel}:generateContent?key=${apiKey}`;
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }]
          })
        });
        
        if (!res.ok) {
          throw new Error(`Ошибка от Gemini API (Статус ${res.status})`);
        }
        const data = await res.json();
        responseText = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
      } else {
        const endpoint = customUrl || 'https://api.openai.com/v1/chat/completions';
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model: targetModel,
            messages: [{ role: 'user', content: prompt }]
          })
        });

        if (!res.ok) {
          throw new Error(`Ошибка API (Статус ${res.status})`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';
      }

      const replyMsg: ChatMessage = {
        id: Math.random().toString(),
        sender: 'assistant',
        text: responseText.trim(),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, replyMsg]);

    } catch (e: any) {
      console.error(e);
      const errReply: ChatMessage = {
        id: Math.random().toString(),
        sender: 'assistant',
        text: `⚠️ Не удалось связаться с внешней моделью ИИ. Причина: ${e.message || 'CORS-блокировка или неверный ключ'}.\n\nНо не переживайте! Я временно включил базового оффлайн-помощника: \n\n${matchedOfflineAnswer || 'Рекомендую отрегулировать слои в конструкторе или загрузить готовый пресет из каталога выше.'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errReply]);
    } finally {
      setLoading(false);
    }
  };

  const userMsgTextCleaning = (text: string) => {
    return text.trim();
  };

  // Perform Direct AI Action inside the App (STATE ALTERING)
  const executeAiAction = (actionKey: string) => {
    setErrorMsg(null);
    let feedback = '';

    switch (actionKey) {
      case 'setup_sentiment':
        onChangeConfig({
          ...config,
          modelName: 'Критик Отзывов ИИ',
          activation: 'sigmoid',
          learningRate: 0.18,
          epochs: 450,
          hiddenLayers: [{ id: 'h1', neurons: 4 }]
        });
        onLoadGeneratedDataset(PRESETS.sentiment.data, 'text');
        feedback = '🚀 ИИ настроил сеть под **Классификатор Отзывов (NLP)**. Загружен текстовый корпус, выбрана функция **Sigmoid**, установлен шаг градиента **0.18** и один стабильный скрытый слой [4 нейрона].';
        break;

      case 'setup_credit':
        onChangeConfig({
          ...config,
          modelName: 'ИИ Скоринг Кредита',
          activation: 'relu',
          learningRate: 0.08,
          epochs: 400,
          hiddenLayers: [
            { id: 'h1', neurons: 5 },
            { id: 'h2', neurons: 3 }
          ]
        });
        onLoadGeneratedDataset(PRESETS.credit.data, 'tabular');
        feedback = '💳 ИИ применил настройки для датасета **Логистика Кредитов**. Загружены показатели скоринга, выбран метод активации **ReLU**, слоев: **2 слоя [5 ➔ 3]** для глубокого вычисления финансовых скрытых признаков.';
        break;

      case 'setup_parabola':
        onChangeConfig({
          ...config,
          modelName: 'ИИ Тригонометрист',
          activation: 'relu',
          learningRate: 0.12,
          epochs: 350,
          hiddenLayers: [{ id: 'h1', neurons: 3 }]
        });
        onLoadGeneratedDataset(PRESETS.parabola.data, 'tabular');
        feedback = '📉 ИИ настроил сеть под **Математический Синус**. Применен волновой нелинейный регрессионный паттерн с оптимизатором **ReLU** и скрытым слоем из **[3]** нейронов.';
        break;

      case 'dnn_monster':
        onChangeConfig({
          ...config,
          hiddenLayers: [
            { id: 'h1', neurons: 6 },
            { id: 'h2', neurons: 4 },
            { id: 'h3', neurons: 3 }
          ]
        });
        feedback = '🧠 ИИ усложнил архитектуру до структуры глубокого обучения **[6 ➔ 4 ➔ 3]** нейронов! Инициализировано **43 синаптических связи**.';
        break;

      case 'simplify_net':
        onChangeConfig({
          ...config,
          hiddenLayers: [{ id: 'h1', neurons: 2 }]
        });
        feedback = '🧬 ИИ упростил модель до скромной емкости скрытого слоя **[2]** нейрона. Это идеально для изучения простых линейно-разделимых поверхностей.';
        break;

      case 'trigger_train':
        onStartTraining();
        feedback = '⚡ ИИ успешно отдал команду на запуск SGD-обучения! Модель начала перерасчет матриц весов.';
        break;

      case 'trigger_pause':
        onPauseTraining();
        feedback = '⏸️ ИИ временно приостановил расчеты. Вы можете детально изучить застывшие веса на SVG-карте синапсов ниже.';
        break;

      case 'trigger_reset':
        onResetNN();
        feedback = '🔄 ИИ заново сгенерировал случайное семя (Seed) весов и обнулил историю ошибок MSE!';
        break;

      default:
        feedback = 'Выбрано неопознанное действие.';
    }

    setActionFeedback(feedback);
    setTimeout(() => setActionFeedback(null), 8500);

    // Also inject custom bot message detailing the action
    const actionBotMsg: ChatMessage = {
      id: Math.random().toString(),
      sender: 'assistant',
      text: feedback,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, actionBotMsg]);
  };

  return (
    <div id="aiAssistantWrapper" className="bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-5 space-y-5">
      
      {/* Header and Brand */}
      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-purple-500/10 text-purple-400">
            <Sparkles className="w-4.5 h-4.5" />
          </span>
          <div>
            <h2 className="text-sm font-bold text-slate-100 font-sans tracking-wide">ИИ Интерактивный Спутник-Копилот</h2>
            <p className="text-[10px] text-slate-500">Обучает, настраивает и запускает математику за вас</p>
          </div>
        </div>
        
        {/* Navigation Tabs */}
        <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-850 text-[10.5px]">
          <button
            type="button"
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1 rounded-lg transition cursor-pointer font-medium flex items-center gap-1 ${
              activeTab === 'chat' ? 'bg-purple-500/10 text-purple-300 border border-purple-505/20' : 'text-slate-450 hover:text-slate-200'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            Чат
          </button>
          
          <button
            type="button"
            onClick={() => setActiveTab('actions')}
            className={`px-3 py-1 rounded-lg transition cursor-pointer font-medium flex items-center gap-1 ${
              activeTab === 'actions' ? 'bg-teal-500/10 text-teal-300 border border-teal-505/20' : 'text-slate-450 hover:text-slate-200'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            Действия
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('tutorial')}
            className={`px-3 py-1 rounded-lg transition cursor-pointer font-medium flex items-center gap-1 ${
              activeTab === 'tutorial' ? 'bg-yellow-500/10 text-yellow-300' : 'text-slate-450 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Гид
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('api')}
            className={`px-2.5 py-1 rounded-lg transition cursor-pointer font-mono font-bold text-[9.5px] ${
              activeTab === 'api' ? 'bg-slate-900 text-slate-100 border border-slate-800' : 'text-slate-500 hover:text-slate-400'
            }`}
          >
            API
          </button>
        </div>
      </div>

      {actionFeedback && (
        <div className="p-3 bg-teal-500/15 border border-teal-500/30 text-teal-300 text-xs rounded-xl flex items-start gap-2.5 leading-relaxed animate-fade-in animate-pulse">
          <Zap className="w-4 h-4 text-teal-400 shrink-0 mt-0.5" />
          <span>{actionFeedback}</span>
        </div>
      )}

      {/* Tab Contents */}

      {/* Tab 1: COMPREHENSIVE CONVERSATIONAL CHAT */}
      {activeTab === 'chat' && (
        <div className="space-y-4">
          <div className="bg-slate-950 rounded-xl border border-slate-850/80 overflow-hidden flex flex-col h-64">
            
            {/* Scrollable chat body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
              {messages.map((m) => (
                <div 
                  key={m.id} 
                  className={`flex flex-col max-w-[85%] ${m.sender === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'}`}
                >
                  <div className={`p-3 rounded-2xl text-xs leading-relaxed whitespace-pre-wrap ${
                    m.sender === 'user' 
                      ? 'bg-purple-600 text-slate-950 font-bold rounded-tr-none' 
                      : 'bg-slate-900 border border-slate-800 text-slate-250 rounded-tl-none'
                  }`}>
                    {m.text}
                  </div>
                  <span className="text-[9px] text-slate-600 font-mono mt-1 px-1">{m.timestamp}</span>
                </div>
              ))}
              
              {loading && (
                <div className="mr-auto items-start max-w-[85%] flex flex-col animate-pulse">
                  <div className="bg-slate-900 border border-slate-800 p-3 rounded-2xl rounded-tl-none text-xs text-slate-500 flex items-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Копилот подбирает математический ответ...
                  </div>
                </div>
              )}
              
              <div ref={chatBottomRef} />
            </div>

            {/* Quick topics recommendation list */}
            <div className="border-t border-slate-900 bg-slate-950/80 px-3 py-2 flex gap-2 overflow-x-auto whitespace-nowrap scrollbar-hide">
              <span className="text-[9px] text-slate-500 self-center font-mono uppercase mr-1">Шпаргалки:</span>
              <button 
                type="button" 
                onClick={() => handleSendMessage('Что такое Синаптические Веса?')}
                className="bg-slate-900 hover:bg-slate-850 border border-slate-800 rounded px-2 py-1 text-[10px] text-slate-350 cursor-pointer font-sans transition"
              >
                🛠️ Что такое Веса?
              </button>
              <button 
                type="button" 
                onClick={() => handleSendMessage('Как работает Backpropagation?')}
                className="bg-slate-900 hover:bg-slate-850 border border-slate-800 rounded px-2 py-1 text-[10px] text-slate-350 cursor-pointer font-sans transition"
              >
                🔄 Что такое Backprop?
              </button>
              <button 
                type="button" 
                onClick={() => handleSendMessage('Зачем нужны скрытые слои нейронов?')}
                className="bg-slate-900 hover:bg-slate-850 border border-slate-800 rounded px-2 py-1 text-[10px] text-slate-350 cursor-pointer font-sans transition"
              >
                🧠 Роль скрытых слоев
              </button>
              <button 
                type="button" 
                onClick={() => handleSendMessage('Почему у меня высокая ошибка MSE и сеть не учится?')}
                className="bg-slate-900 hover:bg-slate-850 border border-slate-800 rounded px-2 py-1 text-[10px] text-slate-350 cursor-pointer font-sans transition"
              >
                🔴 Высокая ошибка (MSE)
              </button>
            </div>
          </div>

          {/* Prompt sending input */}
          <form 
            onSubmit={(e) => { e.preventDefault(); handleSendMessage(chatInput); }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Спроси Копилота: 'Что такое смещение?' или 'Какая роль у функции Sigmoid'..."
              className="flex-1 bg-slate-950 border border-slate-850 rounded-xl px-4 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-purple-500 font-sans"
            />
            <button
              type="submit"
              disabled={!chatInput.trim()}
              className="px-4 py-2.5 bg-purple-600 hover:bg-purple-550 disabled:opacity-40 text-slate-950 rounded-xl transition cursor-pointer flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      )}

      {/* Tab 2: CO-PILOT ACTIVE ACTIONS PANEL */}
      {activeTab === 'actions' && (
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest font-mono border-b border-slate-850 pb-2">
            Центр уполномоченных ИИ-Действий
          </h3>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Нажмите на любую команду, и ИИ перенастроит параметры веб-приложения за вас в реальном времени. Наблюдайте за мгновенным изменением схемы связей!
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            
            {/* Action 1 */}
            <button
              type="button"
              onClick={() => executeAiAction('setup_sentiment')}
              className="p-3 bg-slate-950 text-left border border-slate-850 hover:border-slate-800 hover:bg-[#0c111c] rounded-xl transition duration-150 cursor-pointer flex gap-3 group relative overflow-hidden"
            >
              <div className="w-8 h-8 rounded-lg bg-teal-500/10 text-teal-400 flex items-center justify-center shrink-0 self-center">
                <MessageSquare className="w-4 h-4" />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-slate-200 group-hover:text-teal-350 transition">
                  Настроить под Текст (Спам)
                </div>
                <div className="text-[10px] text-slate-500">
                  Sigmoid + Слой [4] + Литературный текст
                </div>
              </div>
            </button>

            {/* Action 2 */}
            <button
              type="button"
              onClick={() => executeAiAction('setup_credit')}
              className="p-3 bg-slate-950 text-left border border-slate-850 hover:border-slate-800 hover:bg-[#0c111c] rounded-xl transition duration-150 cursor-pointer flex gap-3 group relative overflow-hidden"
            >
              <div className="w-8 h-8 rounded-lg bg-yellow-500/10 text-yellow-400 flex items-center justify-center shrink-0 self-center">
                <Layers className="w-4 h-4" />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-slate-200 group-hover:text-yellow-350 transition">
                  Напустить Кредитный Скоринг
                </div>
                <div className="text-[10px] text-slate-500">
                  ReLU + Слои [5 ➔ 3] + Финансовый рейтинг
                </div>
              </div>
            </button>

            {/* Action 3 */}
            <button
              type="button"
              onClick={() => executeAiAction('setup_parabola')}
              className="p-3 bg-slate-950 text-left border border-slate-850 hover:border-slate-800 hover:bg-[#0c111c] rounded-xl transition duration-150 cursor-pointer flex gap-3 group relative overflow-hidden"
            >
              <div className="w-8 h-8 rounded-lg bg-orange-500/10 text-orange-400 flex items-center justify-center shrink-0 self-center">
                <Zap className="w-4 h-4" />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-slate-200 group-hover:text-orange-350 transition">
                  Аппроксимировать Математический Синус
                </div>
                <div className="text-[10px] text-slate-500">
                  ReLU + Слой [3] + Тригонометрия
                </div>
              </div>
            </button>

            {/* Action 4 */}
            <button
              type="button"
              onClick={() => executeAiAction('dnn_monster')}
              className="p-3 bg-slate-950 text-left border border-slate-850 hover:border-slate-800 hover:bg-[#0c111c] rounded-xl transition duration-150 cursor-pointer flex gap-3 group relative overflow-hidden"
            >
              <div className="w-8 h-8 rounded-lg bg-purple-500/10 text-purple-400 flex items-center justify-center shrink-0 self-center">
                <Brain className="w-4 h-4" />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-slate-200 group-hover:text-purple-350 transition">
                  Углубить Сеть [6 ➔ 4 ➔ 3]
                </div>
                <div className="text-[10px] text-slate-500">
                  Построить мощную DNN сеть-монстр
                </div>
              </div>
            </button>

            {/* Action 5 */}
            <button
              type="button"
              onClick={() => executeAiAction('simplify_net')}
              className="p-3 bg-slate-950 text-left border border-slate-850 hover:border-slate-800 hover:bg-[#0c111c] rounded-xl transition duration-150 cursor-pointer flex gap-3 group relative overflow-hidden"
            >
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center shrink-0 self-center">
                <Award className="w-4 h-4" />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-slate-200 group-hover:text-emerald-350 transition">
                  Упростить Нейросеть до [2]
                </div>
                <div className="text-[10px] text-slate-500">
                  Минимизировать объем вычислений
                </div>
              </div>
            </button>

            {/* Action 6: Training trigger */}
            <button
              type="button"
              onClick={() => executeAiAction('trigger_train')}
              className="p-3 bg-slate-950 text-left border border-slate-850 hover:border-slate-800 hover:bg-[#0c111c] rounded-xl transition duration-150 cursor-pointer flex gap-3 group relative overflow-hidden"
            >
              <div className="w-8 h-8 rounded-lg bg-red-500/10 text-rose-400 flex items-center justify-center shrink-0 self-center">
                <Play className="w-4 h-4 animate-ping" />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-rose-300">
                  Мгновенно Начать Обучение
                </div>
                <div className="text-[10px] text-slate-500">
                  Запустить алгоритм обратного спуска
                </div>
              </div>
            </button>

          </div>

          <div className="flex flex-wrap gap-2 pt-1 border-t border-slate-900">
            <button
              type="button"
              onClick={() => executeAiAction('trigger_pause')}
              className="bg-slate-950 hover:bg-slate-900 border border-slate-850 text-[10px] text-slate-400 px-3 py-1.5 rounded transition cursor-pointer flex items-center gap-1.5"
            >
              <Pause className="w-3 h-3" /> Приостановить
            </button>
            <button
              type="button"
              onClick={() => executeAiAction('trigger_reset')}
              className="bg-slate-950 hover:bg-slate-900 border border-slate-850 text-[10px] text-slate-400 px-3 py-1.5 rounded transition cursor-pointer flex items-center gap-1.5"
            >
              <RefreshCw className="w-3 h-3" /> Сбросить веса к нулю
            </button>
          </div>
        </div>
      )}

      {/* Tab 3: STATIC TUTORIAL INFO */}
      {activeTab === 'tutorial' && (
        <div className="bg-slate-950/60 p-4.5 rounded-xl border border-slate-850 space-y-4 text-xs leading-relaxed text-slate-350">
          <h3 className="text-xs font-bold text-yellow-400 uppercase tracking-widest font-mono">Гид: Как обучить ИИ за 4 шага</h3>
          
          <ul className="space-y-3 font-sans list-none pl-0">
            <li className="flex gap-2.5 items-start">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-purple-500 bg-opacity-10 text-purple-400 font-mono text-[10px] shrink-0 mt-0.5 font-black">1</span>
              <div>
                <strong className="text-slate-100">Загрузите Данные:</strong> Вы можете выбрать текстовый sentiment пресет на русском, или числовую таблицу. Модель векторизует слова в матрицу частот (Bag of Words) автоматически.
              </div>
            </li>
            
            <li className="flex gap-2.5 items-start">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-purple-500 bg-opacity-10 text-purple-400 font-mono text-[10px] shrink-0 mt-0.5 font-black">2</span>
              <div>
                <strong className="text-slate-100">Спроектируйте Скрытые Слои:</strong> Добавьте нейроны в Шаге 2. Наблюдайте на динамичной SVG-карте, как рождаются синапсы, соединяющие каждый узел.
              </div>
            </li>

            <li className="flex gap-2.5 items-start">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-purple-500 bg-opacity-10 text-purple-400 font-mono text-[10px] shrink-0 mt-0.5 font-black">3</span>
              <div>
                <strong className="text-slate-100">Запустите Градиентный Спуск:</strong> В панели "Шаг 3" нажмите старт. Веса начнут плавно пульсировать: зеленый цвет (положительный синапс), оранжевый (отрицательный синапс), а график MSE устремится вниз!
              </div>
            </li>

            <li className="flex gap-2.5 items-start">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-purple-500 bg-opacity-10 text-purple-400 font-mono text-[10px] shrink-0 mt-0.5 font-black">4</span>
              <div>
                <strong className="text-slate-100">Тестируйте Модель:</strong> Введите произвольный текст в интерактивной консоли и посмотрите, под какими углами нейросеть складывает веса, чтобы предсказать результат. Выгрузите модель как портативный HTML.
              </div>
            </li>
          </ul>
        </div>
      )}

      {/* Tab 4: API ACCESS MANAGEMENT */}
      {activeTab === 'api' && (
        <div className="bg-slate-950 p-4.5 rounded-xl border border-slate-850/70 space-y-4">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">Настройки связи со сторонними ИИ</h3>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            
            {/* Provider select */}
            <div className="space-y-1.5">
              <label className="block text-[10px] text-slate-450 font-mono uppercase">Сервис ИИ</label>
              <select
                value={provider}
                onChange={(e) => handleProviderChange(e.target.value as any)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg py-1.5 px-3 text-xs text-slate-300 focus:outline-none focus:border-purple-500 cursor-pointer"
              >
                <option value="disabled">🚫 Встроенный Симулятор</option>
                <option value="gemini">✨ Google Gemini b1-API</option>
                <option value="openai">🧠 OpenAI gpt-API</option>
                <option value="custom">🖥️ Custom/Ollama Endpoint</option>
              </select>
            </div>

            {/* Target model */}
            <div className="space-y-1.5">
              <label className="block text-[10px] text-slate-450 font-mono uppercase">Модель</label>
              <input
                type="text"
                value={targetModel}
                onChange={(e) => setTargetModel(e.target.value)}
                placeholder="gemini-3.5-flash / gpt-4o-mini"
                className="w-full bg-slate-900 border border-slate-800 rounded-lg py-1.5 px-3 text-xs text-slate-300 focus:outline-none focus:border-purple-500 font-mono"
              />
            </div>

            {/* Key */}
            <div className="space-y-1.5 sm:col-span-2">
              <label className="block text-[10px] text-slate-450 font-mono uppercase flex items-center gap-1">
                <Key className="w-3.5 h-3.5 text-purple-400" />
                API Ключ (Ваш локальный секрет)
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={provider === 'disabled' ? 'Ключ не требуется для встроенного ИИ' : 'Вставьте ваш AI_API_KEY...'}
                disabled={provider === 'disabled'}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg py-2 px-3 text-xs text-slate-300 focus:outline-none focus:border-purple-500 font-mono disabled:opacity-50"
              />
            </div>

            {/* Custom endpoint */}
            {(provider === 'custom' || provider === 'openai') && (
              <div className="space-y-1.5 sm:col-span-2">
                <label className="block text-[10px] text-slate-450 font-mono uppercase">Базовый API URL</label>
                <input
                  type="text"
                  value={customUrl}
                  onChange={(e) => setCustomUrl(e.target.value)}
                  placeholder="http://localhost:11434/v1/chat/completions"
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg py-1.5 px-3 text-xs text-slate-400 focus:outline-none focus:border-purple-500 font-mono"
                />
              </div>
            )}

          </div>

          <div className="flex justify-between items-center pt-2">
            <span className="text-[10px] text-slate-500">
              *Все сессии сохраняются локально.
            </span>
            <button
              type="button"
              onClick={saveSettings}
              className="bg-purple-600 hover:bg-purple-500 text-slate-950 font-bold px-4 py-2 rounded-lg text-xs transition duration-150 cursor-pointer flex items-center gap-1"
            >
              <Check className="w-3.5 h-3.5" />
              Применить API данные
            </button>
          </div>
        </div>
      )}

      {successMsg && (
        <div className="p-2.5 bg-emerald-500/10 border border-emerald-550/20 text-emerald-400 text-xs rounded-lg flex items-center gap-2">
          <Check className="w-4 h-4 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

    </div>
  );
}
