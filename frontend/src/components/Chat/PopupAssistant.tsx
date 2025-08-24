/**
 * Popup AI Assistant Component
 * Full-featured AI chat interface that appears as a popup
 */

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Badge } from '../ui/badge';
import { cn } from '../../lib/utils';
import {
  Send,
  Bot,
  User,
  Brain,
  Zap,
  X,
  Minimize2,
  Maximize2,
  MessageSquare,
  Copy,
  Download,
  Trash2,
  Sparkles,
} from 'lucide-react';
import { useNLPQuery, useOllamaQuery, useCurrentContext } from '../../hooks/useAPI';

interface PopupMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  engine: 'pattern' | 'ollama';
}

interface PopupAssistantProps {
  isOpen: boolean;
  onClose: () => void;
}

const PopupAssistant: React.FC<PopupAssistantProps> = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState<PopupMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isMinimized, setIsMinimized] = useState(false);
  const [useOllama, setUseOllama] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const nlpMutation = useNLPQuery();
  const ollamaMutation = useOllamaQuery();
  const { data: contextData } = useCurrentContext();

  // Welcome examples grouped by category
  const exampleQueries = {
    "📊 Analytics": [
      "What are the peak delay hours?",
      "Show me delay trends by route",
      "Which airports have the highest delays?",
    ],
    "🛠️ Optimization": [
      "How can I optimize flight schedules?",
      "Which routes need immediate attention?",
      "Best times to schedule flights?",
    ],
    "🔍 Insights": [
      "Analyze Mumbai to Delhi routes",
      "What causes most delays?",
      "Show cascade delay patterns",
    ],
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isOpen]);

  const formatLLMResponse = (response: string): string => {
    if (!response) return 'No response received';
    
    return response
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/•/g, '•')
      .replace(/\n\n/g, '\n')
      .trim();
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: PopupMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
      engine: useOllama ? 'ollama' : 'pattern',
    };

    setMessages(prev => [...prev, userMessage]);
    const queryText = inputValue;
    setInputValue('');

    try {
      if (useOllama && contextData?.models_status.ollama_llm) {
        ollamaMutation.mutate(
          { 
            query: queryText, 
            context: contextData?.context 
          },
          {
            onSuccess: (response) => {
              const assistantMessage: PopupMessage = {
                id: (Date.now() + 1).toString(),
                type: 'assistant',
                content: formatLLMResponse(response?.response || 'No response received'),
                timestamp: new Date(),
                engine: 'ollama',
              };
              setMessages(prev => [...prev, assistantMessage]);
            },
            onError: () => {
              handlePatternBasedQuery(queryText);
            },
          }
        );
      } else {
        handlePatternBasedQuery(queryText);
      }
    } catch (error) {
      console.error('Popup assistant error:', error);
    }
  };

  const handlePatternBasedQuery = (query: string) => {
    nlpMutation.mutate(query, {
      onSuccess: (response) => {
        const assistantMessage: PopupMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: formatLLMResponse(response?.response || 'No response received'),
          timestamp: new Date(),
          engine: 'pattern',
        };
        setMessages(prev => [...prev, assistantMessage]);
      },
      onError: (error) => {
        const errorMessage: PopupMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: `Error: ${error.message}`,
          timestamp: new Date(),
          engine: 'pattern',
        };
        setMessages(prev => [...prev, errorMessage]);
      },
    });
  };

  const handleExampleQuery = (query: string) => {
    setInputValue(query);
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }, 100);
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const exportChat = () => {
    const chatContent = messages.map(msg => 
      `[${msg.timestamp.toLocaleTimeString()}] ${msg.type.toUpperCase()}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([chatContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flight-ai-chat-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    setMessages([]);
  };

  const isLoading = nlpMutation.isPending || ollamaMutation.isPending;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-end p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/10"
        onClick={onClose}
      />
      
      {/* Popup Container */}
      <Card className={cn(
        "relative bg-white border border-gray-200 shadow-xl transition-all duration-300 ease-out",
        isMinimized 
          ? "w-80 h-16" 
          : "w-96 h-[600px]",
        "flex flex-col overflow-hidden rounded-xl"
      )}>
        {/* Header */}
        <CardHeader className="pb-3 bg-gray-50 border-b border-gray-200 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="p-2 bg-blue-500 rounded-lg">
                  <MessageSquare className="h-5 w-5 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold text-gray-900">AI Assistant</CardTitle>
                <p className="text-xs text-gray-600">
                  {contextData?.context.dataset_summary.total_flights || 0} flights analyzed
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {/* Engine Toggle */}
              {!isMinimized && (
                <Badge 
                  variant="outline" 
                  className={cn(
                    "text-xs cursor-pointer transition-colors",
                    useOllama 
                      ? "bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100" 
                      : "bg-orange-50 border-orange-200 text-orange-700 hover:bg-orange-100"
                  )}
                  onClick={() => setUseOllama(!useOllama)}
                >
                  {useOllama ? (
                    <>
                      <Brain className="h-3 w-3 mr-1" />
                      Ollama
                    </>
                  ) : (
                    <>
                      <Zap className="h-3 w-3 mr-1" />
                      Pattern
                    </>
                  )}
                </Badge>
              )}
              
              {/* Control Buttons */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsMinimized(!isMinimized)}
                className="h-8 w-8 p-0 text-gray-600 hover:bg-gray-100"
              >
                {isMinimized ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0 text-gray-600 hover:bg-gray-100"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        {!isMinimized && (
          <CardContent className="flex-1 flex flex-col p-0 min-h-0">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar min-h-0">
              {messages.length === 0 ? (
                // Welcome Screen
                <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
                  <div className="p-4 bg-blue-500 rounded-xl">
                    <Sparkles className="h-8 w-8 text-white" />
                  </div>
                  
                  <div className="space-y-2">
                    <h3 className="text-lg font-semibold text-gray-900">Welcome to AI Assistant!</h3>
                    <p className="text-sm text-gray-600 max-w-xs">
                      Ask me anything about your flight data, delays, or optimization strategies.
                    </p>
                  </div>

                  <div className="w-full space-y-4">
                    {Object.entries(exampleQueries).map(([category, queries]) => (
                      <div key={category} className="space-y-2">
                        <h4 className="text-xs font-medium text-gray-500 text-left">{category}</h4>
                        <div className="space-y-1">
                          {queries.map((query, index) => (
                            <button
                              key={index}
                              onClick={() => handleExampleQuery(query)}
                              className="w-full text-left p-3 text-sm bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg transition-colors text-gray-700"
                            >
                              {query}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                // Messages Display
                <div className="space-y-4">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={cn(
                        "flex gap-3 animate-fade-in",
                        message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
                      )}
                    >
                      {/* Avatar */}
                      <div className={cn(
                        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
                        message.type === 'user' 
                          ? 'bg-blue-500 text-white' 
                          : message.engine === 'ollama'
                            ? 'bg-purple-500 text-white'
                            : 'bg-orange-500 text-white'
                      )}>
                        {message.type === 'user' ? (
                          <User className="h-4 w-4" />
                        ) : message.engine === 'ollama' ? (
                          <Brain className="h-4 w-4" />
                        ) : (
                          <Zap className="h-4 w-4" />
                        )}
                      </div>
                      
                      {/* Message Bubble */}
                      <div className={cn(
                        "flex-1 max-w-[75%] group",
                        message.type === 'user' ? 'ml-12' : 'mr-12'
                      )}>
                        <div className={cn(
                          "p-3 rounded-lg relative",
                          message.type === 'user' 
                            ? 'bg-blue-500 text-white' 
                            : 'bg-gray-50 border border-gray-200 text-gray-800'
                        )}>
                          <div className="text-sm leading-relaxed whitespace-pre-wrap">
                            {message.content}
                          </div>
                          
                          {/* Message Actions */}
                          {message.type === 'assistant' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => copyMessage(message.content)}
                              className="absolute top-1 right-1 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <Copy className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                        
                        {/* Timestamp */}
                        <div className={cn(
                          "text-xs text-gray-500 mt-1 px-1",
                          message.type === 'user' ? 'text-right' : 'text-left'
                        )}>
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          {message.type === 'assistant' && (
                            <span className="ml-1 opacity-70">
                              via {message.engine === 'ollama' ? 'Ollama' : 'Pattern'}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* Typing Indicator */}
                  {isLoading && (
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded-full bg-gray-400 flex items-center justify-center">
                        <Bot className="h-4 w-4 text-white" />
                      </div>
                      <div className="flex-1 mr-12">
                        <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg">
                          <div className="flex items-center gap-2 text-gray-600">
                            <div className="flex gap-1">
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                            <span className="text-sm">AI is thinking...</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white flex-shrink-0 border-t border-gray-200">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Textarea
                    ref={textareaRef}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Type your question about flight data..."
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                    disabled={isLoading}
                    className="min-h-[44px] max-h-24 resize-none pr-16 border-gray-300 focus:border-blue-500 rounded-lg"
                  />
                  <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                    {inputValue.length}/500
                  </div>
                </div>
                
                <div className="flex flex-col gap-1">
                  <Button
                    onClick={handleSendMessage}
                    disabled={!inputValue.trim() || isLoading}
                    size="sm"
                    className="h-11 w-11 p-0 bg-blue-500 hover:bg-blue-600"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                  
                  {messages.length > 0 && (
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={exportChat}
                        className="h-6 w-6 p-0 text-gray-500 hover:text-blue-600"
                      >
                        <Download className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={clearChat}
                        className="h-6 w-6 p-0 text-gray-500 hover:text-red-600"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
};

export default PopupAssistant;
