/**
 * Enhanced LLM-Powered Chat Interface Component
 * Modern, responsive design with advanced UX features
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Switch } from '../ui/switch';
import { Separator } from '../ui/separator';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import { cn } from '../../lib/utils';
import {
  Send,
  Bot,
  User,
  RefreshCw,
  Copy,
  Download,
  Sparkles,
  Brain,
  CheckCircle,
  Clock,
  ThumbsUp,
  ThumbsDown,
  Zap,
  Mic,
  Square,
  Settings,
  MessageSquare,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';
import { useNLPQuery, useOllamaQuery, useCurrentContext } from '../../hooks/useAPI';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  engine: 'pattern' | 'ollama';
  metadata?: any;
  status?: 'sending' | 'sent' | 'delivered' | 'error';
  reaction?: 'like' | 'dislike' | null;
}

interface TypingState {
  isTyping: boolean;
  startTime?: Date;
}

const LLMChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [useOllama, setUseOllama] = useState(true);
  const [typing, setTyping] = useState<TypingState>({ isTyping: false });
  const [isRecording, setIsRecording] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const nlpMutation = useNLPQuery();
  const ollamaMutation = useOllamaQuery();
  const { data: contextData } = useCurrentContext();

  // Enhanced predefined example queries with categories
  const exampleQueries = {
    optimization: [
      "How can we optimize the 6 AM departure slot?",
      "What's the best time to schedule flights to Mumbai?",
      "How to minimize delays during peak hours?",
    ],
    analysis: [
      "What are the busiest hours to avoid?",
      "Which routes have the most delays?",
      "What causes cascading delays in the network?",
    ],
    prediction: [
      "Predict delays for tomorrow's morning flights",
      "What's the delay probability for BOM-DEL route?",
      "Which flights are at risk of cancellation?",
    ]
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Simulate typing effect for AI responses
  const simulateTyping = useCallback(() => {
    setTyping({ isTyping: true, startTime: new Date() });
    const typingDuration = Math.random() * 2000 + 1000; // 1-3 seconds
    setTimeout(() => {
      setTyping({ isTyping: false });
    }, typingDuration);
  }, []);

  // Handle message reactions
  const handleReaction = useCallback((messageId: string, reaction: 'like' | 'dislike') => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, reaction: msg.reaction === reaction ? null : reaction }
        : msg
    ));
  }, []);

  // Format LLM responses with better structure
  const formatLLMResponse = (content: string) => {
    // Handle undefined or null content
    if (!content || typeof content !== 'string') {
      return (
        <p className="text-sm leading-relaxed text-gray-700">
          No response content available.
        </p>
      );
    }
    
    const lines = content.split('\n');
    const elements: React.ReactNode[] = [];
    let currentSection: string[] = [];
    let sectionTitle = '';

    const flushSection = () => {
      if (currentSection.length > 0) {
        elements.push(
          <div key={elements.length} className="mb-4">
            {sectionTitle && (
              <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                <div className="w-1 h-4 bg-blue-500 rounded-full"></div>
                {sectionTitle}
              </h4>
            )}
            <div className="space-y-1 pl-3">
              {currentSection.map((line, idx) => (
                <p key={idx} className="text-sm leading-relaxed text-gray-700">
                  {line}
                </p>
              ))}
            </div>
          </div>
        );
        currentSection = [];
        sectionTitle = '';
      }
    };

    lines.forEach((line) => {
      const trimmed = line.trim();
      
      // Detect section headers (lines starting with ** or ending with **)
      if (trimmed.startsWith('**') && trimmed.endsWith('**') && trimmed.length > 4) {
        flushSection();
        sectionTitle = trimmed.replace(/\*\*/g, '').trim();
      }
      // Detect numbered lists
      else if (/^\d+\. /.test(trimmed)) {
        if (currentSection.length === 0 && !sectionTitle) {
          sectionTitle = 'Key Points';
        }
        currentSection.push(trimmed);
      }
      // Detect bullet points
      else if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
        if (currentSection.length === 0 && !sectionTitle) {
          sectionTitle = 'Details';
        }
        currentSection.push('• ' + trimmed.substring(2));
      }
      // Regular content
      else if (trimmed.length > 0) {
        currentSection.push(trimmed);
      }
      // Empty line - flush current section
      else if (currentSection.length > 0) {
        flushSection();
      }
    });

    // Flush any remaining content
    flushSection();

    return elements.length > 0 ? elements : (
      <p className="text-sm leading-relaxed text-gray-700 whitespace-pre-wrap">
        {content}
      </p>
    );
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
      engine: useOllama ? 'ollama' : 'pattern',
      status: 'sending',
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    simulateTyping();

    try {
      if (useOllama && contextData?.models_status.ollama_llm) {
        // Use Ollama LLM
        ollamaMutation.mutate(
          { 
            query: inputValue, 
            context: contextData?.context 
          },
          {
            onSuccess: (response) => {
              // Update user message status
              setMessages(prev => prev.map(msg => 
                msg.id === userMessage.id 
                  ? { ...msg, status: 'delivered' }
                  : msg
              ));
              
              const assistantMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                type: 'assistant',
                content: response?.response || 'No response received from Ollama LLM',
                timestamp: new Date(),
                engine: 'ollama',
                status: 'delivered',
                metadata: {
                  processingTime: response?.processing_time,
                  confidence: response?.confidence,
                  contextUsed: response?.context_used,
                },
              };
              setMessages(prev => [...prev, assistantMessage]);
              setTyping({ isTyping: false });
            },
            onError: (error) => {
              // Update user message status to error
              setMessages(prev => prev.map(msg => 
                msg.id === userMessage.id 
                  ? { ...msg, status: 'error' }
                  : msg
              ));
              
              const errorMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                type: 'assistant',
                content: `I encountered an error with the Ollama LLM: ${error.message}. Falling back to pattern-based analysis.`,
                timestamp: new Date(),
                engine: 'pattern',
                status: 'delivered',
              };
              setMessages(prev => [...prev, errorMessage]);
              setTyping({ isTyping: false });
              
              // Fallback to pattern-based NLP
              handlePatternBasedQuery(userMessage.content);
            },
          }
        );
      } else {
        // Use pattern-based NLP
        handlePatternBasedQuery(inputValue);
      }
    } catch (error) {
      console.error('Chat error:', error);
    }
  };

  const handlePatternBasedQuery = (query: string) => {
    nlpMutation.mutate(query, {
      onSuccess: (response) => {
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: response?.response || 'No response received from pattern-based NLP',
          timestamp: new Date(),
          engine: 'pattern',
          status: 'delivered',
          metadata: {
            responseType: response?.response_type,
            confidence: response?.confidence,
            dataUsed: response?.data_used,
            processingTime: response?.processing_time,
            suggestions: response?.suggestions,
          },
        };
        setMessages(prev => [...prev, assistantMessage]);
        setTyping({ isTyping: false });
      },
      onError: (error) => {
        const errorMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: `I'm sorry, I encountered an error: ${error.message}. Please try again.`,
          timestamp: new Date(),
          engine: 'pattern',
          status: 'error',
        };
        setMessages(prev => [...prev, errorMessage]);
        setTyping({ isTyping: false });
      },
    });
  };

  const handleExampleQuery = (query: string) => {
    setInputValue(query);
    // Auto-send after a brief delay for better UX
    setTimeout(() => {
      if (query.trim()) {
        handleSendMessage();
      }
    }, 300);
  };

  // Voice recording simulation (placeholder for future implementation)
  const toggleRecording = () => {
    setIsRecording(!isRecording);
    // Placeholder for voice recording functionality
    setTimeout(() => {
      if (isRecording) {
        setInputValue("Voice input: How can we optimize flight schedules?");
      }
      setIsRecording(false);
    }, 2000);
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const exportChat = () => {
    const chatData = {
      messages,
      exportDate: new Date().toISOString(),
      context: contextData?.context,
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `flight-chat-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    setMessages([]);
  };

  const isLoading = nlpMutation.isPending || ollamaMutation.isPending;

  return (
    <div className="max-w-7xl mx-auto h-screen flex flex-col p-4">
      <Card className="flex-1 flex flex-col bg-white border border-gray-200 shadow-lg rounded-xl max-h-[80vh]">
        <CardHeader className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-xl p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
                <Bot className="h-6 w-6" />
              </div>
              <div>
                <CardTitle className="text-lg font-bold">AI Flight Schedule Assistant</CardTitle>
                <p className="text-white/80 text-sm">Powered by advanced ML models</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 bg-white/20 rounded-lg px-3 py-1.5 backdrop-blur-sm">
                <div className="flex items-center gap-1.5">
                  {useOllama ? <Sparkles className="h-4 w-4" /> : <Brain className="h-4 w-4" />}
                  <span className="text-sm font-medium">
                    {useOllama ? 'Ollama LLM' : 'Pattern NLP'}
                  </span>
                </div>
                <Switch
                  checked={useOllama}
                  onCheckedChange={setUseOllama}
                  disabled={!contextData?.models_status.ollama_llm}
                  className="scale-75"
                />
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={clearChat}
                disabled={messages.length === 0}
                className="h-8 w-8 p-0 hover:bg-white/20 text-white rounded-lg"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col p-0 min-h-0">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar" style={{ maxHeight: 'calc(100vh - 250px)' }}>
            {messages.length === 0 && (
              <div className="text-center py-6 animate-fade-in">
                <div className="relative mb-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl mx-auto flex items-center justify-center shadow-md">
                    <MessageSquare className="h-8 w-8 text-white" />
                  </div>
                </div>
                
                <h3 className="text-xl font-bold text-gray-800 mb-2">
                  Welcome to AI Flight Assistant! ✈️
                </h3>
                <p className="text-gray-600 mb-4 max-w-3xl mx-auto text-sm">
                  I'm here to help you optimize flight schedules, analyze delays, predict trends, and provide data-driven insights.
                </p>
                
                {/* Compact Example Queries */}
                <div className="max-w-5xl mx-auto">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                    {Object.entries(exampleQueries).map(([category, queries]) => (
                      <div key={category} className="text-center p-3 bg-gray-50 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-center gap-1.5 mb-2">
                          {category === 'optimization' && <TrendingUp className="h-4 w-4 text-green-600" />}
                          {category === 'analysis' && <Brain className="h-4 w-4 text-blue-600" />}
                          {category === 'prediction' && <Sparkles className="h-4 w-4 text-purple-600" />}
                          <h4 className="text-sm font-semibold text-gray-700 capitalize">
                            {category}
                          </h4>
                        </div>
                        <div className="space-y-1.5">
                          {queries.map((query, index) => (
                            <Button
                              key={index}
                              variant="outline"
                              onClick={() => handleExampleQuery(query)}
                              className="w-full h-8 text-xs justify-start px-2 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700 transition-all duration-200"
                            >
                              {query}
                            </Button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {messages.map((message, index) => (
                              <div
                key={message.id}
                className={cn(
                  "group flex gap-3 mb-4 transition-all duration-200 hover:scale-[1.005] animate-slide-up",
                  message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
                )}
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                {/* Compact Avatar */}
                <div className={cn(
                  "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-md transition-all duration-200",
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
              
                {/* Enhanced Message Container */}
                <div className={cn(
                  "flex-1 max-w-[85%]",
                  message.type === 'user' ? 'ml-0' : 'mr-0'
                )}>
                  {/* Compact Message Bubble */}
                  <div className={cn(
                    "rounded-lg p-3 shadow-sm border transition-all duration-200 hover:shadow-md",
                    message.type === 'user' 
                      ? 'bg-blue-50 border-blue-200 text-blue-900' 
                      : 'bg-white border-gray-200 text-gray-900',
                    message.status === 'error' && 'border-red-200 bg-red-50'
                  )}>
                    {/* Compact Message Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          "text-xs font-semibold",
                          message.type === 'user' ? 'text-blue-700' : 'text-gray-700'
                        )}>
                          {message.type === 'user' ? 'You' : 'AI Assistant'}
                        </span>
                        {message.type === 'assistant' && (
                          <Badge 
                            variant="outline" 
                            className={cn(
                              "text-xs px-1.5 py-0.5 font-medium border-0 text-xs",
                              message.engine === 'ollama' 
                                ? 'bg-purple-100 text-purple-700' 
                                : 'bg-orange-100 text-orange-700'
                            )}
                          >
                            {message.engine === 'ollama' ? 'Ollama LLM' : 'Pattern NLP'}
                          </Badge>
                        )}
                      </div>
                      <span className="text-xs text-gray-400">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>

                    {/* Clean Message Content */}
                    <div className="text-sm leading-relaxed">
                      {message.type === 'assistant' ? (
                        <div className="space-y-3">
                          {formatLLMResponse(message.content || '')}
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap">
                          {message.content || 'Message content not available'}
                        </p>
                      )}
                    </div>
                    {/* Enhanced Message Footer */}
                    <div className="flex justify-between items-center mt-5 pt-4 border-t border-gray-200">
                      <div className="flex gap-2 items-center flex-wrap">
                        {message.metadata?.confidence && (
                          <Badge variant="outline" className="text-xs h-6 bg-green-50 border-green-400 text-green-800 font-medium">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            {(message.metadata.confidence * 100).toFixed(0)}% confident
                          </Badge>
                        )}
                        {message.metadata?.processingTime && (
                          <Badge variant="outline" className="text-xs h-6 bg-blue-50 border-blue-400 text-blue-800">
                            ⚡ {message.metadata.processingTime}ms
                          </Badge>
                        )}
                        {message.metadata?.responseType && (
                          <Badge variant="outline" className="text-xs h-6 bg-gray-50 border-gray-400 text-gray-700">
                            📊 {message.metadata.responseType}
                          </Badge>
                        )}
                      </div>
                      
                      <div className="flex items-center gap-1">
                        {message.type === 'assistant' && (
                          <>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleReaction(message.id, 'like')}
                                    className={cn(
                                      "h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-all duration-200 hover:bg-green-100",
                                      message.reaction === 'like' && "opacity-100 bg-green-100 text-green-600"
                                    )}
                                  >
                                    <ThumbsUp className="h-4 w-4" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Helpful</TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                            
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleReaction(message.id, 'dislike')}
                                    className={cn(
                                      "h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-all duration-200 hover:bg-red-100",
                                      message.reaction === 'dislike' && "opacity-100 bg-red-100 text-red-600"
                                    )}
                                  >
                                    <ThumbsDown className="h-4 w-4" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Not helpful</TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </>
                        )}
                        
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => copyMessage(message.content)}
                                className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-all duration-200 hover:bg-gray-100"
                              >
                                <Copy className="h-4 w-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Copy message</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                    </div>
                  </div>

                  {/* Simple Suggestions */}
                  {message.metadata?.suggestions && message.metadata.suggestions.length > 0 && (
                    <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-sm font-medium text-gray-700 mb-2">
                        💡 Follow-up suggestions:
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {message.metadata.suggestions.map((suggestion: string, index: number) => (
                          <Button
                            key={index}
                            variant="outline"
                            size="sm"
                            className="h-7 text-xs bg-white hover:bg-blue-50 border-gray-300 text-gray-700 transition-all duration-200"
                            onClick={() => handleExampleQuery(suggestion)}
                          >
                            {suggestion}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Simple Typing Indicator */}
            {(isLoading || typing.isTyping) && (
              <div className="flex gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
                  <Bot className="h-5 w-5 text-purple-600" />
                </div>
                <div className="bg-white rounded-lg p-3 shadow-sm border border-gray-200 max-w-xs">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text-gray-700">AI Assistant</span>
                    <Badge variant="outline" className="text-xs bg-purple-50 text-purple-700 border-0">
                      {useOllama ? 'Ollama LLM' : 'Pattern NLP'}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-xs text-gray-500">
                      {useOllama ? 'Thinking...' : 'Analyzing...'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <Separator className="bg-gradient-to-r from-transparent via-gray-300 to-transparent" />

          {/* Compact Input Area */}
          <div className="p-4 bg-gray-50 flex-shrink-0 border-t border-gray-200">
            {!contextData?.models_status.ollama_llm && useOllama && (
              <Alert className="border-yellow-400 bg-yellow-50 mb-4 shadow-sm">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <AlertDescription className="text-yellow-800 font-medium">
                  Ollama LLM is not available. Using pattern-based NLP instead.
                </AlertDescription>
              </Alert>
            )}
            
            <div className="flex gap-3 items-end">
              <div className="flex-1 relative">
                <Textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="✈️ Ask about flight schedules, delays, optimization, peak hours..."
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  disabled={isLoading || typing.isTyping}
                  className="min-h-[48px] max-h-32 resize-none border border-gray-300 focus:border-blue-500 rounded-lg shadow-sm bg-white text-gray-800 placeholder:text-gray-500 transition-all duration-200 text-sm"
                />
                {inputValue.length > 0 && (
                  <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                    {inputValue.length}/500
                  </div>
                )}
              </div>
              
              {/* Send Button */}
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading || typing.isTyping}
                size="default"
                className={cn(
                  "h-[48px] px-6 rounded-lg transition-all duration-200 shadow-sm text-sm font-medium",
                  !inputValue.trim() || isLoading || typing.isTyping
                    ? "bg-gray-300 cursor-not-allowed"
                    : "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 hover:scale-105"
                )}
              >
                <Send className="h-4 w-4 text-white mr-2" />
                Send
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LLMChat;
