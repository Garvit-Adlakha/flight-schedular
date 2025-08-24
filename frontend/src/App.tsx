/**
 * Main App Component with shadcn/ui and React Query
 */

import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Alert, AlertDescription } from './components/ui/alert';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { RefreshCw, BarChart3, Upload, Bot, Plane, MessageSquare } from 'lucide-react';

// Components
import PeakHoursChart from './components/Dashboard/PeakHoursChart';
import FlightStatistics from './components/Dashboard/FlightStatistics';
import EnrichmentPanel from './components/Dashboard/EnrichmentPanel';
import CascadeNetworkGraph from './components/Dashboard/CascadeNetworkGraph';
// FileUpload replaced by DataBrowser in upload tab
import DataBrowser from './components/Data/DataBrowser';
import LLMChat from './components/Chat/LLMChat';
import PopupAssistant from './components/Chat/PopupAssistant';
import ProjectAnalysis from './components/Chat/ProjectAnalysis';
import ScheduleTuningSimulator from './components/Dashboard/ScheduleTuningSimulator';


import { useHealthCheck, useRefreshData } from './hooks/useAPI';
import { ToastProvider } from './components/ui/toast';
import { useToast } from './components/ui/toast';



// Create Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (replaces cacheTime)
    },
  },
});



function AppContent() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);
  const { data: healthData, error: healthError } = useHealthCheck();
  const { refreshAll } = useRefreshData();
  const { push } = useToast();



  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-background via-background to-muted/30">
      {/* Main Header with Navigation */}
      <header className="sticky top-0 z-50 w-full bg-white/95 backdrop-blur-lg border-b border-gray-200/50 shadow-sm">
        <div className="container mx-auto px-3 sm:px-4">
          <div className="flex h-14 sm:h-16 items-center justify-between gap-2 sm:gap-4">
            {/* Logo and Title */}
            <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
              <div className="relative">
                <div className="p-2 sm:p-2.5 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg sm:rounded-xl shadow-lg">
                  <Plane className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                </div>
                <div className="absolute -top-0.5 -right-0.5 sm:-top-1 sm:-right-1 w-2.5 h-2.5 sm:w-3 sm:h-3 bg-green-500 rounded-full border border-white sm:border-2"></div>
              </div>
              <div className="hidden min-[480px]:block">
                <h1 className="text-base sm:text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent leading-tight">
                  Flight Optimizer
                </h1>
                <p className="text-xs text-muted-foreground hidden sm:block leading-tight">
                  Schedule Analytics Platform
                </p>
              </div>
            </div>
            
            {/* Navigation Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex justify-center max-w-sm lg:max-w-md">
              <TabsList className="grid w-full grid-cols-4 bg-gray-100/80 backdrop-blur-sm border h-10 rounded-lg text-xs">
                <TabsTrigger 
                  value="dashboard" 
                  className="flex items-center gap-1 text-xs font-medium data-[state=active]:bg-white data-[state=active]:text-blue-600 data-[state=active]:shadow-sm transition-all duration-200 rounded-md"
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span className="hidden lg:inline">Analytics</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="project" 
                  className="flex items-center gap-1 text-xs font-medium data-[state=active]:bg-white data-[state=active]:text-blue-600 data-[state=active]:shadow-sm transition-all duration-200 rounded-md"
                >
                  <Plane className="h-3.5 w-3.5" />
                  <span className="hidden lg:inline">Project</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="data" 
                  className="flex items-center gap-1 text-xs font-medium data-[state=active]:bg-white data-[state=active]:text-blue-600 data-[state=active]:shadow-sm transition-all duration-200 rounded-md"
                >
                  <Upload className="h-3.5 w-3.5" />
                  <span className="hidden lg:inline">Data</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="chat" 
                  className="flex items-center gap-1 text-xs font-medium data-[state=active]:bg-white data-[state=active]:text-blue-600 data-[state=active]:shadow-sm transition-all duration-200 rounded-md"
                >
                  <Bot className="h-3.5 w-3.5" />
                  <span className="hidden lg:inline">AI Chat</span>
                </TabsTrigger>
              </TabsList>
            </Tabs>

            {/* Status Indicators and Actions */}
            <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
              {/* Status Badges */}
              <div className="hidden md:flex gap-2">
                {healthData?.status && (
                  <Badge 
                    variant="outline" 
                    className="text-xs border-green-500 text-green-700 bg-green-50 px-2 py-1"
                  >
                    <div className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse"></div>
                    <span className="hidden lg:inline">API Online</span>
                    <span className="lg:hidden">Online</span>
                  </Badge>
                )}
                <Badge 
                  variant="outline" 
                  className="text-xs border-blue-500 text-blue-700 bg-blue-50 px-2 py-1"
                >
                  <span className="hidden lg:inline">Models 4/4</span>
                  <span className="lg:hidden">4/4</span>
                </Badge>
              </div>
              
              {/* Mobile Status Indicator */}
              <div className="md:hidden">
                {healthData?.status && (
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                )}
              </div>
              
              {/* Refresh Button */}
              <Button
                variant="ghost"
                size="icon"
                onClick={refreshAll}
                className="h-8 w-8 sm:h-9 sm:w-9 text-gray-600 hover:text-blue-600 hover:bg-blue-50 transition-colors flex-shrink-0"
              >
                <RefreshCw className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Health Status Alert */}
      {healthError && (
        <div className="mx-4 mt-4 animate-slide-up">
          <Alert className="border-destructive/50 bg-destructive/10 backdrop-blur-sm">
            <AlertDescription className="flex items-center gap-2">
              <div className="status-indicator status-offline"></div>
              <div>
                <strong>Backend Connection Failed:</strong> {healthError.message}
                <br />
                <span className="text-sm text-muted-foreground">
                  Make sure the API server is running on http://localhost:8000
                </span>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      )}

      <div className="container mx-auto px-4 py-6 flex-1 flex flex-col min-h-0">
        {/* Tab Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full h-full flex flex-col">
          <div className="flex-1 min-h-0">
            <TabsContent value="dashboard" className="space-y-8 animate-fade-in h-full overflow-auto">
            <div className="space-y-8">
              <div className="text-center space-y-4">
                <div className="flex items-center justify-center gap-3 mb-4">
                  <div className="p-3 bg-primary/10 rounded-full">
                    <Plane className="h-8 w-8 text-primary" />
                  </div>
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                  Flight Schedule Analytics
                </h1>
                <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                  Comprehensive insights into flight operations, delays, and optimization opportunities
                </p>
                <div className="flex justify-center gap-3 pt-2">
                  <Button variant="outline" size="sm" onClick={async ()=> { 
                    try { const m = await import('./services/api'); await m.downloadPDFReport(); push({type:'success', title:'Report', message:'PDF downloaded'}); }
                    catch(e:any){ push({type:'error', title:'PDF Failed', message:e.message||'Download failed'}); }
                  }}>Download PDF Report</Button>
                </div>
              </div>
              
              {/* Statistics Overview */}
              <div className="animate-slide-up" style={{ animationDelay: '0.1s' }}>
                <FlightStatistics />
              </div>
              
              {/* Peak Hours Analysis */}
              <div className="animate-slide-up" style={{ animationDelay: '0.2s' }}>
                <PeakHoursChart />
              </div>

              <div className="animate-slide-up" style={{ animationDelay: '0.3s' }}>
                <EnrichmentPanel />
              </div>
              <div className="animate-slide-up" style={{ animationDelay: '0.4s' }}>
                <CascadeNetworkGraph />
              </div>
              

            </div>
          </TabsContent>

          <TabsContent value="project" className="space-y-8 animate-fade-in h-full overflow-auto">
            <div className="space-y-8">
              <div className="text-center space-y-4">
                <div className="flex items-center justify-center gap-3 mb-4">
                  <div className="p-3 bg-primary/10 rounded-full">
                    <Plane className="h-8 w-8 text-primary" />
                  </div>
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                  Project Analysis
                </h1>
                <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                  Comprehensive flight schedule optimization using open-source AI tools
                </p>
              </div>
              
              <div className="animate-slide-up">
                <ProjectAnalysis />
              </div>
              <div className="animate-slide-up" style={{ animationDelay: '0.1s' }}>
                <ScheduleTuningSimulator />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="data" className="space-y-8 animate-fade-in h-full overflow-auto">
            <div className="space-y-8">
              <div className="text-center space-y-4">
                <div className="flex items-center justify-center gap-3 mb-2">
                  <div className="p-3 bg-primary/10 rounded-full">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                </div>
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                  Dataset
                </h1>
                <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                  Browse existing flight records and upload/replace data
                </p>
              </div>
              <div className="animate-slide-up">
                <DataBrowser />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="chat" className="animate-fade-in h-full">
            <LLMChat />
            </TabsContent>
          </div>
        </Tabs>
      </div>

      {/* AI Assistant Floating Button */}
      <Button
        onClick={() => setIsAssistantOpen(true)}
        className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 h-12 w-12 sm:h-14 sm:w-14 z-40 rounded-full shadow-lg hover:shadow-xl transition-all duration-200 bg-blue-500 hover:bg-blue-600"
        size="icon"
      >
        <MessageSquare className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white" />
      </Button>

      {/* Popup AI Assistant */}
      <PopupAssistant 
        isOpen={isAssistantOpen} 
        onClose={() => setIsAssistantOpen(false)} 
      />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <AppContent />
      </ToastProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;