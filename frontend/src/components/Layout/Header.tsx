/**
 * Main Header Component with Navigation and Status
 */

import React from 'react';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { cn } from '../../lib/utils';
import {
  RefreshCw,
  Upload,
  BarChart3,
  Bot,
  Database,
  Plane,
  Activity,
} from 'lucide-react';
import { useHealthCheck, useCurrentContext, useRefreshData } from '../../hooks/useAPI';

interface HeaderProps {
  onUploadClick: () => void;
  onAnalyticsClick: () => void;
  onChatClick: () => void;
}

const Header: React.FC<HeaderProps> = ({
  onUploadClick,
  onAnalyticsClick,
  onChatClick,
}) => {
  const { data: healthData, isLoading: healthLoading } = useHealthCheck();
  const { data: contextData, isLoading: contextLoading } = useCurrentContext();
  const { refreshAll } = useRefreshData();

  const getHealthStatus = () => {
    if (healthLoading) return { color: 'warning', text: 'Checking...' };
    if (healthData?.status === 'healthy') return { color: 'success', text: 'Online' };
    return { color: 'error', text: 'Offline' };
  };

  const getModelStatus = () => {
    if (contextLoading || !contextData) return { total: 0, active: 0 };
    
    const models = contextData.models_status;
    const total = Object.keys(models).length;
    const active = Object.values(models).filter(Boolean).length;
    
    return { total, active };
  };

  const healthStatus = getHealthStatus();
  const modelStatus = getModelStatus();

  return (
    <header className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-lg border-b border-gray-200/50 shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center flex-1">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="p-3 gradient-primary rounded-2xl shadow-lg">
                  <Plane className="h-7 w-7 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">
                  Flight Optimizer
                </h1>
                <p className="text-sm text-muted-foreground hidden sm:block">
                  Schedule Analytics Platform
                </p>
              </div>
            </div>
            
            {/* Status Indicators */}
            <div className="ml-6 flex gap-2 hidden lg:flex">
              <Badge 
                variant="outline" 
                className={cn(
                  "text-xs flex items-center gap-1",
                  healthStatus.color === 'success' ? 'border-green-500 text-green-700 bg-green-50' :
                  healthStatus.color === 'warning' ? 'border-yellow-500 text-yellow-700 bg-yellow-50' :
                  'border-red-500 text-red-700 bg-red-50'
                )}
              >
                <Activity className="h-3 w-3" />
                API {healthStatus.text}
              </Badge>
              
              <Badge 
                variant="outline" 
                className={cn(
                  "text-xs",
                  modelStatus.active === modelStatus.total ? 'border-green-500 text-green-700 bg-green-50' : 'border-yellow-500 text-yellow-700 bg-yellow-50'
                )}
              >
                Models {modelStatus.active}/{modelStatus.total}
              </Badge>
              
              {contextData?.database_summary && (
                <Badge variant="outline" className="text-xs border-blue-500 text-blue-700 bg-blue-50">
                  <Database className="mr-1 h-3 w-3" />
                  {contextData.database_summary.total_records_in_db?.toLocaleString() || 0} records
                </Badge>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <div className="hidden md:flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={onUploadClick}
                className="text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
              >
                <Upload className="mr-2 h-4 w-4" />
                Upload
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={onAnalyticsClick}
                className="text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
              >
                <BarChart3 className="mr-2 h-4 w-4" />
                Analytics
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={onChatClick}
                className="text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
              >
                <Bot className="mr-2 h-4 w-4" />
                AI Chat
              </Button>
            </div>
            
            <Button
              variant="ghost"
              size="icon"
              onClick={refreshAll}
              disabled={healthLoading || contextLoading}
              className="text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
            >
              <RefreshCw className={cn(
                "h-4 w-4 transition-transform",
                (healthLoading || contextLoading) && "animate-spin"
              )} />
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
