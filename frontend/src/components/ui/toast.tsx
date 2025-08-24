import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { X } from 'lucide-react';

interface Toast { id: string; title?: string; message: string; type?: 'success'|'error'|'info'; timeout?: number; }
interface ToastContextValue { push: (t: Omit<Toast,'id'>) => void; }

const ToastContext = createContext<ToastContextValue | null>(null);

export const ToastProvider: React.FC<{children:ReactNode}> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const push = useCallback((t: Omit<Toast,'id'>) => {
    const id = Math.random().toString(36).slice(2);
    const toast: Toast = { id, timeout: 4000, type: 'info', ...t };
    setToasts(prev => [...prev, toast]);
    if (toast.timeout) setTimeout(()=> setToasts(p => p.filter(x=> x.id!==id)), toast.timeout);
  }, []);

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-[999] w-72">
        {toasts.map(t => {
          const border = t.type === 'success' ? 'border-green-500' : t.type === 'error' ? 'border-red-500' : 'border-slate-300';
          return (
            <div key={t.id} className={`rounded-md shadow border p-3 text-xs animate-in slide-in-from-bottom bg-white/95 backdrop-blur flex gap-2 ${border}`}>
              <div className="flex-1">
                {t.title && <div className="font-medium mb-0.5 text-slate-800">{t.title}</div>}
                <div className="leading-snug whitespace-pre-wrap text-slate-700">{t.message}</div>
              </div>
              <button onClick={()=> setToasts(p=> p.filter(x=> x.id!==t.id))} className="text-slate-400 hover:text-slate-700">
                <X className="h-4 w-4" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
};
