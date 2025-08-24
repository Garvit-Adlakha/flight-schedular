# Flight Schedule Optimization - React Frontend

A modern, responsive React TypeScript frontend for the Flight Schedule Optimization system, featuring real-time analytics, ML-powered insights, and AI chat capabilities.

## 🚀 Features

### 📊 **Analytics Dashboard**
- **Peak Hour Analysis** with interactive charts
- **Flight Statistics** overview with real-time metrics
- **Route Performance** analysis and recommendations
- **Congestion Visualization** with color-coded indicators

### 📁 **Data Management**
- **Drag & Drop File Upload** (Excel/CSV)
- **Real-time Validation** with detailed feedback
- **Processing History** tracking
- **Data Quality Scoring** with recommendations

### 🤖 **AI-Powered Chat**
- **Ollama LLM Integration** for advanced queries
- **Pattern-based NLP** as fallback
- **Context-aware Responses** using current dataset
- **Chat Export** functionality

### 🎨 **Modern UI/UX**
- **Material-UI Design System** with custom theme
- **Responsive Layout** for all screen sizes
- **Real-time Updates** with React Query
- **Accessibility Features** built-in

## 🛠️ Technology Stack

### **Core Framework**
- **React 19** with TypeScript for type safety
- **Material-UI (MUI)** for consistent design
- **React Query** for data fetching and caching

### **Data Visualization**
- **Recharts** for interactive charts
- **MUI Data Grid** for data tables
- **Custom Components** for specialized visualizations

### **State Management**
- **React Query** for server state
- **React Hooks** for local state
- **Context API** for global state

### **Development Tools**
- **TypeScript** for type safety
- **ESLint** for code quality
- **React DevTools** for debugging

## 📁 Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/        # React components
│   │   ├── Dashboard/     # Analytics components
│   │   ├── Upload/        # File upload components
│   │   ├── Chat/          # AI chat components
│   │   └── Layout/        # Layout components
│   ├── hooks/             # Custom React hooks
│   ├── services/          # API services
│   ├── types/             # TypeScript interfaces
│   ├── utils/             # Utility functions
│   ├── App.tsx            # Main app component
│   ├── index.tsx          # Entry point
│   └── index.css          # Global styles
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## 🔧 Installation & Setup

### **Prerequisites**
- Node.js 16+ 
- npm or yarn
- Backend API running on http://localhost:8000

### **Installation**
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### **Environment Configuration**
Create a `.env` file in the frontend root:
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_NAME=Flight Schedule Optimizer
REACT_APP_ENABLE_OLLAMA=true
```

## 🚀 Available Scripts

### **Development**
```bash
npm start                 # Start dev server (port 3000)
npm run start:dev        # Start with explicit API URL
npm test                 # Run tests
npm run type-check       # TypeScript type checking
npm run lint             # Run ESLint
npm run lint:fix         # Fix ESLint errors
```

### **Production**
```bash
npm run build            # Build for production
npm run build:prod      # Build with production API URL
npm run analyze         # Build and serve locally
```

## 📊 Component Overview

### **Dashboard Components**

#### **PeakHoursChart**
- Interactive bar chart showing hourly congestion
- Color-coded recommendations (avoid/caution/recommended)
- Real-time data from PostgreSQL analytics

#### **FlightStatistics**
- Key metrics overview (total flights, delays, on-time rate)
- Route and airport performance summaries
- System status indicators

### **Upload Components**

#### **FileUpload**
- Drag & drop interface for Excel/CSV files
- Real-time validation with error reporting
- Progress tracking and result feedback

### **Chat Components**

#### **LLMChat**
- Dual-mode chat (Ollama LLM + Pattern NLP)
- Context-aware responses
- Chat history export
- Example query suggestions

## 🔌 API Integration

### **React Query Setup**
```typescript
// Custom hooks for API calls
const { data, isLoading, error } = usePeakHours();
const uploadMutation = useUploadData();
```

### **API Service Layer**
```typescript
// Type-safe API calls
import * as api from './services/api';

const peakHours = await api.getPeakHours();
const result = await api.uploadFlightData(file);
```

### **Error Handling**
- Automatic retry logic
- Graceful degradation
- User-friendly error messages
- Offline capability detection

## 🎯 Best Practices Implemented

### **Performance**
- **Code Splitting** for faster load times
- **React Query Caching** for efficient data fetching
- **Lazy Loading** for non-critical components
- **Memoization** for expensive calculations

### **Accessibility**
- **ARIA Labels** for screen readers
- **Keyboard Navigation** support
- **Color Contrast** compliance
- **Focus Management** for modals/dialogs

### **Type Safety**
- **Strict TypeScript** configuration
- **API Type Definitions** for all endpoints
- **Component Prop Types** for better debugging
- **Generic Utilities** for reusable logic

### **Code Organization**
- **Feature-based Structure** for scalability
- **Custom Hooks** for business logic
- **Service Layer** for API abstraction
- **Consistent Naming** conventions

## 📱 Responsive Design

### **Breakpoints**
- **Mobile** (xs): 0-600px
- **Tablet** (sm): 600-960px
- **Desktop** (md): 960-1280px
- **Large** (lg): 1280-1920px
- **XL** (xl): 1920px+

### **Adaptive Features**
- **Collapsible Navigation** on mobile
- **Responsive Charts** with touch support
- **Flexible Layouts** using CSS Grid/Flexbox
- **Touch-friendly** button sizes

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time Notifications** for system updates
- **Advanced Chart Types** (network graphs, heatmaps)
- **Customizable Dashboards** with drag-drop widgets
- **Multi-language Support** with i18n
- **Dark Mode Theme** toggle
- **Progressive Web App** (PWA) capabilities

### **Performance Optimizations**
- **Virtual Scrolling** for large datasets
- **Service Worker** for offline functionality
- **Bundle Optimization** with webpack analysis
- **CDN Integration** for static assets

## 🛡️ Security Features

- **Input Sanitization** for all user inputs
- **File Type Validation** for uploads
- **XSS Protection** with Content Security Policy
- **HTTPS Enforcement** in production
- **API Rate Limiting** awareness

## 📈 Monitoring & Analytics

### **Performance Monitoring**
- **Web Vitals** tracking
- **React DevTools** integration
- **Bundle Size** analysis
- **Load Time** optimization

### **Error Tracking**
- **Error Boundaries** for component crashes
- **Console Error** monitoring
- **API Error** logging
- **User Experience** metrics

---

## 🌟 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Backend API**
   ```bash
   cd ../ml-model
   python api/start_api.py
   ```

3. **Start Frontend**
   ```bash
   npm start
   ```

4. **Open Browser**
   Navigate to http://localhost:3000

5. **Upload Data**
   Use the "Data Upload" tab to upload flight data

6. **Explore Analytics**
   View insights in the "Analytics Dashboard"

7. **Chat with AI**
   Ask questions in the "AI Assistant" tab

The system is now ready for flight schedule optimization! ✈️🚀